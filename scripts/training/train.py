from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import math

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


class GradientTracker:
    """Track gradient statistics for variance analysis"""
    
    def __init__(self, model):
        self.model = model
        self.gradients = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register gradient hooks on all parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(grad, name)
                )
                self.hooks.append(hook)
    
    def _gradient_hook(self, grad, name):
        """Hook function to capture gradients"""
        if grad is not None:
            self.gradients.append({
                'name': name,
                'grad': grad.detach().clone(),
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'var': grad.var().item()
            })
    
    def get_gradient_stats(self):
        """Compute comprehensive gradient statistics"""
        if not self.gradients:
            return {}
        
        # Overall gradient statistics
        all_norms = [g['norm'] for g in self.gradients]
        all_means = [g['mean'] for g in self.gradients]
        all_stds = [g['std'] for g in self.gradients]
        all_vars = [g['var'] for g in self.gradients]
        
        stats = {
            'grad_norm_mean': np.mean(all_norms),
            'grad_norm_std': np.std(all_norms),
            'grad_norm_max': np.max(all_norms),
            'grad_norm_min': np.min(all_norms),
            'grad_mean_mean': np.mean(all_means),
            'grad_mean_std': np.std(all_means),
            'grad_std_mean': np.mean(all_stds),
            'grad_std_std': np.std(all_stds),
            'grad_var_mean': np.mean(all_vars),
            'grad_var_std': np.std(all_vars),
            'num_params': len(self.gradients)
        }
        
        # Layer-wise statistics (for key layers)
        layer_stats = {}
        for grad_info in self.gradients:
            name = grad_info['name']
            # Group by layer type
            if 'conv' in name.lower():
                layer_type = 'conv'
            elif 'linear' in name.lower() or 'fc' in name.lower():
                layer_type = 'linear'
            elif 'norm' in name.lower():
                layer_type = 'norm'
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_type = 'attention'
            else:
                layer_type = 'other'
            
            if layer_type not in layer_stats:
                layer_stats[layer_type] = []
            layer_stats[layer_type].append(grad_info)
        
        # Compute layer-wise statistics
        for layer_type, grads in layer_stats.items():
            if grads:
                norms = [g['norm'] for g in grads]
                vars = [g['var'] for g in grads]
                stats[f'{layer_type}_grad_norm_mean'] = np.mean(norms)
                stats[f'{layer_type}_grad_norm_std'] = np.std(norms)
                stats[f'{layer_type}_grad_var_mean'] = np.mean(vars)
                stats[f'{layer_type}_grad_var_std'] = np.std(vars)
                stats[f'{layer_type}_num_params'] = len(grads)
        
        return stats
    
    def clear(self):
        """Clear stored gradients"""
        self.gradients.clear()
    
    def remove_hooks(self):
        """Remove all gradient hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    
    # Handle reward function with optional kwargs
    if hasattr(config, 'reward_fn_kwargs') and config.reward_fn_kwargs:
        reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)(**config.reward_fn_kwargs)
    else:
        reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()
    
    # prepare critic fn if using actor-critic method
    critic_fn = None
    if config.use_actor_critic:
        if hasattr(config, 'critic_fn_kwargs') and config.critic_fn_kwargs:
            critic_fn = getattr(ddpo_pytorch.rewards, config.critic_fn)(**config.critic_fn_kwargs)
        else:
            critic_fn = getattr(ddpo_pytorch.rewards, config.critic_fn)()
        # Move critic to device but keep it in float32 (CLIP needs float32)
        if hasattr(critic_fn, 'critic'):
            critic_fn.critic.to(accelerator.device, dtype=torch.float32)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Initialize gradient tracker for variance analysis
    gradient_tracker = GradientTracker(unet)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[
                    prompt_fn(**config.prompt_fn_kwargs)
                    for _ in range(config.sample.batch_size)
                ]
            )

            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            # sample
            with autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            # Check if reward function accepts timesteps (for critic-based rewards)
            if 'timestep' in reward_fn.__code__.co_varnames or hasattr(reward_fn, '__wrapped__'):
                # For critic-based rewards, compute on final denoised images (timestep=0)
                final_timesteps = torch.zeros(images.shape[0], device=accelerator.device)
                rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, final_timesteps)
            else:
                # For regular rewards, no timestep needed
                rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # yield to to make sure reward computation starts
            time.sleep(0)

            # decode intermediate latents for critic if using actor-critic method
            decoded_images = None
            if config.use_actor_critic and critic_fn is not None:
                # Store latents for on-demand decoding to save memory
                # We'll decode timesteps one by one during advantage computation
                decoded_images = latents  # Store latents instead of decoded images

            sample_dict = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[
                    :, :-1
                ],  # each entry is the latent before timestep t
                "next_latents": latents[
                    :, 1:
                ],  # each entry is the latent after timestep t
                "log_probs": log_probs,
                "rewards": rewards,
            }
            
            # Only include decoded_images if using actor-critic method
            if config.use_actor_critic and critic_fn is not None:
                sample_dict["decoded_images"] = decoded_images
            
            samples.append(sample_dict)

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(prompts, rewards)
                        )  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        # compute advantages using critic if using actor-critic method
        if config.use_actor_critic and critic_fn is not None and "decoded_images" in samples:
            # Use critic to compute advantages
            # For each timestep, compute advantage as reward - value_prediction
            advantages = []
            
            # Get latents and timesteps for critic
            latents = samples["decoded_images"]  # (batch_size, num_steps + 1, 4, 64, 64) - stored as latents
            timesteps = samples["timesteps"]  # (batch_size, num_steps)
            
            # Compute advantages for each timestep
            for t in range(timesteps.shape[1]):  # for each timestep
                # Decode latents for this timestep on-demand to save memory
                step_latents = latents[:, t]  # (batch_size, 4, 64, 64)
                step_timesteps = timesteps[:, t]  # (batch_size,)
                
                # Decode latents to images for this timestep only
                with torch.no_grad():
                    decoded_step = pipeline.vae.decode(step_latents / pipeline.vae.config.scaling_factor).sample
                    step_images = (decoded_step + 1.0) / 2.0  # Normalize to [0, 1]
                    step_images = torch.clamp(step_images, 0.0, 1.0)
                
                # Get value predictions from critic
                values, _ = critic_fn(step_images, step_timesteps, prompts, prompt_metadata)
                values = torch.as_tensor(values, device=accelerator.device)
                
                # Compute advantages: for timestep t, advantage = terminal_reward - value_t
                # The terminal reward is the same for all timesteps in a trajectory
                terminal_rewards = torch.as_tensor(rewards, device=accelerator.device)
                step_advantages = terminal_rewards - values
                advantages.append(step_advantages)
                
                # Clean up decoded images for this timestep
                del step_images, decoded_step
                torch.cuda.empty_cache()
            
            advantages = torch.stack(advantages, dim=1)
            
            # Apply time weighting: A_t = A_t * exp(beta * t/T)
            # Normalize advantages if enabled: A^ = (A - mean(A)) / (std(A) + ε)
            if config.train.normalize_advantages:
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                epsilon = 1e-8
                advantages = (advantages - adv_mean) / (adv_std + epsilon)

            if config.train.time_weighting_beta > 0:
                # timesteps shape: (batch_size, num_steps)
                # We need to normalize timesteps to [0, 1] range where 0 = start (high noise), 1 = end (low noise)
                num_steps = timesteps.shape[1]
                
                # Normalize timesteps to [0, 1] range
                # timesteps are typically in [0, 999] range, we need [0, 1]
                # For diffusion, higher timestep = more noise, so we want to invert this
                # t_normalized = (max_timestep - t) / max_timestep
                max_timestep = timesteps.max()
                min_timestep = timesteps.min()
                
                # Normalize so that t=0 (high noise) -> 0, t=max (low noise) -> 1
                time_ratios = (max_timestep - timesteps) / (max_timestep - min_timestep)
                
                # Apply exponential weighting
                time_weights = torch.exp(config.train.time_weighting_beta * time_ratios)
                
                
                # Apply time weighting: (batch_size, num_steps) * (batch_size, num_steps)
                advantages = advantages * time_weights
        else:
            # Use traditional advantage computation
            if config.per_prompt_stat_tracking:
                # gather the prompts across processes
                prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                prompts = pipeline.tokenizer.batch_decode(
                    prompt_ids, skip_special_tokens=True
                )
                advantages = stat_tracker.update(prompts, rewards)
            else:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Expand advantages to all timesteps (same advantage for all timesteps in a trajectory)
            num_timesteps = samples["timesteps"].shape[1]
            advantages = torch.as_tensor(advantages, device=accelerator.device).unsqueeze(1).repeat(1, num_timesteps)

        # Store advantages in samples
        samples["advantages"] = advantages

        del samples["rewards"]
        del samples["prompt_ids"]
        if "decoded_images" in samples:
            del samples["decoded_images"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs", "advantages"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + config.sample.guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        # Get advantages for this specific timestep
                        if sample["advantages"].dim() == 2:
                            # AC method: advantages shape is (batch_size, num_timesteps)
                            step_advantages = sample["advantages"][:, j]
                        else:
                            # Traditional method: advantages shape is (batch_size,)
                            step_advantages = sample["advantages"]
                        
                        advantages = torch.clamp(
                            step_advantages,
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), config.train.max_grad_norm
                            )
                            
                            # Collect gradient statistics for variance analysis
                            grad_stats = gradient_tracker.get_gradient_stats()
                            if grad_stats:
                                # Add method indicator to distinguish AC vs traditional
                                method = "AC" if config.use_actor_critic else "Traditional"
                                # Compute advantage statistics safely
                                adv_mean = advantages.mean().item()
                                adv_min = advantages.min().item()
                                adv_max = advantages.max().item()
                                
                                # Handle std/var computation for single samples
                                if advantages.numel() > 1:
                                    adv_std = advantages.std().item()
                                    adv_var = advantages.var().item()
                                else:
                                    adv_std = 0.0
                                    adv_var = 0.0
                                
                                grad_stats.update({
                                    'method': method,
                                    'timestep': j,
                                    'batch_idx': i,
                                    'advantage_mean': adv_mean,
                                    'advantage_std': adv_std,
                                    'advantage_var': adv_var,
                                    'advantage_min': adv_min,
                                    'advantage_max': adv_max,
                                })
                                # Convert all gradient stats to tensors for consistent logging
                                for key, value in grad_stats.items():
                                    if isinstance(value, (float, np.number)):
                                        info[key].append(torch.tensor(value, device=accelerator.device))
                            
                            gradient_tracker.clear()  # Clear for next iteration
                            
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()

    # Cleanup gradient tracker
    gradient_tracker.remove_hooks()


if __name__ == "__main__":
    app.run(main)
