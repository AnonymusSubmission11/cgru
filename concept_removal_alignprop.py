# concept_removal_alignprop.py
# Reproduce "Concept Removal" with AlignProp (TRL) + OWL-ViT reward.
# The reward penalizes detections of "book(s)" => model learns to avoid rendering them.

import math
from peft import LoraConfig
from pathlib import Path

import random
from dataclasses import dataclass
from typing import Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Compatibility shim for huggingface_hub cached_download removal ---
# Some versions of TRL import `cached_download` which was removed from
# newer huggingface_hub. Provide a fallback to `hf_hub_download`.
try:
    import huggingface_hub as _hf
    if not hasattr(_hf, "cached_download"):
        from huggingface_hub import hf_hub_download as _hf_hub_download
        _hf.cached_download = _hf_hub_download  # make `from huggingface_hub import cached_download` succeed
except Exception:
    # If anything goes wrong here, we'll let the normal import error surface later.
    pass

# --- TRL (AlignProp for diffusion) ---
from trl import AlignPropTrainer, AlignPropConfig, DefaultDDPOStableDiffusionPipeline  # pipeline class works for AlignProp too

# --- Reward model: OWL-ViT (open-vocabulary detector) ---
from transformers import OwlViTForObjectDetection, OwlViTProcessor

# ---------------------------
# 1) Differentiable reward: "absence of books"
# ---------------------------
class BookAbsenceReward(nn.Module):
    """
    Differentiable reward: 1 - P(books present)
    Uses OWL-ViT's raw logits (pre-NMS) so gradients flow to the image.
    """
    def __init__(self, device: torch.device, model_id: str = "google/owlvit-base-patch32"):
        super().__init__()
        self.device = device
        # Text queries that indicate the undesired concept
        self.queries: List[str] = [
            "a photo of a book", "books"
        ]
        # Tokenize text queries once (this is not part of the grad path and is OK on CPU)
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.detector = OwlViTForObjectDetection.from_pretrained(model_id).to(device)
        self.detector.eval()
        for p in self.detector.parameters():
            p.requires_grad_(False)  # we don't train the reward model

        # CLIP mean/std that OWL-ViT expects (keep on device for faster math)
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None])
        self.register_buffer("std",  torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None])
        self.target_size = (768, 768)  # OWL-ViT default image size in docs

    @torch.enable_grad()  # ensure grads can flow back to images
    def forward(
        self,
        images: torch.Tensor,                 # [B, 3, H, W], values in [0,1]
        prompts: Tuple[str, ...],             # unused here, but part of TRL signature
        prompt_meta: Tuple[Any, ...]          # unused
    ) -> Tuple[torch.Tensor, Any]:
        """
        Returns a tuple (reward, metadata):
        - reward: tensor of shape [B], higher is better for RL.
        - metadata: optional info dict for logging.
        Here: reward = 1 - presence_prob_of_books
        """
        B, C, H, W = images.shape
        # Resize with differentiable op, then normalize with CLIP stats
        pix = F.interpolate(images, size=self.target_size, mode="bicubic", align_corners=False)
        pix = (pix - self.mean) / self.std  # keep gradients

        # Prepare text inputs (same queries for each image)
        text_labels = [self.queries for _ in range(B)]
        txt = self.processor(text=text_labels, return_tensors="pt", padding=True)
        txt = {k: v.to(self.device) for k, v in txt.items() if k in ("input_ids", "attention_mask")}

        # Forward pass (NO post_process; we want raw logits to keep differentiability)
        out = self.detector(pixel_values=pix, **txt)
        scores = self.processor.post_process_object_detection(
            outputs=out,
            target_sizes=[self.target_size]*B,
            threshold=0.0,  # return all boxes (no thresholding)
        )

        presence = torch.zeros(B, device=self.device)
        for b in range(B):
            presence[b] = max(scores[b]['scores']) if len(scores[b]['scores']) > 0 else 0.0

        reward = 1.0 - presence  # higher reward for less book presence

        # Optionally stabilize:
        reward = reward.clamp(0.0, 1.0)
        metadata = {"book_presence": presence.detach().float().cpu()}
        return reward, metadata

# ---------------------------
# 2) Prompt function: force "… and books" contexts
# ---------------------------
OBJECTS = [
    "living room", "kitchen", "office", "classroom", "bedroom", "cozy cafe", "library interior",
    "street market", "art studio", "minimalist workspace", "sunlit reading nook", "conference table"
]

def make_prompt_fn(seed: int = 0):
    rng = random.Random(seed)
    def next_prompt() -> Tuple[str, Any]:
        obj = rng.choice(OBJECTS)
        # Encourage scenes where books might normally appear, so the model must learn to *avoid* rendering them.
        prompt = f"photorealistic {obj} with natural lighting and books"
        return prompt, {}  # (prompt, metadata)
    return next_prompt

# ---------------------------
# 3) Training harness (AlignProp on SD-1.5 with LoRA)
# ---------------------------
@dataclass
class TrainArgs:
    sd_model_id: str = "sd-legacy/stable-diffusion-v1-5"
    output_dir: str = "./concept_removal_books_lora"
    num_epochs: int = 2000
    sample_num_steps: int = 50      # denoising steps used when sampling for reward
    guidance_scale: float = 5.0
    train_batch_size: int = 2
    sample_batch_size: int = 2
    lr: float = 1e-5
    use_lora: bool = True
    mixed_precision: str = "fp16"   # bf16 also works on suitable GPUs
    seed: int = 1234

def main():
    Path(TrainArgs.output_dir).mkdir(parents=True, exist_ok=True)  # ensure it exists
    torch.manual_seed(TrainArgs.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reward
    reward_module = BookAbsenceReward(device=device)

    # Prompt generator
    prompt_fn = make_prompt_fn(seed=TrainArgs.seed)

    # SD pipeline with LoRA (TRL's default SD training pipeline works for AlignProp too)
    # Initialize with use_lora=False to avoid attempting to download non-existent LoRA weights
    # from the base SD repo during __init__. We'll enable LoRA immediately after.
    pipe = DefaultDDPOStableDiffusionPipeline(
        pretrained_model_name=TrainArgs.sd_model_id,
        pretrained_model_revision="main",
        use_lora=False,
    )
    if TrainArgs.use_lora:
        pipe.use_lora = True
    pipe.set_progress_bar_config(disable=True)

    # AlignProp config (backprop through denoising)
    cfg = AlignPropConfig(
        num_epochs=TrainArgs.num_epochs,
        # sampling
        sample_num_steps=TrainArgs.sample_num_steps,
        sample_guidance_scale=TrainArgs.guidance_scale,
        # sample_batch_size=TrainArgs.sample_batch_size,
        # training
        train_batch_size=TrainArgs.train_batch_size,
        train_learning_rate=TrainArgs.lr,
        mixed_precision=TrainArgs.mixed_precision,
        allow_tf32=True,
        # truncated backprop to save memory (paper shows randomized truncation works well)
        truncated_backprop_rand=True,
        # (Optional) push_to_hub=False by default
        # log_with=None,  # or "tensorboard"/"wandb" if you want tracking
        # logdir=TrainArgs.output_dir,  # TRL’s own top-level logs/checkpoints
        project_kwargs={               # passed to accelerate.utils.ProjectConfiguration
            "project_dir": TrainArgs.output_dir,
            "logging_dir": TrainArgs.output_dir,  # esp. needed if log_with="tensorboard"
        },
        # (Optional safety for some TRL versions)
        accelerator_kwargs={
            "project_dir": TrainArgs.output_dir,
        },
        log_with="wandb",  # set to "wandb" or "tensorboard" if you want logging
        tracker_project_name="trl-alignprop",
        logdir=TrainArgs.output_dir,
        seed=TrainArgs.seed,
        save_freq=20,
        num_checkpoint_limit=5
    )

    trainer = AlignPropTrainer(
        config=cfg,
        reward_function=reward_module,    # nn.Module is accepted; TRL will call it with (images, prompts, meta)
        prompt_function=prompt_fn,
        sd_pipeline=pipe,
        image_samples_hook=lambda *args, **kwargs: None,
    )

    # Disable image_samples_callback after init to avoid accessing accelerator.trackers
    # when logging is disabled (no trackers created).
    try:
        trainer.image_samples_callback = None
    except Exception:
        pass

    # Train
    trainer.train()

    # Save LoRA adapter weights for later use with diffusers
    # (diffusers exposes save_lora_weights on SD pipelines)
    try:
        print("Saving LoRA weights...")
        pipe.save_lora_weights(TrainArgs.output_dir)  # safetensors file + adapter config
        print("LoRA weights saved.")
    except AttributeError:
        # Fallback: save the entire pipeline (bigger), or access pipe.unet.attn_procs_state_dict() manually.
        pipe.save_pretrained(TrainArgs.output_dir)

    print(f"Finished. LoRA saved to: {TrainArgs.output_dir}")

    # --- quick check: generate images before/after loading LoRA (optional) ---
    # from diffusers import StableDiffusionPipeline
    # base = StableDiffusionPipeline.from_pretrained(TrainArgs.sd_model_id, torch_dtype=torch.float16).to(device)
    # base.load_lora_weights(TrainArgs.output_dir)  # apply the LoRA
    # img = base("photorealistic library interior with natural lighting and books", num_inference_steps=30, guidance_scale=5.0).images[0]
    # img.save("after_concept_removal.png")

if __name__ == "__main__":
    # Tip: run with `accelerate launch concept_removal_alignprop.py` for multi-GPU/AMP configs
    main()
