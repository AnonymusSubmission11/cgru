import os, re, json, random, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel
from timestep_aware_clip import TimestepAwareCLIPVisionModel

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ddpo_pytorch.rewards import nsfw_unlearning, nsfw_q16_unlearning

import wandb


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts_from_txt(filepath: str) -> List[str]:
    """Load prompts from file (supports both single-line and tab-separated format)"""
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                prompts.append(line.split('\t', 1)[1])
            else:
                prompts.append(line)
    return prompts


def load_prompts_from_json(filepath: str, field: str = "unsafe_prompt") -> list[str]:
    """
    Load prompts from a JSONL file where each line is a JSON object.
    Extracts the specified field (default: 'unsafe_prompt').
    """
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if field in item and item[field]:
                    prompts.append(str(item[field]).strip())
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    print(f"Loaded {len(prompts)} '{field}' prompts from {filepath}")
    return prompts

class ValueHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1)
        )
        # init: keep output near 0 at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feats: torch.Tensor):           # feats: [B, in_dim]
        x = self.norm(feats)
        x = self.mlp(x).squeeze(-1)                   # [B]
        return x

# ------------------------------------------------------------
# SD1.5 NSFW Dataset (Cache only x0, compute rewards on-the-fly for xt)
# ------------------------------------------------------------
class NSFWDataset(Dataset):
    """
    Dataset that caches generated x0 images, prompts, and rewards.
    """
    def __init__(
        self,
        prompts: List[str],
        num_samples: int,
        sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_size: int = 512,
        resize_to: int = 224,
        base_seed: int = 42,
        cache_dir: Optional[str] = None,
        negative_prompt: str = "",
    ):
        super().__init__()
        self.prompts = prompts
        self.num_samples = num_samples
        self.sd_model = sd_model
        self.device = device
        self.dtype = torch_dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.base_seed = base_seed
        self.negative_prompt = negative_prompt

        self.to_tensor_01 = T.Compose([
            T.Resize(resize_to, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(resize_to),
            T.ToTensor(),
        ])

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "_meta").mkdir(parents=True, exist_ok=True)
        
        self._pipe = None
        self._reward_computer = None

    def __len__(self):
        return self.num_samples

    def _get_pipe(self):
        if self._pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.sd_model, torch_dtype=self.dtype
            )

            def dummy(images, **kwargs):
                return images, [False]

            pipe.safety_checker = dummy
            pipe.set_progress_bar_config(disable=True)

            self._pipe = pipe.to(self.device)
            try:
                self._pipe.enable_attention_slicing()
                self._pipe.enable_vae_slicing()
            except:
                pass
        return self._pipe

    def _get_reward_computer(self):
        if self._reward_computer is None:
            self._reward_computer = RewardComputer()
        return self._reward_computer

    def _cache_path(self, idx):
        if self.cache_dir:
            return self.cache_dir / f"nsfw_{idx}.pt"
        return None

    def _maybe_load(self, idx):
        """Load cached x0 image, prompt, and reward"""
        if not self.cache_dir:
            return None
        p = self._cache_path(idx)
        if p and p.exists():
            try:
                data = torch.load(p)
                return data["image_01"], data["prompt"], data["reward"]
            except:
                return None
        return None

    def _save(self, idx, img_01, prompt, reward):
        """Save x0 image, prompt, and reward"""
        if not self.cache_dir:
            return
        p = self._cache_path(idx)
        torch.save({
            "image_01": img_01,
            "prompt": prompt,
            "reward": reward,
            "idx": idx,
        }, p)
        with open(self.cache_dir / "_meta" / f"nsfw_{idx}.json", "w") as f:
            json.dump({
                "prompt": prompt,
                "reward": float(reward),
                "idx": idx
            }, f, indent=2)

    @torch.inference_mode()
    def _synthesize(self, prompt, seed):
        pipe = self._get_pipe()
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = pipe(
            prompt=f"{prompt}",
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.image_size,
            width=self.image_size,
            generator=g
        )
        return out.images[0].convert("RGB")

    def __getitem__(self, idx):
        cached = self._maybe_load(idx)
        if cached is not None:
            img_01, prompt, reward = cached
        else:
            prompt = self.prompts[idx % len(self.prompts)]
            seed = self.base_seed + idx
            img = self._synthesize(prompt, seed)
            img_01 = self.to_tensor_01(img)
            
            # Compute reward for x0
            reward_computer = self._get_reward_computer()
            reward = reward_computer.compute_rewards([img], [prompt])[0].item()  # pass PIL
            if reward < 0:
                print(f"Negative reward {reward:.4f} for idx={idx}, prompt='{prompt}'")
            else:
                print(f"Positive reward {reward:.4f} for idx={idx}, prompt='{prompt}'")
            
            self._save(idx, img_01, prompt, reward)
            # if idx % 100 == 0:
            #     print(f"[Dataset] Generated sample {idx}/{self.num_samples} ...")
        
        return {
            "image_01": img_01,  
            "prompt": prompt,
            "reward": reward,
            "idx": idx,
        }


class NSFWSplitView(Dataset):
    def __init__(self, base: NSFWDataset, split: float, kind: str):
        assert 0.0 < split < 1.0 and kind in ("train", "val")
        self.base = base
        n = len(base)
        k = int(n * split)
        if kind == "train":
            self.indices = list(range(0, k))
        else:
            self.indices = list(range(k, n))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        return self.base[self.indices[i]]


# ------------------------------------------------------------
# VAE + DDIM forward
# ------------------------------------------------------------
class VAEDDIMForward:
    def __init__(self, sd_model: str, device: str = "cuda", torch_dtype=torch.float16):
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            sd_model, subfolder="vae", torch_dtype=torch_dtype
        ).to(device)
        self.vae.eval()
        self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(
            sd_model, subfolder="scheduler"
        )
        self.T = self.scheduler.config.num_train_timesteps
        self.device = device
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
    
    @torch.inference_mode()
    def encode_to_latents(self, x01: torch.Tensor, generator=None):
        x = x01 * 2 - 1
        x = x.to(dtype=self.vae.dtype)
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample(generator=generator) * self.scaling_factor
        return z
    
    @torch.inference_mode()
    def decode_from_latents(self, z: torch.Tensor):
        x = self.vae.decode((z / self.scaling_factor).to(self.vae.dtype)).sample
        x = x.clamp(-1, 1)
        x01 = (x + 1) / 2.0
        return x01
    
    @torch.inference_mode()
    def add_noise(self, z0: torch.Tensor, t_idx: torch.Tensor, generator=None):
        noise = torch.randn_like(z0)
        zt = self.scheduler.add_noise(z0, noise, t_idx)
        return zt


class RewardComputer:
    """Compute rewards for images"""
    def __init__(self):
        self._reward_fn = None
    
    def _get_reward_fn(self):
        if self._reward_fn is None:
            self._reward_fn = nsfw_unlearning()
            # self._reward_fn = nsfw_q16_unlearning() 
        return self._reward_fn
    
    @torch.inference_mode()
    def compute_rewards(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        reward_fn = self._get_reward_fn()
        rewards, _ = reward_fn(images, prompts, {})
        return torch.tensor(rewards, dtype=torch.float32)


# ------------------------------------------------------------
# τ sampling
# ------------------------------------------------------------
def sample_uniform(shape, device):
    return torch.rand(shape, device=device).clamp(1e-6, 1 - 1e-6)

def tau_to_index(tau: torch.Tensor, total_steps: int):
    return (tau * (total_steps - 1)).long().clamp(0, total_steps - 1)


# ------------------------------------------------------------
# Collate
# ------------------------------------------------------------
def collate_raw(batch):
    imgs = torch.stack([b["image_01"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
    return {"image_01": imgs, "prompts": prompts, "rewards": rewards}


# ------------------------------------------------------------
# Evaluation (MSE to terminal reward r(x0))
# ------------------------------------------------------------
@torch.inference_mode()
def evaluate(backbone, regressor, loader, device, vae_ddim, clip_mean, clip_std, reward_computer, text_encoder):
    backbone.eval()
    regressor.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    
    for batch in loader:
        x01 = batch["image_01"].to(device)
        prompts = batch["prompts"]
        y_reward = batch["rewards"].to(device)
        y_safe = torch.sigmoid(y_reward / 2.0)

        # Encode text prompts
        h_txt = text_encoder.encode(prompts)

        # Sample timesteps (used only to form (x_t, t); target remains r(x0))
        tau = sample_uniform((x01.size(0),), device=device)
        t_idx = tau_to_index(tau, vae_ddim.T)
        
        # Add noise to get x_t (critic input)
        z0 = vae_ddim.encode_to_latents(x01)
        zt = vae_ddim.add_noise(z0, t_idx)
        xt01 = vae_ddim.decode_from_latents(zt)

        # CLIP normalization of x_t
        pixel_values = (xt01 - clip_mean) / clip_std
        
        # Predict V(x_t, t, prompt)
        out = backbone(pixel_values=pixel_values, timesteps=tau)
        pred_reward = regressor(out.pooler_output, h_txt)  # Pass text embedding
        
        loss = F.binary_cross_entropy_with_logits(pred_reward, y_safe)
        mae = F.l1_loss(pred_reward, y_safe)
        
        bsz = x01.size(0)
        total_loss += loss.item() * bsz
        total_mae += mae.item() * bsz
        count += bsz
    
    if count == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    mse = total_loss / count
    return {"mse": mse, "mae": total_mae / count, "rmse": np.sqrt(mse)}


# ------------------------------------------------------------
# Checkpoint Management
# ------------------------------------------------------------
class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_mse = float("inf")
        self.best_path = None
        self.max_keep = max_keep
        self.saved_best_paths = []  # Track all saved best checkpoints
    
    def save(self, checkpoint_dict: dict, tag: str, consider_best: bool = True):
        # always save a regular checkpoint
        ckpt_path = self.checkpoint_dir / f"nsfw_critic_{tag}.pt"
        # torch.save(checkpoint_dict, ckpt_path)
        # print(f"💾 Saved checkpoint: {ckpt_path}")

        # also save a "best" snapshot if improved (no deletion of old files)
        if consider_best:
            val_mse = checkpoint_dict.get("val_mse", None)
            if val_mse is not None and val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                best_path = self.checkpoint_dir / f"nsfw_critic_best_mse_{val_mse:.4f}.pt"
                torch.save(checkpoint_dict, best_path)
                self.best_path = best_path
                self.saved_best_paths.append(best_path)
                print(f"🏆 New best (MSE={val_mse:.4f}) -> {best_path}")
                
                # Keep only last max_keep checkpoints
                if len(self.saved_best_paths) > self.max_keep:
                    old_path = self.saved_best_paths.pop(0)
                    if old_path.exists():
                        old_path.unlink()
                        print(f"🗑️  Removed old checkpoint: {old_path.name}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # ----------------- Config -----------------
    PROMPT_FILE_TXT = "ddpo_pytorch/assets/nsfw/nsfw_critic_10.txt"
    # PROMPT_FILE_JSON = "ddpo_pytorch/assets/CoPro/cgru_critic_20_sexual.jsonl"
    CACHE_DIR = "./nsfw_cache"
    
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 16
    epochs = 200
    lr_new = 1e-4
    lr_base = 0.0
    weight_decay = 0.01

    # ------------------------------------------
    set_seed(seed)
    
    # Load prompts
    print(f"Loading prompts from: {PROMPT_FILE_TXT}")
    prompts_txt = load_prompts_from_txt(PROMPT_FILE_TXT)
    # print(f"Loading prompts from: {PROMPT_FILE_JSON}")
    # prompts_json = load_prompts_from_json(PROMPT_FILE_JSON, field="unsafe_prompt")
    prompts = prompts_txt  # + prompts_json

    print(f"Loaded {len(prompts)} prompts\n")
    
    total_samples = len(prompts)

    # Create base dataset
    base_ds = NSFWDataset(
        prompts=prompts,
        num_samples=total_samples,
        cache_dir=CACHE_DIR,
        base_seed=seed,
        sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        device=device,
        torch_dtype=torch.float16,
        num_inference_steps=50,
        guidance_scale=7.5,
        image_size=512,
        resize_to=224,
    )
    
    # Split
    train_set = NSFWSplitView(base_ds, split=0.9, kind="train")
    val_set = NSFWSplitView(base_ds, split=0.9, kind="val")
    
    print(f"\nDataset sizes:")
    print(f"   Training: {len(train_set)} samples")
    print(f"   Validation: {len(val_set)} samples\n")
    
    # WandB
    wandb.init(
        project="_nsfw-critic",
        name=f"nsfw_critic_film_{total_samples}",
        config={
            "total_samples": total_samples,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr_new": lr_new,
            "lr_base": lr_base,
            "weight_decay": weight_decay,
            "seed": seed,
            "num_prompts": len(prompts),
            "architecture": "TimestepAwareCLIP + FiLM + LinearHead",
            "conditioning": "prompt_film",
            "cache_format": "x0_only",
            "reward_computation": "on_x0",
        }
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_raw
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_raw
    )
    
    # Model
    backbone = TimestepAwareCLIPVisionModel.from_pretrained_clip(
        "openai/clip-vit-base-patch32", gate_add_one=True, time_embed_mult=4
    ).to(device)
    feat_dim = backbone.config.hidden_size

    # Text encoder for prompt conditioning
    text_encoder = TextEncoder(model_name="openai/clip-vit-base-patch32", device=device)
    
    # FiLM-conditioned regressor
    regressor = FiLMHead(in_dim=feat_dim, txt_dim=512, hidden=512).to(device)
    
    # Optim
    def is_new_param(n):
        return ("time_embed" in n) or ("adaln_attn" in n) or ("adaln_mlp" in n)
    new_params, base_params = [], []
    for n, p in backbone.named_parameters():
        (new_params if is_new_param(n) else base_params).append(p)
    for p in base_params:
        p.requires_grad = (lr_base > 0.0)
    optim = torch.optim.AdamW([
        {"params": new_params, "lr": lr_new, "weight_decay": weight_decay},
        {"params": regressor.parameters(), "lr": lr_new, "weight_decay": weight_decay},
    ])

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    # VAE + DDIM + CLIP norm
    vae_ddim = VAEDDIMForward(
        sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        device=device,
        torch_dtype=torch.float16
    )
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    mean = torch.tensor(clip_proc.image_processor.image_mean, device=device).view(1, -1, 1, 1)
    std  = torch.tensor(clip_proc.image_processor.image_std,  device=device).view(1, -1, 1, 1)
    
    reward_computer = RewardComputer()
    ckpt_manager = CheckpointManager("checkpoints")
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    best_val_mse = float('inf')
    global_step = 0

    # Windowed train metrics accumulators (every 10 batches)
    win_train_loss_sum = 0.0
    win_train_mae_sum = 0.0
    win_train_count = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}")
        backbone.train()
        regressor.train()

        for step, batch in enumerate(train_loader, start=1):
            x01 = batch["image_01"].to(device)
            prompts = batch["prompts"]
            y_reward = batch["rewards"].to(device) 
            y_safe = torch.sigmoid(y_reward / 2.0)
            bsz = x01.size(0)

            # Encode text prompts
            with torch.no_grad():
                h_txt = text_encoder.encode(prompts)

            # Sample timesteps (for forming critic inputs x_t)
            tau = sample_uniform((bsz,), device=device)
            t_idx = tau_to_index(tau, vae_ddim.T)

            # Add noise: x0 -> x_t
            z0 = vae_ddim.encode_to_latents(x01)
            zt = vae_ddim.add_noise(z0, t_idx)
            xt01 = vae_ddim.decode_from_latents(zt)

            # CLIP preprocessing of x_t
            pixel_values = (xt01 - mean) / std

            # Forward
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = backbone(pixel_values=pixel_values, timesteps=tau)
                pred_reward = regressor(out.pooler_output, h_txt)  # FiLM conditioning
                loss = F.binary_cross_entropy_with_logits(pred_reward, y_safe)
                
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(regressor.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            # Windowed train metrics
            with torch.no_grad():
                mae = F.l1_loss(pred_reward, y_safe)
            win_train_loss_sum += loss.item() * bsz
            win_train_mae_sum  += mae.item() * bsz
            win_train_count    += bsz

            global_step += 1
            wandb.log({"train/step_loss": loss.item(), "train/step_mae": mae.item(), "step": global_step, "epoch": epoch})

            # Print and validate every 10 batches
            if step % 10 == 0:
                train_mse = win_train_loss_sum / win_train_count
                train_mae = win_train_mae_sum  / win_train_count
                train_rmse = math.sqrt(train_mse)

                # Validation (targets on x0 as well)
                val_metrics = evaluate(
                    backbone, regressor, val_loader, device, vae_ddim, mean, std, reward_computer, text_encoder,
                )

                print(
                    f"[Epoch {epoch} | Step {step}] "
                    f"TRAIN (window {win_train_count}): MSE {train_mse:.4f} | MAE {train_mae:.4f} | RMSE {train_rmse:.4f}  ||  "
                    f"VAL: MSE {val_metrics['mse']:.4f} | MAE {val_metrics['mae']:.4f} | RMSE {val_metrics['rmse']:.4f}"
                )

                wandb.log({
                    "train/mse": train_mse,
                    "train/mae": train_mae,
                    "train/rmse": train_rmse,
                    "train/count": win_train_count,
                    "val/mse": val_metrics['mse'],
                    "val/mae": val_metrics['mae'],
                    "val/rmse": val_metrics['rmse'],
                    "val/count": len(val_set),
                    "epoch": epoch,
                    "step": global_step,
                })

                # Save best on window val
                is_best = val_metrics['mse'] < best_val_mse
                if is_best:
                    best_val_mse = val_metrics['mse']
                    wandb.run.summary["best_val_mse"] = best_val_mse
                ckpt_manager.save({
                    "epoch": epoch,
                    "step": global_step,
                    "backbone_state_dict": backbone.state_dict(),
                    "regressor_state_dict": regressor.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_num_steps": vae_ddim.T,
                    "vae_scaling_factor": vae_ddim.scaling_factor,
                    "val_mse": val_metrics['mse'],
                    "best_val_mse": best_val_mse,
                }, tag=f"e{epoch}_s{global_step}", consider_best=True)

                # Reset train window accumulators
                win_train_loss_sum = 0.0
                win_train_mae_sum  = 0.0
                win_train_count    = 0

    # Save final model
    final_checkpoint = {
        "backbone_state_dict": backbone.state_dict(),
        "regressor_state_dict": regressor.state_dict(),
        "scheduler_num_steps": vae_ddim.T,
        "vae_scaling_factor": vae_ddim.scaling_factor,
    }
    torch.save(final_checkpoint, "checkpoints/nsfw_critic_final.pt")
    print("✅ Saved final model to checkpoints/nsfw_critic_final.pt")
    
    wandb.finish()


if __name__ == "__main__":
    main()
