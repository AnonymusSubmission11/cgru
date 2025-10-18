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

from transformers import CLIPProcessor
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


def load_prompts_from_file(filepath: str) -> List[str]:
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


# ------------------------------------------------------------
# Binary Classification Head (like CosineClassifier but simpler)
# ------------------------------------------------------------
class BinaryClassifier(nn.Module):
    """Binary classifier: outputs [safe_logit, nsfw_logit]"""
    def __init__(self, in_dim: int, init_logit_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, in_dim))  # 2 classes
        nn.init.xavier_normal_(self.weight)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

    def forward(self, feats: torch.Tensor):
        """
        Args:
            feats: [B, in_dim]
        Returns:
            logits: [B, 2] where [:, 0] = safe class, [:, 1] = nsfw class
            
        Note: 
            Label 0 = Safe (high reward >= threshold)
            Label 1 = NSFW (low reward < threshold)
        """
        feats = F.normalize(feats, dim=-1)
        W = F.normalize(self.weight, dim=-1)
        return feats @ W.t() * self.logit_scale.exp()


# ------------------------------------------------------------
# Metrics (adapted from CLIP code)
# ------------------------------------------------------------
def _confusion(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    TP, FP, FN = np.zeros(K), np.zeros(K), np.zeros(K)
    for t, p in zip(y_true, y_pred):
        if p == t: TP[t] += 1
        else: FP[p] += 1; FN[t] += 1
    return TP, FP, FN

def precision_recall_accuracy(y_true: np.ndarray, y_pred: np.ndarray, K: int = 2):
    TP, FP, FN = _confusion(y_true, y_pred, K)
    prec_c = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
    rec_c  = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(prec_c.mean()),
        "macro_recall": float(rec_c.mean()),
        "micro_precision": float(TP.sum() / max(TP.sum() + FP.sum(), 1.0)),
        "micro_recall": float(TP.sum() / max(TP.sum() + FN.sum(), 1.0)),
        "per_class_precision": prec_c,
        "per_class_recall": rec_c,
        "tp": TP,
        "fp": FP,
        "fn": FN,
    }


# ------------------------------------------------------------
# Multi-Reward Computer (precomputes multiple reward functions)
# ------------------------------------------------------------
class MultiRewardComputer:
    """
    Compute multiple reward functions and cache them.
    Supports: nsfw_unlearning, nsfw_q16_unlearning, etc.
    """
    def __init__(self, reward_methods: List[str]):
        """
        Args:
            reward_methods: List of reward function names, e.g.,
                           ['nsfw_unlearning', 'nsfw_q16_unlearning']
        """
        self.reward_methods = reward_methods
        self._reward_fns = {}
        
    def _get_reward_fn(self, method_name: str):
        """Lazy load reward functions"""
        if method_name not in self._reward_fns:
            if method_name == 'nsfw_unlearning':
                self._reward_fns[method_name] = nsfw_unlearning()
            elif method_name == 'nsfw_q16_unlearning':
                self._reward_fns[method_name] = nsfw_q16_unlearning()
            else:
                raise ValueError(f"Unknown reward method: {method_name}")
        return self._reward_fns[method_name]
    
    @torch.inference_mode()
    def compute_all_rewards(
        self, 
        images: torch.Tensor, 
        prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all reward functions.
        
        Args:
            images: [B, 3, H, W] in [0, 1]
            prompts: list of B prompts
            
        Returns:
            Dict mapping method_name -> [B] tensor of rewards
        """
        rewards_dict = {}
        for method_name in self.reward_methods:
            reward_fn = self._get_reward_fn(method_name)
            rewards, _ = reward_fn(images, prompts, {})
            rewards_dict[method_name] = torch.tensor(rewards, dtype=torch.float32)
        return rewards_dict


# ------------------------------------------------------------
# SD1.5 NSFW Dataset (Cache x0 + ALL rewards)
# ------------------------------------------------------------
class NSFWDataset(Dataset):
    """
    Dataset that caches generated x0 images, prompts, AND precomputed rewards.
    Multiple reward functions are computed and stored.
    """
    def __init__(
        self,
        prompts: List[str],
        num_samples: int,
        reward_methods: List[str] = ['nsfw_q16_unlearning'],
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
        self.reward_methods = reward_methods
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
        self.reward_computer = MultiRewardComputer(reward_methods)

    def __len__(self):
        return self.num_samples

    def _get_pipe(self):
        if self._pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.sd_model, torch_dtype=self.dtype, safety_checker=None
            )
            self._pipe = pipe.to(self.device)
            try:
                self._pipe.enable_attention_slicing()
                self._pipe.enable_vae_slicing()
            except:
                pass
        return self._pipe

    def _cache_path(self, idx):
        if self.cache_dir:
            return self.cache_dir / f"nsfw_{idx}.pt"
        return None

    def _maybe_load(self, idx):
        """Load cached x0 image, prompt, and ALL rewards"""
        if not self.cache_dir:
            return None
        p = self._cache_path(idx)
        if p and p.exists():
            try:
                data = torch.load(p)
                # Verify all reward methods are present
                if 'rewards' in data:
                    cached_methods = set(data['rewards'].keys())
                    required_methods = set(self.reward_methods)
                    if cached_methods >= required_methods:
                        return data["image_01"], data["prompt"], data["rewards"]
                    else:
                        print(f"⚠️  Cache miss for idx {idx}: missing reward methods {required_methods - cached_methods}")
            except Exception as e:
                print(f"⚠️  Error loading cache for idx {idx}: {e}")
        return None

    def _save(self, idx, img_01, prompt, rewards_dict):
        """Save x0 image, prompt, and ALL rewards"""
        if not self.cache_dir:
            return
        p = self._cache_path(idx)
        torch.save({
            "image_01": img_01,
            "prompt": prompt,
            "rewards": rewards_dict,  # Dict[str, float]
            "idx": idx,
            "reward_methods": self.reward_methods,
        }, p)
        
        # Save metadata with reward info
        meta = {
            "prompt": prompt, 
            "idx": idx,
            "rewards": {k: float(v) for k, v in rewards_dict.items()},
            "reward_methods": self.reward_methods,
        }
        with open(self.cache_dir / "_meta" / f"nsfw_{idx}.json", "w") as f:
            json.dump(meta, f, indent=2)

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

    @torch.inference_mode()
    def _compute_rewards(self, img_01, prompt):
        """Compute all reward functions on x0 image"""
        # Add batch dimension
        img_batch = img_01.unsqueeze(0).to(self.device)
        prompts = [prompt]
        
        # Compute all rewards
        rewards_dict = self.reward_computer.compute_all_rewards(img_batch, prompts)
        
        # Remove batch dimension and convert to float
        rewards_dict = {k: float(v[0]) for k, v in rewards_dict.items()}
        return rewards_dict

    def __getitem__(self, idx):
        cached = self._maybe_load(idx)
        if cached is not None:
            img_01, prompt, rewards_dict = cached
        else:
            # Generate image
            prompt = self.prompts[idx % len(self.prompts)]
            seed = self.base_seed + idx
            img = self._synthesize(prompt, seed)
            img_01 = self.to_tensor_01(img)
            
            # Compute rewards on x0
            rewards_dict = self._compute_rewards(img_01, prompt)
            
            # Save to cache
            self._save(idx, img_01, prompt, rewards_dict)
            
            if idx % 100 == 0:
                print(f"[Dataset] Generated sample {idx}/{self.num_samples} ...")
                print(f"  Rewards: {rewards_dict}")
        
        return {
            "image_01": img_01,  
            "prompt": prompt,
            "rewards": rewards_dict,  # Dict[str, float]
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


def rewards_to_labels(rewards: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert continuous rewards to binary labels: 0=safe, 1=nsfw"""
    return (rewards < threshold).long()


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
    # Extract reward scores for the primary method
    rewards_dicts = [b["rewards"] for b in batch]
    return {
        "image_01": imgs, 
        "prompts": prompts,
        "rewards": rewards_dicts,
    }


# ------------------------------------------------------------
# Evaluation (adapted from CLIP code)
# ------------------------------------------------------------
@torch.inference_mode()
def evaluate(
    backbone, 
    classifier, 
    loader, 
    device, 
    vae_ddim, 
    clip_mean, 
    clip_std, 
    reward_method: str,
    threshold: float = 0.5
):
    backbone.eval()
    classifier.eval()
    
    ys, ps = [], []
    all_losses = []
    
    for batch in loader:
        x01 = batch["image_01"].to(device)
        prompts = batch["prompts"]
        rewards_dicts = batch["rewards"]

        # Sample timesteps
        tau = sample_uniform((x01.size(0),), device=device)
        t_idx = tau_to_index(tau, vae_ddim.T)
        
        # Add noise
        z0 = vae_ddim.encode_to_latents(x01)
        zt = vae_ddim.add_noise(z0, t_idx)
        xt01 = vae_ddim.decode_from_latents(zt)
        
        # Get precomputed rewards and convert to labels
        rewards = torch.tensor([rd[reward_method] for rd in rewards_dicts], device=device)
        y = rewards_to_labels(rewards, threshold)
        
        # CLIP normalization
        pixel_values = (xt01 - clip_mean) / clip_std
        
        # Predict
        out = backbone(pixel_values=pixel_values, timesteps=tau)
        logits = classifier(out.pooler_output)  # [B, 2]
        p = logits.argmax(dim=-1)  # [B]
        
        # Loss
        loss = F.cross_entropy(logits, y)
        all_losses.append(loss.item() * x01.size(0))
        
        ys.append(y.cpu().numpy())
        ps.append(p.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    metrics = precision_recall_accuracy(y_true, y_pred, K=2)
    metrics['loss'] = sum(all_losses) / len(y_true)
    
    return metrics


# ------------------------------------------------------------
# Checkpoint Management
# ------------------------------------------------------------
class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_keep: int = 2):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.best_checkpoints = []
    
    def save(self, checkpoint_dict: dict, tag: str, is_best: bool = False):
        if is_best:
            val_loss = checkpoint_dict['val_loss']
            checkpoint_path = self.checkpoint_dir / f"nsfw_critic_best_{tag}_loss_{val_loss:.4f}.pt"
            torch.save(checkpoint_dict, checkpoint_path)
            self.best_checkpoints.append({'path': checkpoint_path, 'tag': tag, 'loss': val_loss})
            print(f"✅ Saved best model: tag {tag}, Loss: {val_loss:.4f}")
            self.best_checkpoints.sort(key=lambda x: x['loss'])
            while len(self.best_checkpoints) > self.max_keep:
                worst = self.best_checkpoints.pop()
                if worst['path'].exists():
                    worst['path'].unlink()
                    print(f"🗑️  Removed checkpoint: tag {worst['tag']}, Loss: {worst['loss']:.4f}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # ----------------- Config -----------------
    PROMPT_FILE = "ddpo_pytorch/assets/nsfw/all-nsfw-dataset-llava.txt"
    CACHE_DIR = "./nsfw_cache"
    
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 16
    epochs = 200
    lr_new = 1e-4
    lr_base = 0.0
    weight_decay = 0.01

    total_samples = 2000
    NUM_SAMPLES_TRAIN = 200
    NUM_SAMPLES_VAL   = 200
    
    # Multiple reward methods to precompute
    REWARD_METHODS = ['nsfw_unlearning', 'nsfw_q16_unlearning']
    PRIMARY_REWARD_METHOD = 'nsfw_unlearning'  # Which one to use for training
    NSFW_THRESHOLD = 0.5

    # ------------------------------------------
    set_seed(seed)
    
    # Load prompts
    print(f"Loading prompts from: {PROMPT_FILE}")
    prompts = load_prompts_from_file(PROMPT_FILE)
    print(f"Loaded {len(prompts)} prompts\n")
    
    # Create base dataset with multiple reward methods
    print(f"Precomputing reward methods: {REWARD_METHODS}")
    base_ds = NSFWDataset(
        prompts=prompts,
        num_samples=total_samples,
        reward_methods=REWARD_METHODS,
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
    print(f"   Validation: {len(val_set)} samples")
    print(f"   Primary reward method for training: {PRIMARY_REWARD_METHOD}\n")
    
    # WandB
    wandb.init(
        project="nsfw-critic-classification",
        name=f"nsfw_classifier_{total_samples}",
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
            "prompt_file": PROMPT_FILE,
            "num_prompts": len(prompts),
            "architecture": "TimestepAwareCLIP + BinaryClassifier",
            "task": "binary_classification",
            "reward_methods": REWARD_METHODS,
            "primary_reward_method": PRIMARY_REWARD_METHOD,
            "nsfw_threshold": NSFW_THRESHOLD,
            "num_samples_train": NUM_SAMPLES_TRAIN,
            "num_samples_val": NUM_SAMPLES_VAL,
        }
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False,
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

    classifier = BinaryClassifier(in_dim=feat_dim).to(device)
    
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
        {"params": classifier.parameters(), "lr": lr_new, "weight_decay": weight_decay},
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
    
    ckpt_manager = CheckpointManager("checkpoints", max_keep=1)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    global_step = 0

    # Windowed train metrics accumulators
    win_train_loss_sum = 0.0
    win_train_correct = 0
    win_train_count = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}")
        backbone.train()
        classifier.train()

        for step, batch in enumerate(train_loader, start=1):
            x01 = batch["image_01"].to(device)
            prompts = batch["prompts"]
            rewards_dicts = batch["rewards"]
            bsz = x01.size(0)

            # Sample timesteps
            tau = sample_uniform((bsz,), device=device)
            t_idx = tau_to_index(tau, vae_ddim.T)

            # Add noise: x0 -> xt
            z0 = vae_ddim.encode_to_latents(x01)
            zt = vae_ddim.add_noise(z0, t_idx)
            xt01 = vae_ddim.decode_from_latents(zt)

            # Get precomputed rewards and convert to binary labels
            rewards = torch.tensor(
                [rd[PRIMARY_REWARD_METHOD] for rd in rewards_dicts], 
                device=device
            )
            y = rewards_to_labels(rewards, NSFW_THRESHOLD)

            # CLIP preprocessing
            pixel_values = (xt01 - mean) / std

            # Forward
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = backbone(pixel_values=pixel_values, timesteps=tau)
                logits = classifier(out.pooler_output)  # [B, 2]
                loss = F.cross_entropy(logits, y)
                
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            # Windowed train metrics
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == y).sum().item()
            
            win_train_loss_sum += loss.item() * bsz
            win_train_correct  += correct
            win_train_count    += bsz

            global_step += 1
            acc = correct / bsz
            wandb.log({
                "train/step_loss": loss.item(), 
                "train/step_accuracy": acc, 
                "step": global_step, 
                "epoch": epoch
            })

            # ---- Trigger validation strictly by NUM_SAMPLES_TRAIN ----
            if win_train_count >= NUM_SAMPLES_TRAIN:
                train_loss = win_train_loss_sum / win_train_count
                train_acc = win_train_correct / win_train_count

                # Run validation
                val_metrics = evaluate(
                    backbone, classifier, val_loader, device, vae_ddim, mean, std,
                    reward_method=PRIMARY_REWARD_METHOD, threshold=NSFW_THRESHOLD
                )

                # Print both train & val window metrics
                print(
                    f"[Epoch {epoch} | Step {step}] "
                    f"TRAIN (window {win_train_count}): Loss {train_loss:.4f} | Acc {train_acc:.4f}  ||  "
                    f"VAL: Loss {val_metrics['loss']:.4f} | Acc {val_metrics['accuracy']:.4f} | "
                    f"Prec {val_metrics['macro_precision']:.4f} | Rec {val_metrics['macro_recall']:.4f}"
                )
                print(f"  Per-class: Safe P={val_metrics['per_class_precision'][0]:.3f} R={val_metrics['per_class_recall'][0]:.3f} | "
                      f"NSFW P={val_metrics['per_class_precision'][1]:.3f} R={val_metrics['per_class_recall'][1]:.3f}")

                # Log both
                wandb.log({
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/count": win_train_count,
                    "val/loss": val_metrics['loss'],
                    "val/accuracy": val_metrics['accuracy'],
                    "val/macro_precision": val_metrics['macro_precision'],
                    "val/macro_recall": val_metrics['macro_recall'],
                    "val/safe_precision": val_metrics['per_class_precision'][0],
                    "val/safe_recall": val_metrics['per_class_recall'][0],
                    "val/nsfw_precision": val_metrics['per_class_precision'][1],
                    "val/nsfw_recall": val_metrics['per_class_recall'][1],
                    "epoch": epoch,
                    "step": global_step,
                })

                # Save best on window val
                is_best = val_metrics['loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['loss']
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_val_accuracy"] = val_metrics['accuracy']
                
                ckpt_manager.save({
                    "epoch": epoch,
                    "step": global_step,
                    "backbone_state_dict": backbone.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_num_steps": vae_ddim.T,
                    "vae_scaling_factor": vae_ddim.scaling_factor,
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy'],
                    "best_val_loss": best_val_loss,
                    "reward_methods": REWARD_METHODS,
                    "primary_reward_method": PRIMARY_REWARD_METHOD,
                    "nsfw_threshold": NSFW_THRESHOLD,
                }, tag=f"e{epoch}_s{global_step}", is_best=is_best)

                # Reset train window accumulators
                win_train_loss_sum = 0.0
                win_train_correct  = 0
                win_train_count    = 0

    # Save final model
    final_checkpoint = {
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "scheduler_num_steps": vae_ddim.T,
        "vae_scaling_factor": vae_ddim.scaling_factor,
        "reward_methods": REWARD_METHODS,
        "primary_reward_method": PRIMARY_REWARD_METHOD,
        "nsfw_threshold": NSFW_THRESHOLD,
    }
    torch.save(final_checkpoint, "checkpoints/nsfw_classifier_final.pt")
    print("✅ Saved final model to checkpoints/nsfw_classifier_final.pt")
    
    wandb.finish()


if __name__ == "__main__":
    main()