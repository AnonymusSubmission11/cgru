# ============================ #
#  Timestep-Aware CLIP (ViT-B/32) on VAE-latent DDIM intermediate steps
# ============================ #
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


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Cosine classifier head
# ------------------------------------------------------------
class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, init_logit_scale: float = 3.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

    def forward(self, feats: torch.Tensor):
        feats = F.normalize(feats, dim=-1)
        W = F.normalize(self.weight, dim=-1)
        return feats @ W.t() * self.logit_scale.exp()


# ------------------------------------------------------------
# Prompt loader (folder with sd_prompt_{object}.txt)
# ------------------------------------------------------------
PROMPT_FILE_RE = re.compile(r"^sd_prompt_(.+)\.txt$", re.IGNORECASE)

def load_prompt_folder(prompt_dir: str) -> Dict[str, List[str]]:
    prompt_dir = Path(prompt_dir)
    mapping: Dict[str, List[str]] = {}
    for p in sorted(prompt_dir.iterdir()):
        m = PROMPT_FILE_RE.match(p.name)
        if not m or not p.is_file(): continue
        cname = m.group(1)
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if not lines: raise ValueError(f"No prompts in {p}")
        mapping[cname] = lines
    if len(mapping) != 20:
        raise ValueError(f"Expected 20 classes; found {len(mapping)} in {prompt_dir}")
    return mapping


# ------------------------------------------------------------
# SD1.5 dataset (returns 224×224 tensor in [0,1] so we can VAE-encode)
# ------------------------------------------------------------
class SDPromptFolderDataset(Dataset):
    def __init__(
        self,
        prompt_dir: str,
        images_per_class: int = 100,
        sd_model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        image_size: int = 512,       # generation
        resize_to: int = 224,        # encoder input
        base_seed: int = 12345,
        cache_dir: Optional[str] = None,
        negative_prompt: Optional[str] = "text, watermark, logo, low quality, blurry, deformed, disfigured, nsfw",
    ):
        super().__init__()
        self.class_to_prompts = load_prompt_folder(prompt_dir)
        self.class_names = sorted(self.class_to_prompts.keys())
        self.images_per_class = int(images_per_class)
        self.total_len = self.images_per_class * len(self.class_names)

        self.sd_model = sd_model
        self.device = device
        self.dtype = torch_dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.base_seed = int(base_seed)
        self.negative_prompt = negative_prompt

        self.to_tensor_01 = T.Compose([
            T.Resize(resize_to, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(resize_to),
            T.ToTensor(),   # [0,1]
        ])

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: (self.cache_dir / "_meta").mkdir(parents=True, exist_ok=True)
        self._pipe = None  # lazy

    def __len__(self): return self.total_len

    def _get_pipe(self):
        if self._pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(self.sd_model, torch_dtype=self.dtype, safety_checker=None)
            self._pipe = pipe.to(self.device)
            try:
                self._pipe.enable_attention_slicing(); self._pipe.enable_vae_slicing()
            except Exception:
                pass
        return self._pipe

    def _index_to_class_local(self, idx): 
        return idx // self.images_per_class, idx % self.images_per_class

    def _seed_for(self, c, j): return self.base_seed + c * 1_000_003 + j

    def _prompt_for(self, cname, seed):
        rnd = random.Random(seed ^ 0x9E3779B97F4A7C15)
        return rnd.choice(self.class_to_prompts[cname])

    def _cache_path(self, cname, seed):
        return self.cache_dir / f"{cname.replace(' ', '_')}_{seed}.webp" if self.cache_dir else None

    def _maybe_load(self, cname, seed):
        if not self.cache_dir: return None
        p = self._cache_path(cname, seed)
        if p and p.exists():
            try: return Image.open(p).convert("RGB")
            except Exception: return None
        return None

    def _save(self, cname, seed, img, meta):
        if not self.cache_dir: return
        p = self._cache_path(cname, seed)
        img.save(p, format="WEBP", quality=95)
        with open(self.cache_dir / "_meta" / f"{cname}_{seed}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @torch.inference_mode()
    def _synthesize(self, prompt, seed):
        pipe = self._get_pipe()
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = pipe(prompt=prompt, negative_prompt=self.negative_prompt,
                   num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale,
                   height=self.image_size, width=self.image_size, generator=g)
        return out.images[0].convert("RGB")

    def __getitem__(self, idx):
        c, j = self._index_to_class_local(idx)
        cname = self.class_names[c]
        seed = self._seed_for(c, j)
        prompt = self._prompt_for(cname, seed)
        img = self._maybe_load(cname, seed)
        if img is None:
            img = self._synthesize(prompt, seed)
            self._save(cname, seed, img, {"class": cname, "prompt": prompt, "seed": seed})
        img_01 = self.to_tensor_01(img)      # (3,224,224), [0,1]
        label = torch.tensor(c, dtype=torch.long)
        return {"image_01": img_01, "label": label, "meta": {"class": cname, "seed": seed, "prompt": prompt}}


# ------------------------------------------------------------
# Per-class split view (deterministic 80/20)
# ------------------------------------------------------------
class SDClassSplitView(Dataset):
    def __init__(self, base: SDPromptFolderDataset, split: float, kind: str):
        assert 0.0 < split < 1.0 and kind in ("train", "val")
        self.base = base; self.indices = []
        n_per = base.images_per_class; k = int(n_per * split)
        for c in range(len(base.class_names)):
            start = c * n_per
            lo, hi = (start, start + k) if kind == "train" else (start + k, start + n_per)
            self.indices.extend(range(lo, hi))
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.base[self.indices[i]]


# ------------------------------------------------------------
# VAE + DDIM forward step (latent space)
#   - encode x0 in [-1,1] -> latent z0 = s * q.sample()
#   - zt = add_noise(z0, eps, t)
#   - decode zt/s -> x_t in [-1,1], then map to [0,1]
# ------------------------------------------------------------
class VAEDDIMForward:
    def __init__(self, sd_model: str, device: str = "cuda", torch_dtype=torch.float16):
        # VAE
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(sd_model, subfolder="vae", torch_dtype=torch_dtype).to(device)
        self.vae.eval()
        # DDIM scheduler (for ᾱ_t and add_noise)
        self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(sd_model, subfolder="scheduler")
        self.T = self.scheduler.config.num_train_timesteps
        self.device = device
        # scaling factor used by SD
        self.scaling_factor: float = getattr(self.vae.config, "scaling_factor", 0.18215)

    @torch.inference_mode()
    def encode_to_latents(self, x01: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        x01: (B,3,H,W) in [0,1]
        returns: z0 in latent space (B,4,H/8,W/8) scaled by scaling_factor
        """
        x = x01 * 2 - 1  # [-1,1]
        x = x.to(dtype=self.vae.dtype)  # ensure same dtype as VAE params
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample(generator=generator) * self.scaling_factor  # SD convention
        return z

    @torch.inference_mode()
    def decode_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: latents scaled by scaling_factor
        returns: x in [0,1]
        """
        x = self.vae.decode((z / self.scaling_factor).to(self.vae.dtype)).sample
        x = x.clamp(-1, 1)
        x01 = (x + 1) / 2.0
        return x01

    @torch.inference_mode()
    def add_noise(self, z0: torch.Tensor, t_idx: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Produce intermediate latent zt using DDIM's variance schedule.
        """
        noise = torch.randn_like(z0)
        zt = self.scheduler.add_noise(z0, noise, t_idx)
        return zt


# ------------------------------------------------------------
# τ sampling
# ------------------------------------------------------------
def sample_logit_normal(shape, device):
    z = torch.randn(shape, device=device)
    return torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)

def sample_uniform(shape, device):
    return torch.rand(shape, device=device).clamp(1e-6, 1 - 1e-6)

def tau_to_index(tau: torch.Tensor, total_steps: int) -> torch.Tensor:
    return (tau * (total_steps - 1)).long().clamp(0, total_steps - 1)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def _confusion(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    TP, FP, FN = np.zeros(K), np.zeros(K), np.zeros(K)
    for t, p in zip(y_true, y_pred):
        if p == t: TP[t] += 1
        else: FP[p] += 1; FN[t] += 1
    return TP, FP, FN

def precision_recall_accuracy(y_true: np.ndarray, y_pred: np.ndarray, K: int):
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
    }


# ------------------------------------------------------------
# Collate (keeps [0,1] tensors)
# ------------------------------------------------------------
def collate_raw(batch):
    imgs = torch.stack([b["image_01"] for b in batch], dim=0)  # [0,1]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    meta = [b["meta"] for b in batch]
    return {"image_01": imgs, "labels": labels, "meta": meta}


# ------------------------------------------------------------
# Evaluation (uniform τ)
# ------------------------------------------------------------
@torch.inference_mode()
def evaluate(backbone, classifier, loader, device, vae_ddim: VAEDDIMForward, clip_mean, clip_std):
    backbone.eval(); classifier.eval()
    ys, ps = [], []
    for batch in loader:
        x01 = batch["image_01"].to(device)  # [0,1]
        y = batch["labels"].cpu().numpy()

        # uniform τ
        tau = sample_uniform((x01.size(0),), device=device)
        t_idx = tau_to_index(tau, vae_ddim.T)

        # VAE encode -> z0 ; zt via DDIM.add_noise ; decode to x_t
        z0 = vae_ddim.encode_to_latents(x01)
        zt = vae_ddim.add_noise(z0, t_idx)
        xt01 = vae_ddim.decode_from_latents(zt)

        pixel_values = (xt01 - clip_mean) / clip_std

        out = backbone(pixel_values=pixel_values, timesteps=tau)
        logits = classifier(out.pooler_output)
        p = logits.argmax(dim=-1).cpu().numpy()

        ys.append(y); ps.append(p)

    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return precision_recall_accuracy(y_true, y_pred, K=classifier.weight.size(0))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # ----------------- Config -----------------
    prompt_dir = "./prompts"         # your 20 files sd_prompt_{object}.txt
    cache_dir = "./sd15_cache"       # optional SD cache
    images_per_class = 200

    seed = 20250922
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    batch_size = 8
    epochs = 10           # demo
    lr_new = 1e-4         # time-embed + AdaLN + classifier
    lr_base = 0.0         # 0 = keep CLIP backbone frozen
    weight_decay = 0.01

    # ------------------------------------------
    set_seed(seed)

    # Dataset + split
    base_ds = SDPromptFolderDataset(
        prompt_dir=prompt_dir,
        images_per_class=images_per_class,
        cache_dir=cache_dir,
        base_seed=seed,
        sd_model="runwayml/stable-diffusion-v1-5",
        device=device,
        torch_dtype=torch.float16,
        num_inference_steps=50,
        guidance_scale=7.5,
        image_size=512,
        resize_to=224,
    )
    train_set = SDClassSplitView(base_ds, split=0.8, kind="train")
    val_set   = SDClassSplitView(base_ds, split=0.8, kind="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_raw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_raw)

    # Timestep-aware backbone
    backbone = TimestepAwareCLIPVisionModel.from_pretrained_clip(
        "openai/clip-vit-base-patch32",
        gate_add_one=True, time_embed_mult=4
    ).to(device)

    # Head
    feat_dim = backbone.config.hidden_size  # 768
    classifier = CosineClassifier(in_dim=feat_dim, num_classes=20).to(device)
    classifier.load_state_dict(torch.load("clip_probe_20cls_ep4.pt")["classifier"])

    # Param groups
    def is_new_param(n): return ("time_embed" in n) or ("adaln_attn" in n) or ("adaln_mlp" in n)
    new_params, base_params = [], []
    for n, p in backbone.named_parameters():
        (new_params if is_new_param(n) else base_params).append(p)
    for p in base_params: p.requires_grad = lr_base > 0.0

    optim = torch.optim.AdamW(
        # [
        [{"params": new_params, "lr": lr_new, "weight_decay": weight_decay},
        #  {"params": base_params, "lr": lr_base, "weight_decay": weight_decay},
         {"params": classifier.parameters(), "lr": lr_new, "weight_decay": weight_decay}],
    )

    # print(f"Trainable params: {sum(p.numel() for p in new_params if p.requires_grad)} (backbone) + "
    #       f"{sum(p.numel() for p in classifier.parameters() if p.requires_grad)} (head) = "
    #       f"{sum(p.numel() for p in new_params if p.requires_grad) + sum(p.numel() for p in classifier.parameters() if p.requires_grad)} total")
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # exit()
    # VAE + DDIM forwarder
    vae_ddim = VAEDDIMForward(sd_model="runwayml/stable-diffusion-v1-5", device=device, torch_dtype=torch.float16)

    # CLIP normalization
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    mean = torch.tensor(clip_proc.image_processor.image_mean, device=device).view(1, -1, 1, 1)
    std  = torch.tensor(clip_proc.image_processor.image_std,  device=device).view(1, -1, 1, 1)

    # ----------------- Train -----------------
    backbone.train(); classifier.train()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_loader, start=1):
            x01 = batch["image_01"].to(device)      # [0,1]
            y = batch["labels"].to(device)

            # τ ~ logit-normal for training
            # tau = sample_logit_normal((x01.size(0),), device=device)
            tau = sample_uniform((x01.size(0),), device=device)
            t_idx = tau_to_index(tau, vae_ddim.T)

            # VAE encode -> z0 ; zt via DDIM.add_noise ; decode
            z0 = vae_ddim.encode_to_latents(x01)
            zt = vae_ddim.add_noise(z0, t_idx)
            xt01 = vae_ddim.decode_from_latents(zt)

            # CLIP normalize
            pixel_values = (xt01 - mean) / std

            print(pixel_values.shape, tau.shape, tau)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = backbone(pixel_values=pixel_values, timesteps=tau)
                logits = classifier(out.pooler_output)
                loss = F.cross_entropy(logits, y)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optim); scaler.update()

            if step % 50 == 0:
                with torch.no_grad():
                    acc = (logits.argmax(-1) == y).float().mean().item()
                print(f"[epoch {epoch}] step {step} | loss {loss.item():.4f} | acc {acc:.3f}")

        # ----------------- Validate (uniform τ) -----------------
        metrics = evaluate(backbone, classifier, val_loader, device, vae_ddim, mean, std)
        print("\n=== Validation (Uniform τ) ===")
        print(f"accuracy          : {metrics['accuracy']:.4f}")
        print(f"micro precision   : {metrics['micro_precision']:.4f}")
        print(f"micro recall      : {metrics['micro_recall']:.4f}")
        print(f"macro precision   : {metrics['macro_precision']:.4f}")
        print(f"macro recall      : {metrics['macro_recall']:.4f}")
        print("Per-class (alphabetical):")
        for cname, p, r in zip(base_ds.class_names, metrics["per_class_precision"], metrics["per_class_recall"]):
            print(f"  {cname:20s} P={p:.3f} R={r:.3f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "class_names": base_ds.class_names,
        "scheduler_num_steps": vae_ddim.T,
        "vae_scaling_factor": vae_ddim.scaling_factor,
    }, "checkpoints/timestep_aware_clip_ddim_vae.pt")
    print("Saved to checkpoints/timestep_aware_clip_ddim_vae.pt")


if __name__ == "__main__":
    main()
