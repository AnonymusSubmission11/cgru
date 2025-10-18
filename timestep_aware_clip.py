import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionEmbeddings,
    CLIPAttention,
    CLIPMLP,
)
from transformers.modeling_utils import PreTrainedModel


# ---------- Timestep embed (DiT-style) ----------
def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """
    timesteps: (B,) int or float. Works for arbitrary scale; we cast to float.
    Returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    timesteps = timesteps.float().view(-1, 1)  # (B,1)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=device).float() / half
    )  # (half,)
    args = timesteps * freqs  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


class TimestepEmbedder(nn.Module):
    """
    DiT-style: sinusoidal -> MLP -> hidden_size
    """
    def __init__(self, hidden_size: int, time_embed_dim: Optional[int] = None, mult: int = 4):
        super().__init__()
        time_embed_dim = time_embed_dim or hidden_size
        self.time_embed_dim = time_embed_dim
        self.fc1 = nn.Linear(time_embed_dim, mult * hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mult * hidden_size, hidden_size)

        # Init like DiT (defaults fine); relies on zero-inited AdaLN heads for stability.

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = _sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        t = self.fc2(self.act(self.fc1(t)))
        return t  # (B, hidden_size)


# ---------- AdaLN-Zero (DiT-style) ----------
class AdaLayerNormZero(nn.Module):
    """
    Adaptive LayerNorm Zero:
      y = LN(x) * (1 + s) + b
      residual gate g is applied to the sublayer output:
        x <- x + g * sublayer(y)

    By default gate = 1 + Linear(t) with zero init so initial behavior == CLIP.
    Set gate_add_one=False to use strict "zero" gating like DiT (residuals start at 0).
    """
    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-5, gate_add_one: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=True)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * hidden_size)
        self.to_gate = nn.Linear(cond_dim, hidden_size)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)
        self.gate_add_one = gate_add_one

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C), cond: (B, C)
        s, b = self.to_scale_shift(cond).chunk(2, dim=-1)  # (B,C), (B,C)
        y = self.norm(x) * (1 + s.unsqueeze(1)) + b.unsqueeze(1)  # broadcast over tokens
        g = self.to_gate(cond)  # (B,C)
        if self.gate_add_one:
            g = 1.0 + g
        return y, g.unsqueeze(1)  # gate shaped (B,1,C) for broadcast


# ---------- Encoder layer with AdaLN around CLIP attention & MLP ----------
class CLIPEncoderLayerAdaLN(nn.Module):
    def __init__(self, config: CLIPVisionConfig, cond_dim: int, gate_add_one: bool = True):
        super().__init__()
        self.attn = CLIPAttention(config)
        self.mlp = CLIPMLP(config)
        self.adaln_attn = AdaLayerNormZero(config.hidden_size, cond_dim, eps=config.layer_norm_eps, gate_add_one=gate_add_one)
        self.adaln_mlp = AdaLayerNormZero(config.hidden_size, cond_dim, eps=config.layer_norm_eps, gate_add_one=gate_add_one)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Attention block
        y, g_attn = self.adaln_attn(x, cond)       # y: (B,N,C), g_attn: (B,1,C)
        attn_out = self.attn(y)[0]                   # (B,N,C)
        x = x + g_attn * attn_out

        # MLP block
        y, g_mlp = self.adaln_mlp(x, cond)
        mlp_out = self.mlp(y)
        x = x + g_mlp * mlp_out
        return x


class CLIPEncoderAdaLN(nn.Module):
    def __init__(self, config: CLIPVisionConfig, cond_dim: int, gate_add_one: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            [CLIPEncoderLayerAdaLN(config, cond_dim, gate_add_one=gate_add_one)
             for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, output_hidden_states: bool = False):
        all_states = () if output_hidden_states else None
        for blk in self.layers:
            if output_hidden_states:
                all_states = all_states + (x,)
            x = blk(x, cond)
        if output_hidden_states:
            all_states = all_states + (x,)
        return x, all_states


# ---------- Vision transformer with AdaLN ----------
class TimestepAwareVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, time_embed_mult: int = 4, gate_add_one: bool = True):
        super().__init__()
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.time_embed = TimestepEmbedder(
            hidden_size=config.hidden_size,
            time_embed_dim=config.hidden_size,
            mult=time_embed_mult,
        )
        self.encoder = CLIPEncoderAdaLN(config, cond_dim=config.hidden_size, gate_add_one=gate_add_one)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        timesteps: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        """
        pixel_values: (B,3,H,W)
        timesteps: (B,) int or float
        """
        x = self.embeddings(pixel_values)  # (B, N, C) includes class token
        cond = self.time_embed(timesteps)  # (B, C)

        x = self.pre_layernorm(x)

        x, all_states = self.encoder(x, cond, output_hidden_states=output_hidden_states)
        pooled = x[:, 0, :]  # CLS
        pooled = self.post_layernorm(pooled)

        return BaseModelOutputWithPooling(
            last_hidden_state=x,
            pooler_output=pooled,
            hidden_states=all_states,
        )


# ---------- PreTrained wrapper with weight porting ----------
class TimestepAwareCLIPVisionModel(PreTrainedModel):
    config_class = CLIPVisionConfig

    def __init__(self, config: CLIPVisionConfig, time_embed_mult: int = 4, gate_add_one: bool = True):
        super().__init__(config)
        self.vision_model = TimestepAwareVisionTransformer(config, time_embed_mult=time_embed_mult, gate_add_one=gate_add_one)

    def forward(
        self,
        pixel_values: torch.Tensor,
        timesteps: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        if output_hidden_states is None:
            output_hidden_states = False
        return self.vision_model(
            pixel_values=pixel_values,
            timesteps=timesteps,
            output_hidden_states=output_hidden_states,
        )

    @classmethod
    def from_pretrained_clip(
        cls,
        clip_vision_name_or_path: str = "openai/clip-vit-base-patch32",
        gate_add_one: bool = True,
        time_embed_mult: int = 4,
        **kwargs,
    ) -> "TimestepAwareCLIPVisionModel":
        """
        Initialize from standard CLIP vision weights. All AdaLN and time-embed params are newly (zero) initialized.
        """
        # Load a vanilla CLIPVisionModel to borrow weights and config
        base = CLIPVisionModel.from_pretrained(clip_vision_name_or_path)
        cfg: CLIPVisionConfig = base.config

        # Build our model
        model = cls(cfg, time_embed_mult=time_embed_mult, gate_add_one=gate_add_one, **kwargs)

        # --- Copy weights ---
        with torch.no_grad():
            # Embeddings (patch+cls+pos)
            model.vision_model.embeddings.load_state_dict(base.vision_model.embeddings.state_dict())

            # if hasattr(base.vision_model, "pre_layernorm"):
            print("Copying pre_layernorm weights")
            model.vision_model.pre_layernorm.load_state_dict(base.vision_model.pre_layrnorm.state_dict())

            # Encoder per-layer: attn & mlp
            for i, (our_blk, their_blk) in enumerate(
                zip(model.vision_model.encoder.layers, base.vision_model.encoder.layers)
            ):
                # Attention and MLP weights match exactly
                our_blk.attn.load_state_dict(their_blk.self_attn.state_dict())
                our_blk.mlp.load_state_dict(their_blk.mlp.state_dict())

                # Copy original LayerNorm affine params into AdaLN's internal LN
                our_blk.adaln_attn.norm.load_state_dict(their_blk.layer_norm1.state_dict())
                our_blk.adaln_mlp.norm.load_state_dict(their_blk.layer_norm2.state_dict())

            # Final/post LayerNorm
            model.vision_model.post_layernorm.load_state_dict(base.vision_model.post_layernorm.state_dict())

        return model
