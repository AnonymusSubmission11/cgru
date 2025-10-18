"""
Standalone CriticModel class for use in DDPO training.
This avoids importing datasets and other heavy dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""
    
    def __init__(self, feature_dim, timestep_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.timestep_dim = timestep_dim
        
        # Generate scale and shift parameters from timestep embedding
        self.scale_net = nn.Sequential(
            nn.Linear(timestep_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.shift_net = nn.Sequential(
            nn.Linear(timestep_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features, timestep_embed):
        """
        Args:
            features: (batch_size, feature_dim)
            timestep_embed: (batch_size, timestep_dim)
        """
        scale = self.scale_net(timestep_embed)
        shift = self.shift_net(timestep_embed)
        return scale * features + shift


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (like in diffusion models)"""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch_size,)
        """
        half_dim = self.embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=timesteps.device)], dim=-1)
        
        return emb


class CriticModel(nn.Module):
    """Critic model using same architecture as aesthetic scorer but with FiLM modulation"""
    
    def __init__(self, dtype=torch.float32):
        super().__init__()
        
        # Load CLIP model (same as aesthetic scorer)
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(embedding_dim=128)
        
        # MLP layers (same as aesthetic scorer)
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        
        # FiLM layers for timestep modulation
        self.film_layers = nn.ModuleList([
            FiLMLayer(1024, 128),  # After first linear layer
            FiLMLayer(128, 128),   # After second linear layer
            FiLMLayer(64, 128),    # After third linear layer
        ])
        
        self.dtype = dtype
        self.eval()
    
    def forward(self, images, timesteps, prompts=None):
        """
        Args:
            images: PIL Images or tensor images
            timesteps: (batch_size,)
            prompts: Optional prompts (not used in this model)
        """
        device = next(self.parameters()).device
        
        # Process images with CLIP (same as aesthetic scorer)
        if isinstance(images, torch.Tensor):
            # Convert tensor to PIL if needed
            if images.dim() == 4:  # (batch, channels, height, width)
                images = [(img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for img in images]
                images = [Image.fromarray(img) for img in images]
        
        inputs = self.processor(images=images, return_tensors="pt")
        # Use float32 for CLIP inputs to avoid dtype mismatch
        inputs = {k: v.to(torch.float32).to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embed = self.clip.get_image_features(**inputs)
            # normalize embedding
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        
        # Get timestep embedding and ensure it's on the correct device
        timestep_embed = self.timestep_embedding(timesteps.to(device))
        
        # Forward through MLP with FiLM modulation
        x = embed
        
        # First layer
        x = self.layers[0](x)  # Linear(768, 1024)
        x = self.film_layers[0](x, timestep_embed)  # FiLM modulation
        x = self.layers[1](x)  # Dropout(0.2)
        
        # Second layer
        x = self.layers[2](x)  # Linear(1024, 128)
        x = self.film_layers[1](x, timestep_embed)  # FiLM modulation
        x = self.layers[3](x)  # Dropout(0.2)
        
        # Third layer
        x = self.layers[4](x)  # Linear(128, 64)
        x = self.film_layers[2](x, timestep_embed)  # FiLM modulation
        x = self.layers[5](x)  # Dropout(0.1)
        
        # Final layers (no modulation)
        x = self.layers[6](x)  # Linear(64, 16)
        x = self.layers[7](x)  # Linear(16, 1)
        
        return x.squeeze(-1)
