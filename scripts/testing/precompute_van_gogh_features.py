#!/usr/bin/env python3
"""
Pre-compute and save Van Gogh style features using reference implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm

def precompute_van_gogh_features():
    """Pre-compute and save Van Gogh style features using reference implementation"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading VGG19 model...")
    # Load VGG19 exactly like reference
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    # Normalization exactly like reference
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # Style layers exactly like reference
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    # Image preprocessing exactly like reference
    imsize = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    
    def image_loader(image):
        """Load and preprocess image exactly like reference"""
        image_tensor = loader(image).unsqueeze(0)
        return image_tensor.to(device, torch.float)
    
    def gram_matrix(input):
        """Gram matrix computation exactly like reference"""
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)  # normalize by number of elements
    
    class StyleLoss(nn.Module):
        """Style loss exactly like reference"""
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
        
        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
    
    class Normalization(nn.Module):
        """Normalization exactly like reference"""
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
        
        def forward(self, img):
            return (img - self.mean) / self.std
    
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, style_layers):
        """Get style model exactly like reference"""
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        style_losses = []
        model = nn.Sequential(normalization)
        
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
            model.add_module(name, layer)
            
            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        
        # Trim off layers after last style loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        
        return model, style_losses
    
    # Load all Van Gogh paintings
    van_gogh_dir = "van_gogh_data/images"
    van_gogh_files = glob.glob(os.path.join(van_gogh_dir, "*.jpg")) + glob.glob(os.path.join(van_gogh_dir, "*.jpeg")) + glob.glob(os.path.join(van_gogh_dir, "*.png"))
    
    print(f"Found {len(van_gogh_files)} Van Gogh paintings")
    
    # Create assets directory if it doesn't exist
    os.makedirs("ddpo_pytorch/assets", exist_ok=True)
    
    # Precompute style features for each painting
    van_gogh_style_features = []
    successful_files = []
    
    print("Extracting features from Van Gogh paintings...")
    
    with torch.no_grad():
        for file_path in tqdm(van_gogh_files):
            try:
                style_img = image_loader(Image.open(file_path).convert('RGB'))
                model, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, style_layers_default)
                
                # Extract style features
                model(style_img)
                style_features = [sl.target for sl in style_losses]
                
                van_gogh_style_features.append(style_features)
                successful_files.append(file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    print(f"Successfully processed {len(van_gogh_style_features)} Van Gogh paintings")
    
    # Save features to .pt file
    features_path = "ddpo_pytorch/assets/van_gogh_style_features.pt"
    torch.save({
        'features': van_gogh_style_features,
        'files': successful_files,
        'style_layers': style_layers_default
    }, features_path)
    
    print(f"Saved features to {features_path}")
    print(f"Features shape: {len(van_gogh_style_features)} paintings, {len(van_gogh_style_features[0])} layers each")
    
    return features_path

if __name__ == "__main__":
    precompute_van_gogh_features()
