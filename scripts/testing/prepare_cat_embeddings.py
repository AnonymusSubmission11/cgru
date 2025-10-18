#!/usr/bin/env python3
"""
Prepare cat embeddings from COCO dataset for cat-likeness reward function.
Downloads COCO dataset, extracts cat images, and computes CLIP embeddings.
"""

import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import requests
from tqdm import tqdm
import pickle
from transformers import CLIPModel, CLIPProcessor

def download_coco_data():
    """Download COCO annotations and images"""
    os.makedirs("coco_data", exist_ok=True)
    
    # Download annotations
    ann_file = "coco_data/annotations/instances_train2017.json"
    if not os.path.exists(ann_file):
        os.makedirs("coco_data/annotations", exist_ok=True)
        print("Downloading COCO annotations...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        os.system(f"wget {url} -O coco_data/annotations.zip")
        os.system(f"cd coco_data && unzip annotations.zip")
    
    return ann_file

def extract_cat_images(ann_file, max_images=1000):
    """Extract cat images from COCO dataset"""
    coco = COCO(ann_file)
    
    # Get cat category ID
    cat_ids = coco.getCatIds(catNms=['cat'])
    if not cat_ids:
        raise ValueError("No cat category found in COCO dataset")
    cat_id = cat_ids[0]
    
    # Get all image IDs containing cats
    img_ids = coco.getImgIds(catIds=[cat_id])
    print(f"Found {len(img_ids)} images with cats")
    
    # Limit to max_images for efficiency
    if len(img_ids) > max_images:
        img_ids = img_ids[:max_images]
        print(f"Limited to {max_images} images")
    
    # Download images
    images_dir = "coco_data/images"
    os.makedirs(images_dir, exist_ok=True)
    
    cat_images = []
    for img_id in tqdm(img_ids, desc="Downloading cat images"):
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        
        # Download image
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
            except:
                continue
        
        if os.path.exists(img_path):
            cat_images.append(img_path)
    
    print(f"Successfully downloaded {len(cat_images)} cat images")
    return cat_images

def compute_cat_embeddings(cat_images, batch_size=32):
    """Compute CLIP embeddings for cat images"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    embeddings = []
    
    for i in tqdm(range(0, len(cat_images), batch_size), desc="Computing embeddings"):
        batch_images = cat_images[i:i+batch_size]
        
        # Load and process images
        pil_images = []
        for img_path in batch_images:
            try:
                img = Image.open(img_path).convert("RGB")
                pil_images.append(img)
            except:
                continue
        
        if not pil_images:
            continue
            
        # Process with CLIP
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"Computed {all_embeddings.shape[0]} cat embeddings")
    
    return all_embeddings

def main():
    print("Preparing cat embeddings from COCO dataset...")
    
    # Download COCO data
    ann_file = download_coco_data()
    
    # Extract cat images
    cat_images = extract_cat_images(ann_file, max_images=2000)
    
    # Compute embeddings
    cat_embeddings = compute_cat_embeddings(cat_images)
    
    # Save embeddings
    os.makedirs("ddpo_pytorch/assets", exist_ok=True)
    torch.save(cat_embeddings, "ddpo_pytorch/assets/cat_embeddings.pt")
    
    print(f"Saved {cat_embeddings.shape[0]} cat embeddings to ddpo_pytorch/assets/cat_embeddings.pt")
    print(f"Embedding dimension: {cat_embeddings.shape[1]}")

if __name__ == "__main__":
    main()
