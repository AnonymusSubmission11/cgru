#!/usr/bin/env python3
"""
Simple script to generate images using the same process as training scripts.
"""

import sys
import os
import torch
import glob
from diffusers import StableDiffusionPipeline, DDIMScheduler
from accelerate import Accelerator

# Add project root to path
sys.path.append('..')

# Global variables

PROMPT = "Waterfall through a heart-shaped rock formation"
NUM_IMAGES = 4
OUTPUT_DIR = "generated_images"

def get_next_image_number():
    """Find the next available image number in the output directory"""
    if not os.path.exists(OUTPUT_DIR):
        return 1
    
    # Find all existing image files
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "image_*.png"))
    
    if not existing_files:
        return 1
    
    # Extract numbers from filenames and find the highest
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Extract number from filename like "image_01.png"
        try:
            number = int(filename.split('_')[1].split('.')[0])
            numbers.append(number)
        except (IndexError, ValueError):
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def generate_images():
    """Generate images using the same process as training scripts"""
    
    print(f"Generating {NUM_IMAGES} images with prompt: '{PROMPT}'")
    
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load pipeline (same as in training)
    print("Loading Stable Diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Set scheduler (same as in training)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find next available image number
    start_number = get_next_image_number()
    print(f"Starting from image number: {start_number}")
    
    # Generate images
    print("Generating images...")
    with torch.no_grad():
        # Generate images with same parameters as training
        images = pipeline(
            prompt=[PROMPT] * NUM_IMAGES,
            num_inference_steps=50,
            guidance_scale=5.0,
            height=512,
            width=512,
            generator=torch.Generator(device=device).manual_seed(42)  # Fixed seed for reproducibility
        ).images
    
    # Save images with consecutive numbering
    print(f"Saving images to {OUTPUT_DIR}/")
    for i, image in enumerate(images):
        image_number = start_number + i
        filename = f"{OUTPUT_DIR}/image_{image_number:02d}.png"
        image.save(filename)
        print(f"Saved: {filename}")
    
    print(f"Generated {len(images)} images successfully!")
    print(f"Images saved as: image_{start_number:02d}.png to image_{start_number + len(images) - 1:02d}.png")
    return images

if __name__ == "__main__":
    generate_images()
