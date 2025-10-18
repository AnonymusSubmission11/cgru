#!/usr/bin/env python3
"""
Simple script to generate images using the same process as training scripts.
"""

import sys
import os
import torch
import glob
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler

from accelerate import Accelerator

# Add project root to path
sys.path.append('..')

# Global variables

theme_available=["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
 "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
 "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
 "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
 "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
 "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
 "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
 "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]



class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                   "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                   "Trees", "Waterfalls"]

NUM_IMAGES = 10  # Total images to generate per style-class combination
BATCH_SIZE = 200  # Total number of images to generate in parallel per batch (adjust based on GPU memory)
                 # This can be much larger since we process multiple style-class combinations simultaneously
                 # Higher values = faster generation but more GPU memory usage
                 # Lower values = slower generation but less GPU memory usage
OUTPUT_DIR = "fid_generated_images"
CHKP = "checkpoints/ac_cats/checkpoint_5"

def load_checkpoint_pipeline(checkpoint_path, device="cuda"):
    """Load a pipeline with a specific checkpoint using modern diffusers 0.35.1 approach."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    
    # Load LoRA weights using the same method as training
    lora_weights_path = os.path.join(checkpoint_path, "pytorch_lora_weights.bin")
    if os.path.exists(lora_weights_path):
        print(f"Loading LoRA weights from {lora_weights_path}")
        # Load the LoRA weights directly using load_attn_procs (same as training)
        pipeline.unet.load_attn_procs(checkpoint_path)
        print("LoRA weights loaded successfully")
    else:
        raise ValueError(f"LoRA weights not found at {lora_weights_path}")
    return pipeline

def get_next_image_number(style, class_name):
    """Find the next available image number for a specific style-class combination"""
    style_class_dir = os.path.join(OUTPUT_DIR, f"{style}_{class_name}")
    if not os.path.exists(style_class_dir):
        return 1
    
    # Find all existing image files for this style-class combination
    pattern = os.path.join(style_class_dir, f"{style}_{class_name}_*.png")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract numbers from filenames and find the highest
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Extract number from filename like "style_class_01.png"
        try:
            number = int(filename.split('_')[-1].split('.')[0])
            numbers.append(number)
        except (IndexError, ValueError):
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def generate_images():
    """Generate images for each style x class combination"""
    
    total_combinations = len(theme_available) * len(class_available)
    print(f"Generating {NUM_IMAGES} images for each of {total_combinations} style x class combinations")
    print(f"Total images to generate: {total_combinations * NUM_IMAGES}")
    
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load pipeline (same as in training)
    print("Loading Stable Diffusion pipeline...")
    if CHKP:
        pipeline = load_checkpoint_pipeline(CHKP)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
    
    # Set scheduler (same as in training)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_generated = 0
    
    # Create all style-class combinations
    combinations = []
    for style in theme_available:
        for class_name in class_available:
            combinations.append((style, class_name))
    
    print(f"Created {len(combinations)} style-class combinations")
    
    # Process combinations in batches
    combination_idx = 0
    while combination_idx < len(combinations):
        # Determine how many combinations to process in this batch
        remaining_combinations = len(combinations) - combination_idx
        images_per_combination = NUM_IMAGES
        max_combinations_per_batch = BATCH_SIZE // images_per_combination
        
        if max_combinations_per_batch == 0:
            # If BATCH_SIZE is smaller than NUM_IMAGES, process one combination with smaller batches
            max_combinations_per_batch = 1
            images_per_combination = min(BATCH_SIZE, NUM_IMAGES)
        
        current_batch_combinations = min(max_combinations_per_batch, remaining_combinations)
        
        print(f"\nProcessing batch: combinations {combination_idx + 1}-{combination_idx + current_batch_combinations} of {len(combinations)}")
        
        # Prepare prompts and metadata for this batch
        batch_prompts = []
        batch_metadata = []  # Store (style, class_name, start_number) for each image
        
        for i in range(current_batch_combinations):
            style, class_name = combinations[combination_idx + i]
            
            # Create subdirectory for this style-class combination
            style_class_dir = os.path.join(OUTPUT_DIR, f"{style}_{class_name}")
            os.makedirs(style_class_dir, exist_ok=True)
            
            # Find next available image number for this combination
            start_number = get_next_image_number(style, class_name)
            
            # Create prompt in the specified format
            prompt = f"A {class_name} image in {style} style"
            
            # Add prompts and metadata for this combination
            for j in range(images_per_combination):
                batch_prompts.append(prompt)
                batch_metadata.append((style, class_name, start_number + j))
        
        print(f"Generating {len(batch_prompts)} images in one batch...")
        
        # Generate all images for this batch
        with torch.no_grad():
            batch_images = pipeline(
                prompt=batch_prompts,
                num_inference_steps=50,
                guidance_scale=5.0,
                height=512,
                width=512,
                generator=torch.Generator(device=device).manual_seed(42 + combination_idx)  # Different seed per batch
            ).images
        
        # Save images with the specified naming convention
        print(f"Saving {len(batch_images)} images...")
        for i, (image, (style, class_name, image_number)) in enumerate(zip(batch_images, batch_metadata)):
            style_class_dir = os.path.join(OUTPUT_DIR, f"{style}_{class_name}")
            filename = f"{style_class_dir}/{style}_{class_name}_{image_number:02d}.png"
            image.save(filename)
            if i % 10 == 0:  # Print progress every 10 images
                print(f"  Saved: {filename}")
        
        total_generated += len(batch_images)
        print(f"Generated {len(batch_images)} images for {current_batch_combinations} combinations")
        
        # Clear GPU cache after each batch to manage memory
        torch.cuda.empty_cache()
        
        combination_idx += current_batch_combinations
    
    print(f"\n=== Generation Complete ===")
    print(f"Total images generated: {total_generated}")
    print(f"Images saved in subdirectories under {OUTPUT_DIR}/")
    print(f"Naming format: {style}_{class_name}_number.png")

if __name__ == "__main__":
    generate_images()
