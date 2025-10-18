#!/usr/bin/env python3
"""
Test script to evaluate unlearning progression across different checkpoints.
Generates images for cat and non-cat prompts and creates a comparison figure.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import argparse
from pathlib import Path

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
        print(f"Warning: LoRA weights not found at {lora_weights_path}")
    
    return pipeline

def generate_images(pipeline, prompts, num_images=1, seed=42):
    """Generate images for given prompts."""
    images = []
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image for: {prompt}")
        
        with torch.no_grad():
            result = pipeline(
                prompt,
                num_images_per_prompt=num_images,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            images.append(result.images[0])
    
    return images

def create_comparison_figure(all_results, output_path="unlearning_progression.png"):
    """Create a comparison figure showing progression across checkpoints."""
    n_prompts = len(list(all_results.keys())[0])  # Number of prompts
    n_checkpoints = len(all_results)  # Number of checkpoints
    
    fig, axes = plt.subplots(n_prompts, n_checkpoints, figsize=(n_checkpoints * 3, n_prompts * 3))
    if n_checkpoints == 1:
        axes = axes.reshape(-1, 1)
    if n_prompts == 1:
        axes = axes.reshape(1, -1)
    
    checkpoint_names = list(all_results.keys())
    prompt_names = list(all_results[checkpoint_names[0]].keys())
    
    for i, prompt_name in enumerate(prompt_names):
        for j, checkpoint_name in enumerate(checkpoint_names):
            ax = axes[i, j]
            
            # Display image
            image = all_results[checkpoint_name][prompt_name]
            ax.imshow(image)
            ax.axis('off')
            
            # Set titles
            if i == 0:  # Top row
                ax.set_title(f"Checkpoint {checkpoint_name.split('_')[-1]}", fontsize=10)
            if j == 0:  # Left column
                ax.set_ylabel(prompt_name, fontsize=8, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison figure saved to {output_path}")
    return fig

def main():
    parser = argparse.ArgumentParser(description="Test unlearning progression")
    parser.add_argument("--log_dir", type=str, default="logs/machine_unlearning_cat_2025.09.05_11.09.29", 
                       help="Directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, default="unlearning_test_results", 
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define test prompts
    cat_prompts = [
        "a cute orange cat sitting on a windowsill",
        "a fluffy white cat playing with a ball of yarn", 
        "a black cat with green eyes in a garden",
        "a tabby cat sleeping on a cozy blanket",
        "a Siamese cat looking at the camera",
        "a strong cat reading a book"
    ]
    
    non_cat_prompts = [
        "a golden retriever dog playing in the park",
        "a beautiful landscape with mountains and trees"
    ]
    
    all_prompts = cat_prompts + non_cat_prompts
    prompt_names = [f"cat_{i+1}" for i in range(len(cat_prompts))] + [f"non_cat_{i+1}" for i in range(len(non_cat_prompts))]
    
    # Find all checkpoints (same path structure as train.py)
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    checkpoint_paths = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint_")])
    
    print(f"Found {len(checkpoint_paths)} checkpoints: {checkpoint_paths}")
    
    # Store results for all checkpoints
    all_results = {}
    
    for checkpoint_name in checkpoint_paths:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        print(f"\n=== Processing {checkpoint_name} ===")
        
        try:
            # Load pipeline with checkpoint using same method as train.py
            pipeline = load_checkpoint_pipeline(checkpoint_path, args.device)
            
            # Generate images for all prompts with fixed seed
            images = generate_images(pipeline, all_prompts, seed=42)
            
            # Store results
            checkpoint_results = {}
            for prompt_name, image in zip(prompt_names, images):
                checkpoint_results[prompt_name] = image
                
                # Save individual image
                image_path = os.path.join(args.output_dir, f"{checkpoint_name}_{prompt_name}.png")
                image.save(image_path)
            
            all_results[checkpoint_name] = checkpoint_results
            
            # Clean up memory
            del pipeline
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {checkpoint_name}: {e}")
            continue
    
    # Create comparison figure
    if all_results:
        output_path = os.path.join(args.output_dir, "unlearning_progression.png")
        create_comparison_figure(all_results, output_path)
        
        # Also create separate figures for cat vs non-cat prompts
        cat_results = {k: {p: v for p, v in results.items() if p.startswith("cat_")} 
                      for k, results in all_results.items()}
        non_cat_results = {k: {p: v for p, v in results.items() if p.startswith("non_cat_")} 
                          for k, results in all_results.items()}
        
        if any(cat_results.values()):
            cat_output_path = os.path.join(args.output_dir, "cat_unlearning_progression.png")
            create_comparison_figure(cat_results, cat_output_path)
        
        if any(non_cat_results.values()):
            non_cat_output_path = os.path.join(args.output_dir, "non_cat_progression.png")
            create_comparison_figure(non_cat_results, non_cat_output_path)
        
        print(f"\nResults saved to {args.output_dir}/")
        print("Files created:")
        print("- unlearning_progression.png: All prompts comparison")
        print("- cat_unlearning_progression.png: Cat prompts only")
        print("- non_cat_progression.png: Non-cat prompts only")
        print("- Individual images: checkpoint_prompt.png")
    
    else:
        print("No checkpoints were successfully processed!")

if __name__ == "__main__":
    main()
