#!/usr/bin/env python3
"""
Unified reward testing script that evaluates all images in generated_images folder
with any specified reward function and optional critic function.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Global configuration - change this to select which reward function to test
SELECTED_REWARD = "cat_clip_unlearning"  # Options: aesthetic_score, cat_likeness, books_unlearning, van_gogh_style, jpeg_incompressibility, jpeg_compressibility, cat_clip_unlearning

def load_images_from_folder(folder_path):
    """Load all images from the specified folder"""
    image_files = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.jpeg"))
    image_files.sort()  # Sort for consistent ordering
    
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return images, image_files

def add_score_to_image(image, reward_score, critic_score=None, reward_name="reward", critic_name="critic"):
    """Add score text overlay to the image"""
    # Create a copy to avoid modifying the original
    img_with_score = image.copy()
    draw = ImageDraw.Draw(img_with_score)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Prepare text
    reward_text = f"{reward_name}: {reward_score:.3f}"
    text_lines = [reward_text]
    
    if critic_score is not None:
        critic_text = f"{critic_name}: {critic_score:.3f}"
        text_lines.append(critic_text)
    
    # Calculate text position (top-left corner with some padding)
    text_x = 10
    text_y = 10
    line_height = 25
    
    # Draw background rectangle for better text visibility
    max_text_width = max([draw.textlength(line, font=font) for line in text_lines])
    rect_coords = [text_x - 5, text_y - 5, text_x + max_text_width + 10, text_y + len(text_lines) * line_height + 5]
    draw.rectangle(rect_coords, fill=(0, 0, 0, 128))  # Semi-transparent black background
    
    # Draw text
    for i, line in enumerate(text_lines):
        draw.text((text_x, text_y + i * line_height), line, fill=(255, 255, 255), font=font)
    
    return img_with_score

def test_reward_function(reward_fn, reward_name, critic_fn=None, critic_name="critic"):
    """Test a reward function on all images in generated_images folder"""
    
    print(f"\n{'='*60}")
    print(f"Testing {reward_name} reward function")
    if critic_fn:
        print(f"With {critic_name} critic")
    print(f"{'='*60}")
    
    # Load images
    generated_folder = "generated_images"
    if not os.path.exists(generated_folder):
        print(f"Error: {generated_folder} folder not found!")
        return
    
    images, image_files = load_images_from_folder(generated_folder)
    if not images:
        print(f"No images found in {generated_folder}")
        return
    
    print(f"Loaded {len(images)} images")
    
    # Create output folder
    output_folder = "scored_images"
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert PIL images to numpy arrays for reward function
    image_arrays = []
    for img in images:
        img_array = np.array(img)
        image_arrays.append(img_array)
    
    image_arrays = np.array(image_arrays)
    
    # Create dummy prompts and metadata
    prompts = [f"test_prompt_{i}" for i in range(len(images))]
    metadata = [{} for _ in range(len(images))]
    
    # Evaluate with reward function
    print("Evaluating with reward function...")
    try:
        reward_scores, reward_metadata = reward_fn(image_arrays, prompts, metadata)
        print(f"Reward scores: {reward_scores}")
    except Exception as e:
        print(f"Error evaluating reward function: {e}")
        return
    
    # Evaluate with critic if provided
    critic_scores = None
    if critic_fn:
        print("Evaluating with critic...")
        try:
            # Convert images to tensor format expected by critic
            image_tensors = torch.from_numpy(image_arrays).permute(0, 3, 1, 2).float() / 255.0
            critic_scores = critic_fn(image_tensors, prompts, metadata)
            print(f"Critic scores: {critic_scores}")
            print(f"Critic mean: {np.mean(critic_scores):.3f}, std: {np.std(critic_scores):.3f}")
        except Exception as e:
            print(f"Error evaluating critic: {e}")
            critic_scores = None
    
    # Save scored images
    print("Saving scored images...")
    for i, (image, image_file, reward_score) in enumerate(zip(images, image_files, reward_scores)):
        # Get critic score for this image if available
        critic_score = critic_scores[i] if critic_scores is not None else None
        
        # Add score overlay
        scored_image = add_score_to_image(
            image, 
            reward_score, 
            critic_score, 
            reward_name, 
            critic_name
        )
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        output_filename = f"{base_name}_{reward_name}_r{reward_score:.3f}"
        if critic_score is not None:
            output_filename += f"_c{critic_score:.3f}"
        output_filename += ".png"
        
        output_path = os.path.join(output_folder, output_filename)
        scored_image.save(output_path)
        print(f"Saved: {output_path}")

def main():
    """Main function to test the selected reward function"""
    
    print("Unified Reward Testing Script")
    print("=" * 50)
    print(f"Selected reward: {SELECTED_REWARD}")
    print("=" * 50)
    
    # Initialize reward function and critic based on selection
    if SELECTED_REWARD == "aesthetic_score":
        from ddpo_pytorch.rewards import aesthetic_score, aesthetic_critic
        reward_fn = aesthetic_score()
        reward_name = "aesthetic_score"
        critic_fn = aesthetic_critic()
        critic_name = "aesthetic_critic"
        
    elif SELECTED_REWARD == "cat_likeness":
        from ddpo_pytorch.rewards import cat_likeness, cat_likeness_critic
        reward_fn = cat_likeness()
        reward_name = "cat_likeness"
        critic_fn = cat_likeness_critic()
        critic_name = "cat_critic"
        
    elif SELECTED_REWARD == "books_unlearning":
        from ddpo_pytorch.rewards import books_unlearning, books_critic
        reward_fn = books_unlearning()
        reward_name = "books_unlearning"
        critic_fn = books_critic()
        critic_name = "books_critic"
        
    elif SELECTED_REWARD == "van_gogh_style":
        from ddpo_pytorch.rewards import van_gogh_style
        reward_fn = van_gogh_style()
        reward_name = "van_gogh_style"
        critic_fn = None
        critic_name = None
        
    elif SELECTED_REWARD == "jpeg_incompressibility":
        from ddpo_pytorch.rewards import jpeg_incompressibility
        reward_fn = jpeg_incompressibility()
        reward_name = "jpeg_incompressibility"
        critic_fn = None
        critic_name = None
        
    elif SELECTED_REWARD == "jpeg_compressibility":
        from ddpo_pytorch.rewards import jpeg_compressibility
        reward_fn = jpeg_compressibility()
        reward_name = "jpeg_compressibility"
        critic_fn = None
        critic_name = None
        
    elif SELECTED_REWARD == "cat_clip_unlearning":
        from ddpo_pytorch.rewards import cat_clip_unlearning
        reward_fn = cat_clip_unlearning()
        reward_name = "cat_clip_unlearning"
        critic_fn = None
        critic_name = None
        
    else:
        print(f"Error: Unknown reward function '{SELECTED_REWARD}'")
        print("Available options: aesthetic_score, cat_likeness, books_unlearning, van_gogh_style, jpeg_incompressibility, jpeg_compressibility, cat_clip_unlearning")
        return
    
    # Run the test
    try:
        test_reward_function(reward_fn, reward_name, critic_fn, critic_name)
    except Exception as e:
        print(f"Error testing {reward_name}: {e}")

    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"Results saved in: scored_images/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
