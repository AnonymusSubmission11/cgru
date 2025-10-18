#!/usr/bin/env python3
"""
Simple, non-recursive: caption images in a folder with LLaVA and
save CLIP-length-safe prompts to prompts.txt.

Run: python llava-prompts.py
"""

import os
import re
from typing import List
from PIL import Image, UnidentifiedImageError

import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers import LlavaForConditionalGeneration


# ---------- helpers ----------

def clean_text(t: str) -> str:
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip('\"""\'`')
    return t

def trim_to_clip_tokens(text: str, clip_tokenizer, max_tokens: int = 75) -> str:
    enc = clip_tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return clean_text(clip_tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True))

def make_llava_prompt(user_prompt: str) -> str:
    return f"USER: <image>\n{user_prompt}\nASSISTANT:"

def caption_images_batch(model, processor, images: List[Image.Image], user_prompt: str, max_new_tokens: int = 80) -> List[str]:
    """Process multiple images at once (MUCH faster!)"""
    prompts = [make_llava_prompt(user_prompt) for _ in images]
    
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            no_repeat_ngram_size=3,  # avoid template-y repeats
            repetition_penalty=1.1,  # gentle anti-repetition
        )
    
    gen_only = gen[:, inputs["input_ids"].shape[1]:]
    outputs = processor.batch_decode(gen_only, skip_special_tokens=True)
    return [clean_text(out) for out in outputs]


# ---------- main ----------

def main():
    # ======= Configuration =======
    IMAGE_FOLDER = "nsfw-dataset/nsfw_dataset_v1/drawings"
    OUTPUT_TXT = "drawings-prompts.txt"
    MAX_FILES = None
    
    # ===== SPEED OPTIMIZATION SETTINGS =====
    BATCH_SIZE = 8  # Process 8 images at once (adjust based on VRAM)
    # BATCH_SIZE = 4  # Use if you get OOM errors
    # BATCH_SIZE = 16  # Use if you have A100/H100
    
    USE_FLASH_ATTENTION = True  # Faster attention (if available)
    USE_COMPILE = False  # torch.compile (slower first run, faster after)
    # ========================================

    LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

    CAPTION_USER_PROMPT = (
        "Describe this image in one concise sentence (15-25 words): "
        "main subject, pose/action, setting. "
        "Include what NSFW or suggestive elements are present in detail (nudity, revealing clothing, poses, etc.). "
        "Make it sound like a stable diffusion prompt."
    )

    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    CLIP_MAX_TOKENS = 75
    MAX_NEW_TOKENS = 80
    # ====================================

    # Collect image files
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    folder = os.path.abspath(IMAGE_FOLDER)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fn)) and os.path.splitext(fn)[1].lower() in exts
    ]
    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    if not files:
        print(f"No images found in {folder}")
        return
    print(f"Found {len(files)} images in {folder}")

    # Load models
    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {LLAVA_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    
    # Optional: torch.compile for faster inference (PyTorch 2.0+)
    if USE_COMPILE and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    clip_tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_ID)
    print(f"Model loaded successfully!")

    # Fresh output
    if os.path.exists(OUTPUT_TXT):
        os.remove(OUTPUT_TXT)

    written = 0
    errors = 0
    
    # Process in batches
    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        for batch_start in range(0, len(files), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(files))
            batch_files = files[batch_start:batch_end]
            
            # Load images for this batch
            batch_images = []
            batch_fnames = []
            
            for path in batch_files:
                fname = os.path.basename(path)
                try:
                    with Image.open(path) as im:
                        batch_images.append(im.convert("RGB").copy())  # .copy() to keep in memory
                        batch_fnames.append(fname)
                except (UnidentifiedImageError, OSError) as e:
                    print(f"❌ Cannot read {fname}: {e}")
                    f.write(f"unreadable image file\n")
                    errors += 1
            
            if not batch_images:
                continue
            
            # Process batch
            try:
                captions = caption_images_batch(
                    model=model,
                    processor=processor,
                    images=batch_images,
                    user_prompt=CAPTION_USER_PROMPT,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                
                # Write results
                for caption in captions:
                    if not caption or len(caption) < 10:
                        caption = "A photographic image with no text or additional objects."
                    
                    prompt = trim_to_clip_tokens(caption, clip_tokenizer, CLIP_MAX_TOKENS)
                    f.write(f"{prompt}\n")
                    written += 1
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_start}-{batch_end}: {e}")
                # Write fallback for all images in batch
                for _ in batch_images:
                    f.write(f"A photographic image with no text or additional objects.\n")
                    written += 1
                errors += len(batch_images)
            
            # Progress update
            if (batch_end) % (BATCH_SIZE * 5) == 0:
                print(f"✅ Processed {batch_end}/{len(files)} images... ({errors} errors)")

    print(f"\n{'='*60}")
    print(f"Done! Wrote {written} prompts to {OUTPUT_TXT}")
    print(f"Errors: {errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
