# i2p_generate.py
import os, re, torch, random
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from tqdm import tqdm

def main():
    # ===== Config (no argparse) =====
    out_dir       = "./i2p_nsfw"
    out_dir_sd     = "./i2p_nsfw_sd"
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    seed          = 1234
    lora_path     = "logs/_nsfw_ddpo_unlearning/_nsfw_ddpo_unlearning_2025.10.16_16.31.36/checkpoints/checkpoint_8"   
    num_per_prompt = 1                
    height, width  = 512, 512
    guidance_scale = 7.5
    num_inference_steps = 50
    lora_scale     = 1.0        

    # Check if LoRA checkpoint exists
    if not os.path.exists(lora_path):
        print(f"❌ LoRA checkpoint not found at: {lora_path}")
        print(f"Please train the model using DDPO or provide a valid checkpoint path.")
        return

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    # ===== Load base pipeline =====
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    pipe_sd = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    lora_weight_path = os.path.join(lora_path, 'pytorch_lora_weights.bin')
    print(f"LoRA path: {lora_path}")
    print(f"LoRA scale: {lora_scale}")
    
    state_dict = torch.load(lora_weight_path, map_location=device)
    pipe.unet.load_attn_procs(state_dict)

    # ===== Load I2P dataset =====
    ds = load_dataset("AIML-TUDA/i2p", split="train")
    prompt_key = "prompt"

    def dummy(images, **kwargs):
        return images, [False]

    pipe.safety_checker = dummy
    pipe.set_progress_bar_config(disable=True)

    pipe_sd.safety_checker = dummy
    pipe_sd.set_progress_bar_config(disable=True)
    # ===== Generate =====
    total = len(ds) 
    print(f"Generating {total} prompts...")

    for i in tqdm(range(total)):
        ex = ds[i]
        prompt = ex[prompt_key]
        os.makedirs(out_dir, exist_ok=True)

        fname = f"{i:06d}.png"
        path = os.path.join(out_dir, fname)

        # if os.path.exists(path):
        #     print(f"Image already exists, skipping: {path}")
        #     continue

        generator = torch.Generator(device=device).manual_seed(seed + i * 1000)
        img = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]  # PIL.Image

        img.save(path, quality=95)

        # os.makedirs(out_dir_sd, exist_ok=True)
        # path = os.path.join(out_dir_sd, fname)
        # generator = torch.Generator(device=device).manual_seed(seed + i * 1000)
        # img = pipe_sd(
        #     prompt,
        #     height=height,
        #     width=width,
        #     guidance_scale=guidance_scale,
        #     num_inference_steps=num_inference_steps,
        #     generator=generator
        # ).images[0]  # PIL.Image

        # img.save(path, quality=95)

    print(f"✅ Done. Images saved under: {out_dir} {out_dir_sd}")

if __name__ == "__main__":
    main()
