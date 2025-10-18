from importlib import resources
import os
import functools
import random
import inflect
import json

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")


def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata


def cats_50_percent():
    """Dataset with 50% cat prompts and 50% other animals"""
    return from_file("cats_50_percent.txt")


def cats_90_percent():
    """Dataset with 90% cat prompts and 10% other animals"""
    return from_file("cats_90_percent.txt")


def cats_100_percent():
    """Dataset with 100% cat prompts"""
    return from_file("cats_100_percent.txt")

def books_diffusion_dataset():
    """Dataset for books diffusion model training"""
    return from_file("books_diffusion_dataset.txt")


def van_gogh_dataset():
    """Dataset for van gogh diffusion model training"""
    return from_file("van_gogh_diffusion_dataset.txt")


# Class-specific dataset functions for unlearning
def architectures_clip_dataset():
    """Dataset for architectures CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Architectures.txt")


def bears_clip_dataset():
    """Dataset for bears CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Bears.txt")


def birds_clip_dataset():
    """Dataset for birds CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Birds.txt")


def butterfly_clip_dataset():
    """Dataset for butterfly CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Butterfly.txt")


def cats_clip_dataset():
    """Dataset for cats CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Cats.txt")


def dogs_clip_dataset():
    """Dataset for dogs CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Dogs.txt")


def fishes_clip_dataset():
    """Dataset for fishes CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Fishes.txt")


def flame_clip_dataset():
    """Dataset for flame CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Flame.txt")


def flowers_clip_dataset():
    """Dataset for flowers CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Flowers.txt")


def frogs_clip_dataset():
    """Dataset for frogs CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Frogs.txt")


def horses_clip_dataset():
    """Dataset for horses CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Horses.txt")


def human_clip_dataset():
    """Dataset for human CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Human.txt")


def jellyfish_clip_dataset():
    """Dataset for jellyfish CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Jellyfish.txt")


def rabbits_clip_dataset():
    """Dataset for rabbits CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Rabbits.txt")


def sandwiches_clip_dataset():
    """Dataset for sandwiches CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Sandwiches.txt")


def sea_clip_dataset():
    """Dataset for sea CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Sea.txt")


def statues_clip_dataset():
    """Dataset for statues CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Statues.txt")


def towers_clip_dataset():
    """Dataset for towers CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Towers.txt")


def trees_clip_dataset():
    """Dataset for trees CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Trees.txt")


def waterfalls_clip_dataset():
    """Dataset for waterfalls CLIP classifier training"""
    return from_file("class_rem_2_prompts/sd_prompt_Waterfalls.txt")


def nsfw():
    """
    Load prompts from LLaVA-generated NSFW prompts.
    File format: one prompt per line (no filename prefix).
    """
    return from_file("nsfw/nsfw-dataset-llava.txt")

def nsfw_nibbler(): # https://github.com/google-research-datasets/adversarial-nibbler/tree/main
    return from_file("nsfw-dataset-nibbler-submitted/nsfw-prompts.txt")

def _sample_from_split(split_path, field, source_tag):
    """
    split_path: e.g. 'lg_splits/cgru_critic_5.jsonl'
    field: 'unsafe_prompt' or 'safe_prompt'
    source_tag: 'critic_5' or 'policy_10'
    """
    items = _load_jsonl(split_path)
    ex = random.choice(items)
    prompt = ex[field].strip()
    meta = {
        "concept":  ex.get("concept", ""),
        "category": ex.get("category", ""),
        "label":    "unsafe" if field == "unsafe_prompt" else "safe",
        "source":   source_tag,
    }
    return prompt, meta

# -------- Convenience functions --------

def copro_critic_unsafe():
    """Random UNSAFE prompt from 5% critic split."""
    return _sample_from_split("lg_splits/cgru_critic_5.jsonl", "unsafe_prompt", "critic_5")

def copro_critic_safe():
    """Random SAFE prompt from 5% critic split."""
    return _sample_from_split("lg_splits/cgru_critic_5.jsonl", "safe_prompt", "critic_5")

def copro_policy_unsafe():
    """Random UNSAFE prompt from 10% policy split."""
    return _sample_from_split("lg_splits/ddpo_policy_10.jsonl", "unsafe_prompt", "policy_10")

def copro_policy_safe():
    """Random SAFE prompt from 10% policy split."""
    return _sample_from_split("lg_splits/ddpo_policy_10.jsonl", "safe_prompt", "policy_10")