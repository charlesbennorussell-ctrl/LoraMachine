"""
Training configuration presets for different use cases.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""

    name: str
    steps: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 1
    resolution: int = 1024
    trigger_word: str = "ohwx"
    lora_rank: int = 16
    lora_alpha: int = 16
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_every_n_steps: int = 500
    mixed_precision: str = "bf16"
    seed: Optional[int] = None


# Preset configurations for different scenarios
PRESETS = {
    "quick_test": TrainingConfig(
        name="quick_test",
        steps=100,
        resolution=512,
        learning_rate=5e-5,
    ),
    "person_portrait": TrainingConfig(
        name="person",
        steps=1500,
        resolution=1024,
        learning_rate=1e-4,
        trigger_word="ohwx person",
    ),
    "art_style": TrainingConfig(
        name="style",
        steps=2000,
        resolution=1024,
        learning_rate=5e-5,
        lora_rank=32,
        trigger_word="in the style of ohwx",
    ),
    "object": TrainingConfig(
        name="object",
        steps=1000,
        resolution=1024,
        learning_rate=1e-4,
        trigger_word="ohwx object",
    ),
    "character": TrainingConfig(
        name="character",
        steps=2000,
        resolution=1024,
        learning_rate=1e-4,
        lora_rank=32,
        lora_alpha=32,
        trigger_word="ohwx character",
    ),
}


def get_preset(name: str) -> TrainingConfig:
    """Get a training preset by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def get_optimal_config(
    num_images: int,
    concept_type: str = "person",
    vram_gb: float = 16.0
) -> TrainingConfig:
    """
    Get an optimal configuration based on the number of training images
    and available VRAM.

    Args:
        num_images: Number of training images
        concept_type: Type of concept (person, style, object, character)
        vram_gb: Available VRAM in GB
    """
    # Base steps calculation
    if num_images < 5:
        base_steps = 200 * num_images
    elif num_images < 15:
        base_steps = 100 * num_images + 500
    else:
        base_steps = 50 * num_images + 1000

    # Adjust for concept type
    type_multipliers = {
        "person": 1.0,
        "style": 1.5,
        "object": 0.8,
        "character": 1.3,
    }
    steps = int(base_steps * type_multipliers.get(concept_type, 1.0))

    # Resolution based on VRAM
    if vram_gb >= 24:
        resolution = 1024
        batch_size = 2
    elif vram_gb >= 16:
        resolution = 1024
        batch_size = 1
    elif vram_gb >= 12:
        resolution = 768
        batch_size = 1
    else:
        resolution = 512
        batch_size = 1

    # LoRA rank based on concept complexity
    lora_rank = 16 if concept_type in ["person", "object"] else 32

    # Trigger word suggestions
    trigger_words = {
        "person": "ohwx person",
        "style": "in the style of ohwx",
        "object": "ohwx",
        "character": "ohwx character",
    }

    return TrainingConfig(
        name=f"{concept_type}_lora",
        steps=steps,
        resolution=resolution,
        batch_size=batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_rank,
        trigger_word=trigger_words.get(concept_type, "ohwx"),
        learning_rate=1e-4 if concept_type != "style" else 5e-5,
    )
