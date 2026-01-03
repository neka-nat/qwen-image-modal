"""Static configuration shared by the Modal app and the local CLI.

Everything that pins a version lives here so that a single edit moves the
whole project to a new model revision or a new diffusers commit.
"""

from __future__ import annotations

import math
import os

# --- Modal ----------------------------------------------------------------
APP_NAME = os.environ.get("QWEN_IMAGE_MODAL_APP", "qwen-image-modal")
INFERENCE_CLASS_NAME = "Inference"
GPU = os.environ.get("QWEN_IMAGE_MODAL_GPU", "H100")
VOLUME_NAME = "qwen-image-hf-cache"
CACHE_DIR = "/cache"

# --- Model ----------------------------------------------------------------
MODEL_ID = "Qwen/Qwen-Image-2.1"
# Commit of Qwen/Qwen-Image-2.1 on the Hugging Face Hub (2026-09-21).
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
# QwenImage21Pipeline is only on diffusers main (not in 0.40.0), so pin a commit.
DIFFUSERS_COMMIT = "e0abab83b5df05de9e7abd788643c1a7c1e42e28"  # main @ 2026-09-26

# --- Generation defaults ----------------------------------------------------
DEFAULT_STEPS = 40  # Qwen's recommendation
DEFAULT_TRUE_CFG_SCALE = 1.0  # model is meant to be sampled without guidance
SIZE_MULTIPLE = 32  # height/width must be divisible by vae_scale_factor * 2

# Official 2K presets from the model card, keyed by aspect ratio.
ASPECT_2K: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
ASPECT_RATIOS = tuple(ASPECT_2K)
SIZE_PRESETS: dict[str, int] = {"1k": 1024, "2k": 2048}

RGBA_PROMPT_TEMPLATE = (
    "This is an RGBA image with transparency. {prompt}. "
    "The image has alpha channel and the background is transparent."
)


def resolve_size(aspect: str, size: str) -> tuple[int, int]:
    """Return (width, height) for an aspect-ratio preset and a size preset.

    2K uses the official table verbatim. Other sizes keep the same area as
    ``side * side`` and round each edge to a multiple of 32, mirroring
    diffusers' ``calculate_dimensions``.
    """
    if aspect not in ASPECT_2K:
        raise ValueError(f"unknown aspect {aspect!r}; choose from {', '.join(ASPECT_RATIOS)}")
    if size not in SIZE_PRESETS:
        raise ValueError(f"unknown size {size!r}; choose from {', '.join(SIZE_PRESETS)}")
    if size == "2k":
        return ASPECT_2K[aspect]
    side = SIZE_PRESETS[size]
    w2k, h2k = ASPECT_2K[aspect]
    ratio = w2k / h2k
    width = math.sqrt(side * side * ratio)
    height = width / ratio
    return _round_to_multiple(width), _round_to_multiple(height)


def _round_to_multiple(value: float, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, round(value / multiple) * multiple)


def validate_dimension(value: int, name: str) -> int:
    if value <= 0 or value % SIZE_MULTIPLE != 0:
        raise ValueError(f"{name} must be a positive multiple of {SIZE_MULTIPLE}, got {value}")
    return value
