"""Local CLI that calls the deployed Modal app.

The CLI depends only on ``modal``; all GPU libraries live in the remote image.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from qwen_image_modal.config import (
    APP_NAME,
    ASPECT_RATIOS,
    DEFAULT_STEPS,
    DEFAULT_TRUE_CFG_SCALE,
    INFERENCE_CLASS_NAME,
    RGBA_PROMPT_TEMPLATE,
    SIZE_PRESETS,
    resolve_size,
    validate_dimension,
)

MAX_SEED = 2**31 - 1


def build_request(
    *,
    prompt: str,
    aspect: str,
    size: str,
    width: Optional[int],
    height: Optional[int],
    steps: int,
    seed: Optional[int],
    num_images: int,
    negative_prompt: Optional[str],
    cfg: float,
    rgba: bool,
    image_paths: list[Path],
) -> dict[str, Any]:
    """Turn CLI options into keyword arguments for ``Inference.generate``."""
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if num_images <= 0:
        raise ValueError("--num-images must be positive")
    if cfg < 1.0:
        raise ValueError("--cfg must be >= 1.0 (1.0 disables guidance)")
    if negative_prompt and cfg <= 1.0:
        print("warning: --negative-prompt is ignored unless --cfg > 1.0", file=sys.stderr)

    if (width is None) != (height is None):
        raise ValueError("--width and --height must be given together")
    if width is not None and height is not None:
        w, h = validate_dimension(width, "--width"), validate_dimension(height, "--height")
    elif image_paths:
        # Editing: let the pipeline derive the size from the last condition image.
        w = h = None
    else:
        w, h = resolve_size(aspect, size)

    condition_images = [p.read_bytes() for p in image_paths] or None
    final_prompt = RGBA_PROMPT_TEMPLATE.format(prompt=prompt.rstrip(". ")) if rgba else prompt

    return {
        "prompt": final_prompt,
        "width": w,
        "height": h,
        "output_resolution": SIZE_PRESETS[size],
        "steps": steps,
        "seed": seed if seed is not None else random.randint(0, MAX_SEED),
        "num_images": num_images,
        "negative_prompt": negative_prompt or None,
        "true_cfg_scale": cfg,
        "rgba": rgba,
        "condition_images": condition_images,
    }


def save_results(
    results: list[dict[str, Any]], out_dir: str | Path, slug_text: Optional[str] = None
) -> list[Path]:
    """Write each PNG plus a JSON sidecar; return the PNG paths.

    ``slug_text`` (normally the user's original prompt) names the files; it
    falls back to the prompt actually sent, which may be the RGBA-wrapped one.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    paths: list[Path] = []
    for i, r in enumerate(results):
        slug = _slugify(slug_text or r["prompt"])
        stem = f"{stamp}_{slug}_seed{r['seed']}" if len(results) == 1 else f"{stamp}_{slug}_seed{r['seed']}_{i:02d}"
        png_path = out / f"{stem}.png"
        png_path.write_bytes(r["png"])
        meta = {k: v for k, v in r.items() if k != "png"}
        (out / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        paths.append(png_path)
    return paths


def _slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "image"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="qwen-image-modal",
        description="Generate images with Qwen-Image-2.1 on Modal.",
    )
    p.add_argument("prompt", help="text prompt")
    p.add_argument("--aspect", choices=ASPECT_RATIOS, default="1:1", help="aspect ratio preset (default: 1:1)")
    p.add_argument("--size", choices=tuple(SIZE_PRESETS), default="2k", help="size preset (default: 2k)")
    p.add_argument("--width", type=int, help="explicit width (multiple of 32); overrides --aspect/--size")
    p.add_argument("--height", type=int, help="explicit height (multiple of 32); overrides --aspect/--size")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"denoising steps (default: {DEFAULT_STEPS})")
    p.add_argument("--seed", type=int, help="random seed; images use seed, seed+1, ... (default: random)")
    p.add_argument("-n", "--num-images", type=int, default=1, help="number of images (default: 1)")
    p.add_argument("--negative-prompt", help="negative prompt; only used when --cfg > 1")
    p.add_argument(
        "--cfg",
        type=float,
        default=DEFAULT_TRUE_CFG_SCALE,
        help="true CFG scale; 1.0 disables guidance as Qwen recommends (default: 1.0)",
    )
    p.add_argument("--rgba", action="store_true", help="generate a transparent PNG (wraps the prompt)")
    p.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="condition image for editing; repeat for multiple references",
    )
    p.add_argument("-o", "--out", type=Path, default=Path("outputs"), help="output directory (default: ./outputs)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    for path in args.image:
        if not path.is_file():
            print(f"error: condition image not found: {path}", file=sys.stderr)
            return 2
    try:
        request = build_request(
            prompt=args.prompt,
            aspect=args.aspect,
            size=args.size,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            num_images=args.num_images,
            negative_prompt=args.negative_prompt,
            cfg=args.cfg,
            rgba=args.rgba,
            image_paths=args.image,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    import modal

    size_desc = f"{request['width']}x{request['height']}" if request["width"] else f"~{request['output_resolution']}px"
    print(
        f"generating {request['num_images']} image(s) {size_desc} steps={request['steps']} "
        f"seed={request['seed']} via Modal app '{APP_NAME}' ...",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    call = None
    try:
        inference = modal.Cls.from_name(APP_NAME, INFERENCE_CLASS_NAME)()
        # spawn + get (instead of .remote) so Ctrl-C can cancel the GPU work remotely.
        call = inference.generate.spawn(**request)
        results = call.get()
    except modal.exception.NotFoundError:
        print(
            f"error: Modal app '{APP_NAME}' is not deployed. Run:\n"
            "  modal run -m qwen_image_modal.app::download_model\n"
            "  modal deploy -m qwen_image_modal.app",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        if call is not None:
            call.cancel()
            print("\ncancelled remote generation", file=sys.stderr)
        return 130
    total = time.perf_counter() - t0

    paths = save_results(results, args.out, slug_text=args.prompt)
    gen_time = sum(r["elapsed_s"] for r in results)
    print(f"done in {total:.1f}s (generation {gen_time:.1f}s, rest is cold start + transfer)", file=sys.stderr)
    for path in paths:
        print(path)
    return 0
