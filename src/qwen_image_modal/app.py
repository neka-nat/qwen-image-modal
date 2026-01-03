"""Modal app: Qwen-Image-2.1 inference on a GPU.

Usage (see README for details):

    modal run -m qwen_image_modal.app::download_model   # once, fills the cache Volume
    modal deploy -m qwen_image_modal.app                # publishes the Inference class
    modal run -m qwen_image_modal.app --prompt "..."    # dev-time generation with streamed logs
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Optional

import modal

from qwen_image_modal.config import (
    APP_NAME,
    CACHE_DIR,
    DEFAULT_STEPS,
    DEFAULT_TRUE_CFG_SCALE,
    DIFFUSERS_COMMIT,
    GPU,
    MODEL_ID,
    MODEL_REVISION,
    VOLUME_NAME,
)

MINUTES = 60

image = (
    modal.Image.debian_slim(python_version="3.12")  # 3.13 needs Image Builder >= 2024.10 (workspace setting)
    .apt_install("git")
    .uv_pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",  # Qwen3VLProcessor imports torchvision
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "huggingface-hub[hf_xet]==1.33.0",
        "pillow==12.3.0",
        f"diffusers @ git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}",
    )
    .env(
        {
            "HF_HUB_CACHE": CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("qwen_image_modal")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(volumes={CACHE_DIR: volume}, timeout=60 * MINUTES, cpu=4, memory=8192)
def download_model() -> str:
    """Prefetch the model weights into the cache Volume on a cheap CPU container."""
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    volume.commit()
    print(f"downloaded {MODEL_ID}@{MODEL_REVISION[:8]} to {path} in {time.perf_counter() - t0:.0f}s")
    return path


@app.cls(
    gpu=GPU,
    volumes={CACHE_DIR: volume},
    timeout=15 * MINUTES,
    scaledown_window=10 * MINUTES,
)
class Inference:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        t0 = time.perf_counter()
        self.pipe = QwenImage21Pipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16
        ).to("cuda")
        print(f"pipeline loaded on {GPU} in {time.perf_counter() - t0:.1f}s")

    @modal.method()
    def generate(
        self,
        prompt: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        output_resolution: int = 1024,
        steps: int = DEFAULT_STEPS,
        seed: int = 0,
        num_images: int = 1,
        negative_prompt: Optional[str] = None,
        true_cfg_scale: float = DEFAULT_TRUE_CFG_SCALE,
        rgba: bool = False,
        condition_images: Optional[list[bytes]] = None,
    ) -> list[dict[str, Any]]:
        """Generate ``num_images`` images, one at a time, with seeds ``seed, seed+1, ...``.

        Returns a list of dicts with the PNG bytes and the parameters that
        produced them. Each PNG also carries the parameters in a ``parameters``
        text chunk so a file is reproducible on its own.
        """
        import torch
        from PIL import Image, PngImagePlugin

        images = None
        if condition_images:
            images = [Image.open(io.BytesIO(b)).convert("RGBA") for b in condition_images]

        results: list[dict[str, Any]] = []
        for i in range(num_images):
            this_seed = seed + i
            generator = torch.Generator("cuda").manual_seed(this_seed)
            t0 = time.perf_counter()
            out = self.pipe(
                prompt=prompt,
                image=images,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                width=width,
                height=height,
                output_resolution=output_resolution,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
            elapsed = time.perf_counter() - t0
            if not rgba:
                out = out.convert("RGB")

            params = {
                "model": f"{MODEL_ID}@{MODEL_REVISION}",
                "diffusers_commit": DIFFUSERS_COMMIT,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "true_cfg_scale": true_cfg_scale,
                "width": out.width,
                "height": out.height,
                "steps": steps,
                "seed": this_seed,
                "rgba": rgba,
                "num_condition_images": len(images) if images else 0,
                "gpu": GPU,
                "elapsed_s": round(elapsed, 2),
            }
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", json.dumps(params, ensure_ascii=False))
            buf = io.BytesIO()
            out.save(buf, format="PNG", pnginfo=info)
            print(f"image {i + 1}/{num_images} seed={this_seed} {out.width}x{out.height} {elapsed:.1f}s")
            results.append({"png": buf.getvalue(), **params})
        return results


@app.local_entrypoint()
def main(
    prompt: str,
    aspect: str = "1:1",
    size: str = "2k",
    steps: int = DEFAULT_STEPS,
    seed: Optional[int] = None,
    num_images: int = 1,
    negative_prompt: Optional[str] = None,
    cfg: float = DEFAULT_TRUE_CFG_SCALE,
    rgba: bool = False,
    out: str = "outputs",
) -> None:
    """Dev entrypoint for ``modal run``; the installed CLI uses the deployed app instead."""
    from qwen_image_modal.cli import build_request, save_results

    request = build_request(
        prompt=prompt,
        aspect=aspect,
        size=size,
        width=None,
        height=None,
        steps=steps,
        seed=seed,
        num_images=num_images,
        negative_prompt=negative_prompt,
        cfg=cfg,
        rgba=rgba,
        image_paths=[],
    )
    results = Inference().generate.remote(**request)
    for path in save_results(results, out, slug_text=prompt):
        print(path)
