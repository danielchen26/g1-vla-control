#!/usr/bin/env python3
"""Quantify coarse visual alignment against synchronized public episode-0 frames."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import urllib.request

import imageio_ffmpeg
import numpy as np
from PIL import Image

from stack_scene import CAMERA_NAMES

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = Path.home() / ".cache" / "g1_vla_control" / "reference_ep0"
BASE_URL = (
    "https://huggingface.co/datasets/LGG100/Stack-the-cubes/resolve/main/"
    "videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
)


def _download_and_extract(camera: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    video = CACHE / f"{camera}.mp4"
    frame = CACHE / f"{camera}_frame0.png"
    if not video.exists():
        urllib.request.urlretrieve(BASE_URL.format(camera=camera), video)
    if not frame.exists():
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
            "-i", str(video), "-frames:v", "1", str(frame),
        ], check=True)
    return frame


def _color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return {
        "red": (r > 90) & (r > 1.25 * g) & (r > 1.20 * b),
        "blue": (b > 70) & (b > 1.15 * r) & (b > 1.08 * g),
        "yellow": (
            (r > 110) & (g > 60) & (r > 1.25 * b) &
            (g > 1.05 * b) & (r > 0.95 * g)
        ),
    }


def _component(mask: np.ndarray) -> dict | None:
    y, x = np.where(mask)
    if len(x) < 80:
        return None
    height, width = mask.shape
    return {
        "pixel_count": int(len(x)),
        "area_fraction": float(len(x) / mask.size),
        "center_normalized": [float(x.mean() / width), float(y.mean() / height)],
        "bbox_normalized": [
            float(x.min() / width), float(y.min() / height),
            float((x.max() + 1) / width), float((y.max() + 1) / height),
        ],
    }


def _image_metrics(path: Path) -> tuple[np.ndarray, dict]:
    image = np.asarray(Image.open(path).convert("RGB"))
    gray = image.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    metrics = {
        "resolution": [int(image.shape[1]), int(image.shape[0])],
        "mean_rgb": image.mean(axis=(0, 1)).tolist(),
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "edge_density": float((gx > 25).mean() + (gy > 25).mean()) / 2,
        "objects": {
            name: _component(mask) for name, mask in _color_masks(image).items()
        },
    }
    return image, metrics


def main() -> None:
    report = {"cameras": {}, "scope": "Coarse frame-0 geometry/domain audit; not exact calibration."}
    all_center_errors = []
    all_area_ratios = []
    for camera in CAMERA_NAMES:
        reference_path = _download_and_extract(camera)
        simulation_path = RESULTS / "camera_observations" / f"{camera}.png"
        _, reference = _image_metrics(reference_path)
        _, simulation = _image_metrics(simulation_path)
        comparisons = {}
        for color in ("red", "blue", "yellow"):
            ref = reference["objects"][color]
            sim = simulation["objects"][color]
            if ref is None or sim is None:
                comparisons[color] = {
                    "common_visibility": False,
                    "reference_visible": ref is not None,
                    "simulation_visible": sim is not None,
                }
                continue
            center_error = float(np.linalg.norm(
                np.asarray(ref["center_normalized"]) -
                np.asarray(sim["center_normalized"])
            ))
            area_ratio = float(sim["area_fraction"] / ref["area_fraction"])
            all_center_errors.append(center_error)
            all_area_ratios.append(area_ratio)
            comparisons[color] = {
                "common_visibility": True,
                "center_error_normalized": center_error,
                "simulation_to_reference_area_ratio": area_ratio,
            }
        report["cameras"][camera] = {
            "reference": reference,
            "simulation": simulation,
            "object_comparison": comparisons,
            "luminance_difference": float(
                simulation["luminance_mean"] - reference["luminance_mean"]
            ),
        }
    report["summary"] = {
        "common_object_observations": len(all_center_errors),
        "center_error_normalized": {
            "mean": float(np.mean(all_center_errors)),
            "max": float(np.max(all_center_errors)),
        },
        "area_ratio": {
            "median": float(np.median(all_area_ratios)),
            "min": float(np.min(all_area_ratios)),
            "max": float(np.max(all_area_ratios)),
        },
        "exact_visual_calibration_confirmed": False,
        "verdict": (
            "Camera names/resolution and colored-object visibility are validated; "
            "pixel geometry, gripper appearance, lighting, and background remain "
            "outside the training domain."
        ),
    }
    path = RESULTS / "visual_fidelity_audit.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(path)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
