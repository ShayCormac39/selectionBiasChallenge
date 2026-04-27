"""
Assemble the four-panel statistics meme (selection bias visualization).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _resize_gray(img: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Resize 2D float [0,1] image to (height, width)."""
    h, w = shape_hw
    if img.ndim != 2:
        raise ValueError("Expected 2D grayscale array")
    if img.shape[:2] == (h, w):
        return np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="L")
    pil = pil.resize((w, h), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float64) / 255.0


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Build a 1×4 figure: Reality | Your Model | Selection Bias | Estimate,
    then save as PNG.

    Panel images are resized to match ``original_img`` if shapes differ.
    """
    ref_hw = original_img.shape[:2]
    panels = [
        _resize_gray(original_img, ref_hw),
        _resize_gray(stipple_img, ref_hw),
        _resize_gray(block_letter_img, ref_hw),
        _resize_gray(masked_stipple_img, ref_hw),
    ]
    titles = ["Reality", "Your Model", "Selection Bias", "Estimate"]

    n = len(panels)
    fig_w = 3.8 * n
    fig_h = 4.2
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_w, fig_h),
        facecolor=background_color,
    )
    if n == 1:
        axes = np.array([axes])

    for ax, data, title in zip(axes, panels, titles):
        ax.imshow(data, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#b0b0b0")
            spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.06, wspace=0.12)
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=background_color,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.25,
        format="png",
    )
    plt.close(fig)
