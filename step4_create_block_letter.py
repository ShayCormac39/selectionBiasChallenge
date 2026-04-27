"""
Step 4: Render a block letter (default "S") for the selection-bias meme overlay.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system bold fonts; fall back to PIL default."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a single block letter centered on a white canvas; letter is black.

    Parameters
    ----------
    height, width : int
        Output array shape (rows, cols).
    letter : str
        Character(s) to draw (default "S").
    font_size_ratio : float
        Initial font size as a fraction of min(height, width); may be reduced
        so the text fits inside the image.

    Returns
    -------
    np.ndarray
        Shape (height, width), float in [0, 1]; 0 = ink, 1 = background.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    letter = letter if letter else "S"

    # PIL images are (width, height)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    base = int(max(8, min(height, width) * font_size_ratio))
    font = _load_bold_font(base)

    # Shrink font until text fits with small margin
    margin = max(2, min(height, width) // 40)
    for _ in range(40):
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= width - 2 * margin and th <= height - 2 * margin:
            break
        new_size = max(8, int(getattr(font, "size", base) * 0.92))
        if new_size == getattr(font, "size", base):
            break
        font = _load_bold_font(new_size)

    bbox = draw.textbbox((0, 0), letter, font=font)
    left, top, right, bottom = bbox
    tw, th = right - left, bottom - top
    x = (width - tw) // 2 - left
    y = (height - th) // 2 - top
    draw.text((x, y), letter, fill=(0, 0, 0), font=font)

    gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    return np.clip(gray, 0.0, 1.0)
