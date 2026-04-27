"""
Step 5: Apply block-letter mask to stippled image to show biased estimate.
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply mask: dark mask pixels (below threshold) become white (stipples removed);
    light mask pixels keep the stippled image unchanged.

    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image (height, width), values in [0, 1].
    mask_img : np.ndarray
        Mask; same shape. 0 = letter (remove data), 1 = keep.
    threshold : float
        Pixels with mask value strictly below this count as mask region.

    Returns
    -------
    np.ndarray
        Same shape as inputs; biased stipple image with "S" region cleared to white.
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError("stipple_img and mask_img must have the same shape")
    out = stipple_img.copy()
    remove = mask_img < threshold
    out[remove] = 1.0
    return out
