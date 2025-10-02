import warnings
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute SSIM (Structural Similarity Index) between two images.

    Args:
        image1: First image as numpy array
        image2: Second image as numpy array

    Returns:
        SSIM value between 0 and 1
    """
    if image1 is None or image2 is None:
        return 0.0

    if image1.shape != image2.shape:
        warnings.warn(f"Image shapes don't match: {image1.shape} vs {image2.shape}, defaulting SSIM to 0 (assuming screen switch)")
        return 0.0

    if len(image1.shape) == 3:
        # Convert RGB to grayscale
        gray1 = np.dot(image1[..., :3], [0.2989, 0.5870, 0.1140])
        gray2 = np.dot(image2[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray1 = image1
        gray2 = image2

    # Compute SSIM
    try:
        ssim_value = ssim(gray1, gray2, data_range=gray1.max() - gray1.min())
        return float(ssim_value)
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        return 0.0
