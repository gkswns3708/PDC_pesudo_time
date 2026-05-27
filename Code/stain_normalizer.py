"""
Stain Normalization for Histopathology Images.

Implements Reinhard and Macenko normalization methods based on:
- Khan et al. (2026) "Staining normalization in histopathology"
- Reference impl: github.com/wanghao14/Stain_Normalization

Usage:
    normalizer = ReinhardNormalizer()
    normalizer.fit(target_image)          # RGB uint8 image
    normalized = normalizer.transform(source_image)

    normalizer = MacenkoNormalizer()
    normalizer.fit(target_image)
    normalized = normalizer.transform(source_image)
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Tissue mask utilities
# ---------------------------------------------------------------------------

def get_tissue_mask(image, sat_threshold=15):
    """Create binary tissue mask by excluding white/near-white background.

    Uses HSV saturation channel: tissue has higher saturation than background.

    Args:
        image: RGB uint8 image (H, W, 3)
        sat_threshold: minimum saturation to consider as tissue (0-255)

    Returns:
        Boolean mask (H, W), True = tissue pixel
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    mask = saturation > sat_threshold
    # Also exclude very dark pixels (likely artifacts)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = mask & (gray > 15)
    return mask


# ---------------------------------------------------------------------------
# Color space conversions
# ---------------------------------------------------------------------------

def rgb_to_lab(image):
    """Convert RGB uint8 to L*a*b* float64."""
    # OpenCV expects BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return lab


def lab_to_rgb(lab):
    """Convert L*a*b* float64 to RGB uint8."""
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def rgb_to_od(image):
    """Convert RGB uint8 to Optical Density space.

    OD = -log10(I / I_0), where I_0 = 255.
    """
    image = image.astype(np.float64) + 1.0  # avoid log(0)
    od = -np.log10(image / 255.0)
    return od


def od_to_rgb(od):
    """Convert Optical Density back to RGB uint8."""
    rgb = 255.0 * np.power(10, -od)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Reinhard Normalizer
# ---------------------------------------------------------------------------

class ReinhardNormalizer:
    """Reinhard color normalization in L*a*b* color space.

    Transfers mean and std of each L*a*b* channel from target to source.
    Reference: Reinhard et al. (2001), used in Khan et al. (2026).
    """

    def __init__(self):
        self.target_mean = None
        self.target_std = None

    def fit(self, target_image, mask=None):
        """Compute target color statistics.

        Args:
            target_image: RGB uint8 image
            mask: optional boolean mask (True = tissue). If None, auto-generated.
        """
        if mask is None:
            mask = get_tissue_mask(target_image)

        lab = rgb_to_lab(target_image)
        tissue_pixels = lab[mask]  # (N, 3)

        if len(tissue_pixels) == 0:
            raise ValueError("No tissue pixels found in target image")

        self.target_mean = tissue_pixels.mean(axis=0)
        self.target_std = tissue_pixels.std(axis=0)
        # Avoid division by zero
        self.target_std[self.target_std < 1e-6] = 1.0

    def transform(self, source_image, mask=None):
        """Normalize source image to match target color statistics.

        Args:
            source_image: RGB uint8 image
            mask: optional boolean mask for source tissue pixels.

        Returns:
            Normalized RGB uint8 image.
        """
        if self.target_mean is None:
            raise RuntimeError("Call fit() before transform()")

        if mask is None:
            mask = get_tissue_mask(source_image)

        lab = rgb_to_lab(source_image)
        tissue_pixels = lab[mask]

        if len(tissue_pixels) == 0:
            return source_image.copy()

        src_mean = tissue_pixels.mean(axis=0)
        src_std = tissue_pixels.std(axis=0)
        src_std[src_std < 1e-6] = 1.0

        # Normalize: for each channel, (x - src_mean) / src_std * tgt_std + tgt_mean
        result = lab.copy()
        for c in range(3):
            result[:, :, c] = ((lab[:, :, c] - src_mean[c]) / src_std[c]) * self.target_std[c] + self.target_mean[c]

        return lab_to_rgb(result)

    def get_stats(self):
        """Return fitted target statistics."""
        return {"mean": self.target_mean.copy(), "std": self.target_std.copy()}


# ---------------------------------------------------------------------------
# Macenko Normalizer
# ---------------------------------------------------------------------------

class MacenkoNormalizer:
    """Macenko stain normalization via SVD in optical density space.

    Decomposes H&E stain vectors using SVD, then matches stain concentrations.
    Reference: Macenko et al. (2009), used in Khan et al. (2026).
    """

    def __init__(self, beta=0.15, alpha=1.0):
        """
        Args:
            beta: OD threshold for transparent pixels (default 0.15)
            alpha: percentile tolerance for stain vector angle (default 1.0)
        """
        self.beta = beta
        self.alpha = alpha
        self.stain_matrix_target = None
        self.maxC_target = None

    def _get_stain_matrix(self, image, mask=None):
        """Extract 2x3 stain matrix (H, E vectors) from image using SVD.

        Returns:
            stain_matrix: (2, 3) array, rows are H and E stain vectors
            C: (N, 2) stain concentrations
        """
        od = rgb_to_od(image)
        od_flat = od.reshape(-1, 3)

        # Use tissue mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            od_flat = od_flat[mask_flat]

        # Remove transparent pixels (low OD)
        od_thresh = od_flat[np.all(od_flat > self.beta, axis=1)]

        if len(od_thresh) < 10:
            # Fallback: use all non-zero OD pixels
            od_thresh = od_flat[np.any(od_flat > 0.01, axis=1)]
            if len(od_thresh) < 10:
                raise ValueError("Not enough tissue pixels for stain decomposition")

        # SVD on thresholded OD values
        _, _, Vt = np.linalg.svd(od_thresh, full_matrices=False)

        # First two principal directions
        V = Vt[:2, :]  # (2, 3)

        # Make sure vectors point in positive direction
        if V[0, 0] < 0:
            V[0] *= -1
        if V[1, 0] < 0:
            V[1] *= -1

        # Project OD values onto the plane defined by the top 2 SVD directions
        projected = od_thresh @ V.T  # (N, 2)

        # Convert to angles
        angles = np.arctan2(projected[:, 1], projected[:, 0])

        # Find robust extremes (percentiles)
        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100.0 - self.alpha)

        # Stain vectors correspond to the extreme angles
        vec1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ V
        vec2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ V

        # Hematoxylin should be more blue (higher OD in blue channel = index 2)
        if vec1[0] > vec2[0]:
            stain_matrix = np.array([vec1, vec2])
        else:
            stain_matrix = np.array([vec2, vec1])

        # Normalize stain vectors
        stain_matrix /= np.linalg.norm(stain_matrix, axis=1, keepdims=True)

        return stain_matrix

    def _get_concentrations(self, image, stain_matrix, mask=None):
        """Get stain concentrations by solving OD = C @ stain_matrix.

        Returns:
            C: (H*W, 2) concentration matrix
        """
        od = rgb_to_od(image).reshape(-1, 3)
        # Solve least squares: C = OD @ pinv(stain_matrix)
        C = od @ np.linalg.pinv(stain_matrix)
        return C

    def fit(self, target_image, mask=None):
        """Compute target stain matrix and max concentrations.

        Args:
            target_image: RGB uint8 image
            mask: optional boolean tissue mask
        """
        if mask is None:
            mask = get_tissue_mask(target_image)

        self.stain_matrix_target = self._get_stain_matrix(target_image, mask)
        C = self._get_concentrations(target_image, self.stain_matrix_target, mask)

        # Use 99th percentile as max concentration (robust to outliers)
        self.maxC_target = np.percentile(C, 99, axis=0)

    def transform(self, source_image, mask=None):
        """Normalize source image staining to match target.

        Args:
            source_image: RGB uint8 image
            mask: optional boolean tissue mask for source

        Returns:
            Normalized RGB uint8 image.
        """
        if self.stain_matrix_target is None:
            raise RuntimeError("Call fit() before transform()")

        if mask is None:
            mask = get_tissue_mask(source_image)

        h, w, _ = source_image.shape

        try:
            stain_matrix_source = self._get_stain_matrix(source_image, mask)
        except ValueError:
            return source_image.copy()

        C_source = self._get_concentrations(source_image, stain_matrix_source, mask)

        # Max concentrations of source (99th percentile)
        maxC_source = np.percentile(C_source, 99, axis=0)
        maxC_source[maxC_source < 1e-6] = 1.0

        # Scale concentrations
        C_normalized = C_source * (self.maxC_target / maxC_source)

        # Reconstruct OD
        od_normalized = C_normalized @ self.stain_matrix_target

        # Convert back to RGB
        rgb_normalized = od_to_rgb(od_normalized.reshape(h, w, 3))

        return rgb_normalized

    def get_stain_matrix(self):
        """Return fitted target stain matrix."""
        return self.stain_matrix_target.copy()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def normalize_reinhard(source, target, source_mask=None, target_mask=None):
    """One-shot Reinhard normalization."""
    norm = ReinhardNormalizer()
    norm.fit(target, mask=target_mask)
    return norm.transform(source, mask=source_mask)


def normalize_macenko(source, target, source_mask=None, target_mask=None):
    """One-shot Macenko normalization."""
    norm = MacenkoNormalizer()
    norm.fit(target, mask=target_mask)
    return norm.transform(source, mask=source_mask)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path

    consep_dir = Path("/app/app/CoNSeP/Train/Images")
    images = sorted(consep_dir.glob("*.png"))

    if len(images) >= 2:
        target = cv2.cvtColor(cv2.imread(str(images[0])), cv2.COLOR_BGR2RGB)
        source = cv2.cvtColor(cv2.imread(str(images[1])), cv2.COLOR_BGR2RGB)

        print(f"Target: {images[0].name}, shape={target.shape}")
        print(f"Source: {images[1].name}, shape={source.shape}")

        # Test Reinhard
        reinhard = ReinhardNormalizer()
        reinhard.fit(target)
        result_r = reinhard.transform(source)
        print(f"Reinhard result shape: {result_r.shape}, dtype: {result_r.dtype}")

        # Test Macenko
        macenko = MacenkoNormalizer()
        macenko.fit(target)
        result_m = macenko.transform(source)
        print(f"Macenko result shape: {result_m.shape}, dtype: {result_m.dtype}")

        # Save results
        os.makedirs("/app/results/stain_norm_test", exist_ok=True)
        cv2.imwrite("/app/results/stain_norm_test/target.png",
                     cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/app/results/stain_norm_test/source.png",
                     cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/app/results/stain_norm_test/reinhard.png",
                     cv2.cvtColor(result_r, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/app/results/stain_norm_test/macenko.png",
                     cv2.cvtColor(result_m, cv2.COLOR_RGB2BGR))
        print("Test images saved to /app/results/stain_norm_test/")
    else:
        print("Not enough CoNSeP images for testing")
