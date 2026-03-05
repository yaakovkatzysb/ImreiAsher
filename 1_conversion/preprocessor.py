"""
Image pre-processing for improving OCR quality on poor/medium quality scans.
Uses OpenCV for image enhancement before sending to Google Vision.
"""

import io

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Pre-process images to improve OCR accuracy."""

    def __init__(self, config: dict):
        self.denoise = config.get("denoise", True)
        self.binarize = config.get("binarize", True)
        self.deskew = config.get("deskew", True)
        self.enhance_contrast = config.get("enhance_contrast", True)
        self.upscale_factor = config.get("upscale_factor", 2)

    def process(self, image_bytes: bytes, quality_level: str = "poor") -> bytes:
        """
        Apply pre-processing based on quality level.

        Args:
            image_bytes: Raw image bytes
            quality_level: 'good', 'medium', or 'poor'

        Returns:
            Processed image bytes (PNG format)
        """
        img = self._bytes_to_cv2(image_bytes)

        if quality_level == "good":
            return image_bytes  # no processing needed

        if quality_level == "medium":
            img = self._light_processing(img)
        else:  # poor
            img = self._full_processing(img)

        return self._cv2_to_bytes(img)

    def _light_processing(self, img: np.ndarray) -> np.ndarray:
        """Light processing for medium quality images."""
        if self.enhance_contrast:
            img = self._apply_clahe(img)
        if self.denoise:
            img = self._apply_denoise(img, strength=5)
        return img

    def _full_processing(self, img: np.ndarray) -> np.ndarray:
        """Full processing for poor quality images."""
        # Upscale first if needed
        if self.upscale_factor > 1:
            img = self._upscale(img)

        if self.enhance_contrast:
            img = self._apply_clahe(img)

        if self.denoise:
            img = self._apply_denoise(img, strength=10)

        if self.deskew:
            img = self._apply_deskew(img)

        if self.binarize:
            img = self._apply_binarize(img)

        return img

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _apply_denoise(self, img: np.ndarray, strength: int = 10) -> np.ndarray:
        """Remove noise from image."""
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, strength, strength)
        return cv2.fastNlMeansDenoising(img, None, strength)

    def _apply_deskew(self, img: np.ndarray) -> np.ndarray:
        """Fix skewed/rotated text."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Detect skew angle using Hough transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return img

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:  # only consider near-horizontal lines
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:  # skip if almost straight
            return img

        # Rotate image
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _apply_binarize(self, img: np.ndarray) -> np.ndarray:
        """Convert to black and white using adaptive threshold."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """Increase image resolution."""
        h, w = img.shape[:2]
        return cv2.resize(img, (w * self.upscale_factor, h * self.upscale_factor),
                          interpolation=cv2.INTER_CUBIC)

    def _bytes_to_cv2(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _cv2_to_bytes(self, img: np.ndarray) -> bytes:
        """Convert OpenCV image to PNG bytes."""
        success, encoded = cv2.imencode(".png", img)
        if not success:
            raise RuntimeError("Failed to encode image to PNG")
        return encoded.tobytes()

    def assess_quality(self, image_bytes: bytes) -> tuple[str, float]:
        """
        Assess image quality to decide processing level.

        Returns:
            (quality_level, score) - e.g. ('good', 0.95)
        """
        img = self._bytes_to_cv2(image_bytes)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        scores = []

        # 1. Resolution score
        h, w = gray.shape
        resolution_score = min(1.0, (h * w) / (2000 * 3000))
        scores.append(resolution_score)

        # 2. Contrast score (std deviation of pixel values)
        contrast = gray.std() / 128.0  # normalize to 0-1 range
        contrast_score = min(1.0, contrast)
        scores.append(contrast_score)

        # 3. Sharpness score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(1.0, sharpness / 500.0)
        scores.append(sharpness_score)

        # 4. Noise level (lower is better)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.abs(gray.astype(float) - blurred.astype(float)).mean()
        noise_score = max(0.0, 1.0 - noise / 30.0)
        scores.append(noise_score)

        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.75:
            return ("good", round(avg_score, 3))
        elif avg_score >= 0.50:
            return ("medium", round(avg_score, 3))
        else:
            return ("poor", round(avg_score, 3))
