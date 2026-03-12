"""
Layout detection using Surya for accurate column detection.

Surya uses a trained AI model to visually identify page layout regions
(columns, headers, text blocks) from the page image itself, which is
far more robust than histogram-based heuristics on OCR word positions.

Usage:
    detector = SuryaLayoutDetector()          # loads model once
    columns = detector.find_columns(image_bytes)  # per page
    # columns is a list of [x_min, y_min, x_max, y_max] bounding boxes,
    # sorted right-to-left (Hebrew reading order), or None if single-column.

Install:
    pip install surya-ocr Pillow
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports — only fail when actually used, not at import time.
# This lets the rest of the pipeline work even without surya installed.
_surya_available: Optional[bool] = None


def _check_surya() -> bool:
    global _surya_available
    if _surya_available is None:
        try:
            from surya.foundation import FoundationPredictor  # noqa: F401
            from surya.layout import LayoutPredictor  # noqa: F401
            _surya_available = True
        except ImportError:
            _surya_available = False
    return _surya_available


class SuryaLayoutDetector:
    """Detects page layout (columns) using Surya's trained model."""

    def __init__(self):
        if not _check_surya():
            raise ImportError(
                "surya-ocr is not installed. Install with: pip install surya-ocr"
            )
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor

        logger.info("טוען מודל Surya layout...")
        self._foundation = FoundationPredictor()
        self._predictor = LayoutPredictor(self._foundation)
        logger.info("מודל Surya layout נטען בהצלחה")

    def find_columns(self, image_bytes: bytes) -> Optional[list[list[float]]]:
        """
        Detect column regions from a page image.

        Args:
            image_bytes: Raw image bytes (PNG/JPEG) of the page.

        Returns:
            List of column bounding boxes [[x_min, y_min, x_max, y_max], ...],
            sorted right-to-left (Hebrew reading order).
            Returns None if the page is single-column.
        """
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        results = self._predictor([image])

        if not results:
            return None

        page = results[0]

        # Find all "Column" / "Text" regions detected by Surya
        columns = [
            block for block in page.bboxes
            if block.label.lower() in ("column", "text")
            and block.confidence > 0.3
        ]

        logger.debug(
            f"  surya | זוהו {len(columns)} אזורים: "
            + ", ".join(
                f"{b.label}(x={b.bbox[0]:.0f}-{b.bbox[2]:.0f}, conf={b.confidence:.2f})"
                for b in columns
            )
        )

        if len(columns) < 2:
            logger.debug("  surya | עמודה אחת (פחות מ-2 אזורים)")
            return None

        # Take the two largest (by area) column regions
        by_area = sorted(
            columns,
            key=lambda b: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]),
            reverse=True,
        )
        main_cols = by_area[:2]

        # Sort right-to-left (Hebrew reading order: rightmost column first)
        main_cols.sort(key=lambda b: -b.bbox[0])

        result = [list(b.bbox) for b in main_cols]

        logger.debug(
            f"  surya | עמודות: "
            + ", ".join(
                f"[{r[0]:.0f},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f}]"
                for r in result
            )
        )

        return result
