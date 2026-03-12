"""
Layout detection using Surya for accurate column boundary detection.

Surya uses a trained AI model to visually identify page layout regions
(columns, headers, text blocks) from the page image itself, which is
far more robust than histogram-based heuristics on OCR word positions.

Usage:
    detector = SuryaLayoutDetector()          # loads model once
    boundary = detector.find_boundary(image_bytes)  # per page
    # boundary is the x-coordinate separating left/right columns,
    # or None if the page is single-column.

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

    def find_boundary(self, image_bytes: bytes) -> Optional[float]:
        """
        Detect column boundary from a page image.

        Args:
            image_bytes: Raw image bytes (PNG/JPEG) of the page.

        Returns:
            The x-coordinate of the column boundary (gutter center),
            or None if the page is single-column.
        """
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        results = self._predictor([image])

        if not results:
            return None

        page = results[0]

        # Find all "Column" regions detected by Surya
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

        # Sort columns by x-position (left to right)
        columns.sort(key=lambda b: b.bbox[0])

        # For a two-column layout, find the gap between the two main columns.
        # Take the two largest (by area) column regions.
        by_area = sorted(
            columns,
            key=lambda b: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]),
            reverse=True,
        )
        main_cols = sorted(by_area[:2], key=lambda b: b.bbox[0])

        # Boundary = midpoint between right edge of left column and
        # left edge of right column
        left_col_right = main_cols[0].bbox[2]   # x_max of left column
        right_col_left = main_cols[1].bbox[0]   # x_min of right column

        if right_col_left <= left_col_right:
            # Columns overlap — can't determine boundary
            logger.debug(
                f"  surya | עמודות חופפות "
                f"(שמאל_ימין={left_col_right:.0f}, ימין_שמאל={right_col_left:.0f})"
            )
            return None

        boundary = (left_col_right + right_col_left) / 2

        logger.debug(
            f"  surya | גבול ב-x={boundary:.0f} "
            f"(שמאל עד {left_col_right:.0f}, ימין מ-{right_col_left:.0f})"
        )

        return boundary
