"""
Column detector for multi-column Hebrew text layouts.

Analyzes paragraph bounding boxes from Google Vision to detect columns
and reorder text in the correct Hebrew reading order (right column first,
then left column, top to bottom within each column).
"""

import logging

logger = logging.getLogger(__name__)


class ColumnDetector:
    """Detects and handles multi-column layouts in OCR results."""

    def __init__(self, min_column_gap_ratio: float = 0.15):
        """
        Args:
            min_column_gap_ratio: Minimum gap between column centers as ratio of page width.
                If the largest gap in x_center values exceeds this, we split into columns.
        """
        self.min_column_gap_ratio = min_column_gap_ratio

    def reorder_blocks_by_columns(self, blocks: list[dict]) -> str:
        """
        Detect columns and return text in correct reading order.

        Args:
            blocks: List of dicts with 'text' and 'bbox' keys.
                    bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].

        Returns:
            Text reordered by columns (right-to-left for Hebrew).
        """
        if not blocks:
            return ""

        if len(blocks) <= 1:
            return blocks[0]["text"] if blocks else ""

        # Calculate x-ranges for each block
        block_info = []
        for block in blocks:
            bbox = block["bbox"]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            block_info.append({
                "text": block["text"],
                "x_min": min(xs),
                "x_max": max(xs),
                "x_center": sum(xs) / len(xs),
                "y_min": min(ys),
                "y_max": max(ys),
                "width": max(xs) - min(xs),
            })

        # Find page dimensions
        page_x_min = min(b["x_min"] for b in block_info)
        page_x_max = max(b["x_max"] for b in block_info)
        page_width = page_x_max - page_x_min

        if page_width <= 0:
            sorted_blocks = sorted(block_info, key=lambda b: b["y_min"])
            return "\n".join(b["text"] for b in sorted_blocks)

        # Filter out full-width blocks (headers, footers) — width > 70% of page
        full_width_threshold = page_width * 0.70
        narrow_blocks = [b for b in block_info if b["width"] < full_width_threshold]
        wide_blocks = [b for b in block_info if b["width"] >= full_width_threshold]

        # Debug logging
        logger.info(f"    Column detection: {len(block_info)} paragraphs "
                     f"({len(narrow_blocks)} narrow, {len(wide_blocks)} wide)")

        # Detect columns from narrow blocks only
        columns = self._detect_columns(narrow_blocks, page_width)

        logger.info(f"    Columns detected: {len(columns)}")
        for i, col in enumerate(columns):
            x_centers = [b["x_center"] for b in col]
            logger.info(
                f"      Column {i}: {len(col)} paragraphs, "
                f"avg_x_center={sum(x_centers)/len(x_centers):.0f}"
            )

        if len(columns) <= 1:
            # Single column - just sort top to bottom
            sorted_blocks = sorted(block_info, key=lambda b: b["y_min"])
            return "\n".join(b["text"] for b in sorted_blocks)

        # Multiple columns: sort columns right-to-left (Hebrew reading order)
        columns.sort(key=lambda col: -self._column_x_center(col))

        # Place wide blocks (headers/footers) before the columns
        parts = []
        if wide_blocks:
            wide_blocks.sort(key=lambda b: b["y_min"])
            # Wide blocks that come before all columns go first
            min_col_y = min(b["y_min"] for col in columns for b in col)
            header_blocks = [b for b in wide_blocks if b["y_min"] < min_col_y]
            footer_blocks = [b for b in wide_blocks if b["y_min"] >= min_col_y]

            if header_blocks:
                parts.append("\n".join(b["text"] for b in header_blocks))

            for column in columns:
                column.sort(key=lambda b: b["y_min"])
                parts.append("\n".join(b["text"] for b in column))

            if footer_blocks:
                parts.append("\n".join(b["text"] for b in footer_blocks))
        else:
            for column in columns:
                column.sort(key=lambda b: b["y_min"])
                parts.append("\n".join(b["text"] for b in column))

        return "\n\n".join(parts)

    def _detect_columns(self, block_info: list[dict], page_width: float) -> list[list[dict]]:
        """
        Cluster blocks into columns by finding the largest gap in x_center values.
        """
        if not block_info:
            return []

        if len(block_info) == 1:
            return [block_info]

        min_gap = page_width * self.min_column_gap_ratio

        # Sort by x_center
        sorted_blocks = sorted(block_info, key=lambda b: b["x_center"])

        # Find the largest gap between consecutive x_centers
        best_gap = 0
        best_gap_idx = -1
        for i in range(1, len(sorted_blocks)):
            gap = sorted_blocks[i]["x_center"] - sorted_blocks[i - 1]["x_center"]
            if gap > best_gap:
                best_gap = gap
                best_gap_idx = i

        logger.info(f"    Largest x_center gap: {best_gap:.0f} "
                     f"(threshold: {min_gap:.0f}, page_width: {page_width:.0f})")

        if best_gap >= min_gap and best_gap_idx > 0:
            # Split into two columns at the largest gap
            left_group = sorted_blocks[:best_gap_idx]
            right_group = sorted_blocks[best_gap_idx:]

            # Verify both groups have enough content (at least 2 paragraphs each)
            if len(left_group) >= 2 and len(right_group) >= 2:
                return [left_group, right_group]

        return [block_info]

    def _column_x_center(self, column: list[dict]) -> float:
        """Get average x_center of a column."""
        return sum(b["x_center"] for b in column) / len(column)
