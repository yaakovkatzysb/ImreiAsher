"""
Column detector for multi-column Hebrew text layouts.

Analyzes block bounding boxes from Google Vision to detect columns
and reorder text in the correct Hebrew reading order (right column first,
then left column, top to bottom within each column).
"""


class ColumnDetector:
    """Detects and handles multi-column layouts in OCR results."""

    def __init__(self, overlap_threshold: float = 0.4, min_column_gap_ratio: float = 0.05):
        """
        Args:
            overlap_threshold: Max horizontal overlap ratio between columns (0-1).
                Blocks overlapping more than this are considered same column.
            min_column_gap_ratio: Minimum gap between columns as ratio of page width.
        """
        self.overlap_threshold = overlap_threshold
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
            })

        # Detect columns
        columns = self._detect_columns(block_info)

        if len(columns) <= 1:
            # Single column - just sort top to bottom
            sorted_blocks = sorted(block_info, key=lambda b: b["y_min"])
            return "\n".join(b["text"] for b in sorted_blocks)

        # Multiple columns: sort columns right-to-left (Hebrew reading order)
        # Then sort blocks within each column top-to-bottom
        columns.sort(key=lambda col: -self._column_x_center(col))

        parts = []
        for column in columns:
            column.sort(key=lambda b: b["y_min"])
            column_text = "\n".join(b["text"] for b in column)
            parts.append(column_text)

        return "\n\n".join(parts)

    def _detect_columns(self, block_info: list[dict]) -> list[list[dict]]:
        """
        Cluster blocks into columns based on horizontal position.

        Uses a simple approach: sort blocks by x_center, then find gaps
        that indicate column boundaries.
        """
        if not block_info:
            return []

        # Find page width
        page_x_min = min(b["x_min"] for b in block_info)
        page_x_max = max(b["x_max"] for b in block_info)
        page_width = page_x_max - page_x_min

        if page_width <= 0:
            return [block_info]

        min_gap = page_width * self.min_column_gap_ratio

        # Sort blocks by x_center
        sorted_blocks = sorted(block_info, key=lambda b: b["x_center"])

        # Find natural column boundaries by looking at gaps in x_center values
        columns = [[sorted_blocks[0]]]

        for i in range(1, len(sorted_blocks)):
            current = sorted_blocks[i]
            prev = sorted_blocks[i - 1]

            # Check if there's a significant gap between this block and the previous
            # Use the gap between the right edge of left block and left edge of right block
            gap = current["x_min"] - prev["x_max"]

            # Also check if blocks overlap horizontally
            overlap = self._horizontal_overlap_ratio(prev, current)

            if gap > min_gap and overlap < self.overlap_threshold:
                # New column
                columns.append([current])
            else:
                # Same column as previous block
                columns[-1].append(current)

        # Filter out noise: columns with very little text compared to main columns
        if len(columns) > 1:
            max_blocks = max(len(col) for col in columns)
            columns = [col for col in columns if len(col) >= max(1, max_blocks * 0.15)]

        return columns

    def _horizontal_overlap_ratio(self, block_a: dict, block_b: dict) -> float:
        """Calculate horizontal overlap ratio between two blocks."""
        overlap_start = max(block_a["x_min"], block_b["x_min"])
        overlap_end = min(block_a["x_max"], block_b["x_max"])

        if overlap_start >= overlap_end:
            return 0.0

        overlap_width = overlap_end - overlap_start
        min_width = min(
            block_a["x_max"] - block_a["x_min"],
            block_b["x_max"] - block_b["x_min"],
        )

        if min_width <= 0:
            return 0.0

        return overlap_width / min_width

    def _column_x_center(self, column: list[dict]) -> float:
        """Get average x_center of a column."""
        return sum(b["x_center"] for b in column) / len(column)
