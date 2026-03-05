"""
Column detector for multi-column Hebrew text layouts.

Works at the word level: groups words into text lines, detects column
boundaries via a histogram of x-positions, then splits lines by column
and outputs text in Hebrew reading order (right column first, then left).
"""

import logging

logger = logging.getLogger(__name__)


class ColumnDetector:
    """Detects and handles multi-column layouts using word-level positioning."""

    def reorder_by_columns(self, words: list[dict]) -> str:
        """
        Detect columns from word positions and return text in correct reading order.

        Args:
            words: List of dicts with 'text' and 'bbox' keys.
                   bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].

        Returns:
            Text reordered by columns (right-to-left for Hebrew).
        """
        if not words:
            return ""

        # Extract positions
        word_info = []
        for w in words:
            bbox = w["bbox"]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            text = w["text"].strip()
            if not text:
                continue
            word_info.append({
                "text": text,
                "x_min": min(xs),
                "x_max": max(xs),
                "x_center": (min(xs) + max(xs)) / 2,
                "y_min": min(ys),
                "y_max": max(ys),
                "y_center": (min(ys) + max(ys)) / 2,
                "height": max(ys) - min(ys),
            })

        if not word_info:
            return ""

        # Page dimensions
        page_x_min = min(w["x_min"] for w in word_info)
        page_x_max = max(w["x_max"] for w in word_info)
        page_width = page_x_max - page_x_min

        if page_width <= 0:
            return " ".join(w["text"] for w in word_info)

        # Group words into text lines
        lines = self._group_into_lines(word_info)

        logger.info(f"    Column detection: {len(word_info)} words, {len(lines)} lines")

        # Detect column boundary using histogram
        boundary = self._find_column_boundary(word_info, page_width, page_x_min)

        if boundary is None:
            logger.info("    Columns detected: 1")
            return self._lines_to_text(lines)

        logger.info(f"    Column boundary at x={boundary:.0f}")

        # Split lines into right and left column segments
        right_col_lines = []
        left_col_lines = []

        for line in lines:
            right_words = [w for w in line if w["x_center"] > boundary]
            left_words = [w for w in line if w["x_center"] <= boundary]

            if right_words:
                right_words.sort(key=lambda w: -w["x_center"])  # RTL
                y = sum(w["y_center"] for w in right_words) / len(right_words)
                right_col_lines.append((y, right_words))

            if left_words:
                left_words.sort(key=lambda w: -w["x_center"])  # RTL
                y = sum(w["y_center"] for w in left_words) / len(left_words)
                left_col_lines.append((y, left_words))

        # Sort by y within each column
        right_col_lines.sort(key=lambda x: x[0])
        left_col_lines.sort(key=lambda x: x[0])

        logger.info(
            f"    Columns detected: 2 "
            f"(right: {len(right_col_lines)} lines, left: {len(left_col_lines)} lines)"
        )

        # Build text: right column first (Hebrew reading order), then left
        parts = []

        if right_col_lines:
            right_text = "\n".join(
                " ".join(w["text"] for w in line) for _, line in right_col_lines
            )
            parts.append(right_text)

        if left_col_lines:
            left_text = "\n".join(
                " ".join(w["text"] for w in line) for _, line in left_col_lines
            )
            parts.append(left_text)

        return "\n\n".join(parts)

    def _group_into_lines(self, word_info: list[dict]) -> list[list[dict]]:
        """Group words into text lines based on y-proximity."""
        # Estimate line height
        heights = [w["height"] for w in word_info if w["height"] > 5]
        avg_height = sum(heights) / len(heights) if heights else 30
        threshold = avg_height * 0.5

        # Sort by y_center
        sorted_words = sorted(word_info, key=lambda w: w["y_center"])

        lines = []
        current_line = [sorted_words[0]]
        line_y = sorted_words[0]["y_center"]

        for w in sorted_words[1:]:
            if abs(w["y_center"] - line_y) <= threshold:
                current_line.append(w)
                line_y = sum(ww["y_center"] for ww in current_line) / len(current_line)
            else:
                current_line.sort(key=lambda w: -w["x_center"])  # RTL
                lines.append(current_line)
                current_line = [w]
                line_y = w["y_center"]

        current_line.sort(key=lambda w: -w["x_center"])
        lines.append(current_line)

        return lines

    def _find_column_boundary(
        self, word_info: list[dict], page_width: float, page_x_min: float
    ) -> float | None:
        """
        Find column boundary using a histogram of word x-positions.

        Divides the page into bins and looks for an empty/sparse bin
        in the middle third - this indicates the column gutter.
        """
        bin_width = max(30, page_width / 40)
        num_bins = int(page_width / bin_width) + 1
        bins = [0] * num_bins

        for w in word_info:
            bin_idx = min(int((w["x_center"] - page_x_min) / bin_width), num_bins - 1)
            bins[bin_idx] += 1

        # Look for the emptiest bin in the middle third
        third_start = num_bins // 3
        third_end = 2 * num_bins // 3

        if third_start >= third_end:
            return None

        min_count = float("inf")
        min_idx = -1
        for i in range(third_start, third_end):
            if bins[i] < min_count:
                min_count = bins[i]
                min_idx = i

        # The gutter bin should have very few words compared to average
        total_words = sum(bins)
        avg_per_bin = total_words / num_bins if num_bins > 0 else 0

        logger.info(
            f"    Histogram: gutter bin {min_idx} has {min_count} words "
            f"(avg {avg_per_bin:.1f}/bin, threshold {max(1, avg_per_bin * 0.15):.1f})"
        )

        if min_count <= max(1, avg_per_bin * 0.15):
            boundary_x = page_x_min + (min_idx + 0.5) * bin_width
            return boundary_x

        return None

    def _lines_to_text(self, lines: list[list[dict]]) -> str:
        """Convert lines of words to text (single column)."""
        return "\n".join(" ".join(w["text"] for w in line) for line in lines)
