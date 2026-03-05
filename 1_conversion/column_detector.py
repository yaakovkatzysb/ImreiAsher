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

        # Calculate typical column line word count for comparison
        page_center = (page_x_min + page_x_max) / 2

        # Split lines into right column, left column, or spanning (headers)
        right_col_lines = []
        left_col_lines = []
        spanning_lines = []  # Lines that span both columns (e.g. headers)

        for line in lines:
            right_words = [w for w in line if w["x_center"] > boundary]
            left_words = [w for w in line if w["x_center"] <= boundary]

            # Check if this line spans both columns (e.g. a header)
            if right_words and left_words:
                if self._is_spanning_line(line, boundary):
                    line_sorted = sorted(line, key=lambda w: -w["x_center"])  # RTL
                    y = sum(w["y_center"] for w in line) / len(line)
                    spanning_lines.append((y, line_sorted))
                    continue

            # Check if this is a short centered line (header fragment)
            if self._is_centered_line(line, boundary, page_center, page_width):
                line_sorted = sorted(line, key=lambda w: -w["x_center"])  # RTL
                y = sum(w["y_center"] for w in line) / len(line)
                spanning_lines.append((y, line_sorted))
                continue

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
        spanning_lines.sort(key=lambda x: x[0])

        logger.info(
            f"    Columns detected: 2 "
            f"(right: {len(right_col_lines)} lines, left: {len(left_col_lines)} lines, "
            f"spanning: {len(spanning_lines)} lines)"
        )

        # Build text: spanning lines at top, then right column, then left
        parts = []

        # Add spanning lines that appear before the column content
        if spanning_lines and right_col_lines:
            first_col_y = min(right_col_lines[0][0],
                              left_col_lines[0][0] if left_col_lines else float("inf"))
            top_spanning = [s for s in spanning_lines if s[0] < first_col_y]
            mid_spanning = [s for s in spanning_lines if s[0] >= first_col_y]
        else:
            top_spanning = spanning_lines
            mid_spanning = []

        if top_spanning:
            top_text = "\n".join(
                " ".join(w["text"] for w in line) for _, line in top_spanning
            )
            parts.append(top_text)

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

        # Add spanning lines that appear within/after column content
        if mid_spanning:
            mid_text = "\n".join(
                " ".join(w["text"] for w in line) for _, line in mid_spanning
            )
            parts.append(mid_text)

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

    def _is_centered_line(
        self, line: list[dict], boundary: float, page_center: float, page_width: float
    ) -> bool:
        """
        Check if a line is a short centered line (like a header fragment).

        Short lines (few words) whose center is near the page center are likely
        headers, not column content. Column content lines are longer and aligned
        to one side.
        """
        if not line:
            return False

        # Line extent
        line_x_min = min(w["x_min"] for w in line)
        line_x_max = max(w["x_max"] for w in line)
        line_width = line_x_max - line_x_min
        line_center = (line_x_min + line_x_max) / 2

        # A centered line:
        # 1. Is short relative to page width (covers less than ~40% of page)
        # 2. Its center is near the page center / boundary
        width_ratio = line_width / page_width if page_width > 0 else 1
        center_offset = abs(line_center - page_center) / page_width if page_width > 0 else 1

        # Short line near center -> header
        if width_ratio < 0.4 and center_offset < 0.15:
            logger.info(
                f"    Centered header line detected: '{' '.join(w['text'] for w in line)}' "
                f"(width={width_ratio:.0%}, offset={center_offset:.0%})"
            )
            return True

        return False

    def _is_spanning_line(self, line: list[dict], boundary: float) -> bool:
        """
        Check if a line spans across the column boundary continuously,
        indicating it's a header or full-width line rather than two column segments.

        Looks at word gaps: if there's no unusually large gap near the boundary,
        the line is spanning (not split into columns).
        """
        if len(line) <= 1:
            return False

        # Sort words by x position (left to right)
        sorted_words = sorted(line, key=lambda w: w["x_min"])

        # Calculate gaps between consecutive words
        gaps = []
        for i in range(len(sorted_words) - 1):
            gap = sorted_words[i + 1]["x_min"] - sorted_words[i]["x_max"]
            gaps.append((gap, sorted_words[i]["x_max"], sorted_words[i + 1]["x_min"]))

        if not gaps:
            return False

        # Find the gap closest to the boundary
        boundary_gap = None
        other_gaps = []
        for gap_size, left_edge, right_edge in gaps:
            if left_edge <= boundary <= right_edge or abs((left_edge + right_edge) / 2 - boundary) < abs(gap_size) + 50:
                boundary_gap = gap_size
            else:
                other_gaps.append(gap_size)

        if boundary_gap is None:
            return True  # No gap near boundary -> spanning

        # If the gap at the boundary is not much larger than typical word gaps,
        # this is a spanning line
        avg_gap = sum(g for g in other_gaps) / len(other_gaps) if other_gaps else 0
        max_other_gap = max(other_gaps) if other_gaps else 0

        # Spanning if boundary gap is less than 3x the average gap or similar to max gap
        return boundary_gap < max(avg_gap * 3, max_other_gap * 1.5, 80)

    def _lines_to_text(self, lines: list[list[dict]]) -> str:
        """Convert lines of words to text (single column)."""
        return "\n".join(" ".join(w["text"] for w in line) for line in lines)
