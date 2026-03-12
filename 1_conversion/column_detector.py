"""
Column detector for multi-column Hebrew text layouts.

Works at the word level: groups words into text lines, detects column
boundaries via a histogram of x-positions, then splits lines by column
and outputs text in Hebrew reading order (right column first, then left).
"""

import re
import logging

logger = logging.getLogger(__name__)

# Punctuation tokens that should not have a space before them
_PUNCT_NO_SPACE_BEFORE = re.compile(r"^[,\.;:!?\)\]\}\"״׳']+$")
# Opening brackets/parens - should not have a space after them
_PUNCT_NO_SPACE_AFTER = re.compile(r"^[\(\[\{]+$")


def _join_words(words) -> str:
    """Join word texts, suppressing spaces around punctuation tokens."""
    parts = []
    skip_space = False
    for w in words:
        token = w["text"] if isinstance(w, dict) else w
        if parts and _PUNCT_NO_SPACE_BEFORE.match(token):
            parts.append(token)
        elif skip_space:
            parts.append(token)
        else:
            if parts:
                parts.append(" ")
            parts.append(token)
        skip_space = bool(_PUNCT_NO_SPACE_AFTER.match(token))
    return "".join(parts)


class ColumnDetector:
    """Detects and handles multi-column layouts using word-level positioning."""

    # Header detection: words taller than avg_height * this factor are "large"
    HEADER_HEIGHT_FACTOR = 1.3

    def reorder_by_columns(self, words: list[dict], external_boundary: float | None = None) -> str:
        """
        Detect columns from word positions and return text in correct reading order.

        Args:
            words: List of dicts with 'text' and 'bbox' keys.
                   bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
            external_boundary: Optional x-coordinate for column boundary,
                   provided by an external layout detector (e.g. Surya).
                   When given, skips histogram-based boundary detection.

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
            return _join_words(word_info)

        # Group words into text lines
        lines = self._group_into_lines(word_info)

        logger.debug(f"  עמודות | קלט: {len(word_info)} מילים → {len(lines)} שורות")

        # Detect and extract header lines by font size (before column split)
        avg_height = self._calc_avg_height(word_info)
        header_lines, body_lines = self._split_headers_by_font_size(lines, avg_height)

        if header_lines:
            header_texts = [h["text"] for h in header_lines]
            logger.debug(f"  עמודות | כותרות ({len(header_lines)}): {header_texts}")

        # Continue column detection on body lines only
        lines = body_lines

        # Detect column boundary using histogram (from body words only)
        body_words = [w for line in body_lines for w in line]
        if not body_words:
            # All lines are headers
            return "\n".join(h["text"] for h in header_lines)

        if external_boundary is not None:
            boundary = external_boundary
            logger.debug(f"  עמודות | גבול חיצוני (Surya) ב-x={boundary:.0f}")
        else:
            boundary = self._find_column_boundary(body_words, page_width, page_x_min)

        if boundary is None:
            logger.debug("  עמודות | תוצאה: עמודה אחת")
            body_text = self._lines_to_text(lines)
            return self._prepend_headers(header_lines, body_text)

        if external_boundary is None:
            logger.debug(f"  עמודות | גבול (היסטוגרמה) ב-x={boundary:.0f}")

        page_center = (page_x_min + page_x_max) / 2

        # --- Column-first approach ---
        # Split body words into right/left columns BEFORE grouping into
        # lines.  This avoids the fragile gutter-split logic that fails
        # when OCR produces irregular inter-word gaps.
        #
        # Classify words into right/left columns using a robust rule:
        # A word belongs to the RIGHT column if its rightmost edge (x_max)
        # extends clearly past the boundary.  In Hebrew RTL text, even the
        # last word on a right-column line (sitting near the gutter) has
        # its x_max well into the right column zone.  Left-column words,
        # by contrast, have their x_max near or to the left of the boundary.
        #
        # We add a small tolerance (20% of avg word width) to avoid pulling
        # left-column words whose x_max barely crosses the boundary.
        avg_word_width = (
            sum(w["x_max"] - w["x_min"] for w in body_words) / len(body_words)
        )
        split_x_max = boundary + avg_word_width * 0.2

        logger.debug(
            f"  עמודות | גבול={boundary:.0f}, סף_x_max={split_x_max:.0f}px "
            f"(גבול + {avg_word_width * 0.2:.0f}px)"
        )

        right_words = [w for w in body_words if w["x_max"] > split_x_max]
        left_words = [w for w in body_words if w["x_max"] <= split_x_max]

        logger.debug(
            f"  עמודות | חלוקת מילים: ימין={len(right_words)}, שמאל={len(left_words)}"
        )

        # Group into lines within each column independently
        right_col_lines = []
        if right_words:
            r_lines = self._group_into_lines(right_words)
            for line in r_lines:
                line.sort(key=lambda w: -w["x_center"])  # RTL
                y = sum(w["y_center"] for w in line) / len(line)
                right_col_lines.append((y, line))

        left_col_lines = []
        if left_words:
            l_lines = self._group_into_lines(left_words)
            for line in l_lines:
                line.sort(key=lambda w: -w["x_center"])  # RTL
                y = sum(w["y_center"] for w in line) / len(line)
                left_col_lines.append((y, line))

        # Detect centered/spanning lines: check each line from BOTH columns
        # for short centered lines that should span both columns (e.g. sub-headers)
        spanning_lines = []
        filtered_right = []
        filtered_left = []

        for y, line in right_col_lines:
            if self._is_centered_line(line, boundary, page_center, page_width):
                spanning_lines.append((y, line))
                logger.debug(
                    f"    → כותרת ממורכזת (ימין): '"
                    + " ".join(w["text"] for w in line) + "'"
                )
            else:
                filtered_right.append((y, line))

        for y, line in left_col_lines:
            if self._is_centered_line(line, boundary, page_center, page_width):
                spanning_lines.append((y, line))
                logger.debug(
                    f"    → כותרת ממורכזת (שמאל): '"
                    + " ".join(w["text"] for w in line) + "'"
                )
            else:
                filtered_left.append((y, line))

        right_col_lines = filtered_right
        left_col_lines = filtered_left

        # Sort by y within each column
        right_col_lines.sort(key=lambda x: x[0])
        left_col_lines.sort(key=lambda x: x[0])
        spanning_lines.sort(key=lambda x: x[0])

        logger.debug(
            f"  עמודות | תוצאה: 2 עמודות "
            f"(ימין={len(right_col_lines)}, שמאל={len(left_col_lines)}, חוצות={len(spanning_lines)})"
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
                _join_words(line) for _, line in top_spanning
            )
            parts.append(top_text)

        if right_col_lines:
            right_text = "\n".join(
                _join_words(line) for _, line in right_col_lines
            )
            parts.append(right_text)

        if left_col_lines:
            left_text = "\n".join(
                _join_words(line) for _, line in left_col_lines
            )
            parts.append(left_text)

        # Add spanning lines that appear within/after column content
        if mid_spanning:
            mid_text = "\n".join(
                _join_words(line) for _, line in mid_spanning
            )
            parts.append(mid_text)

        body_text = "\n\n".join(parts)
        return self._prepend_headers(header_lines, body_text)

    def _calc_avg_height(self, word_info: list[dict]) -> float:
        """Calculate average word height (proxy for body text font size)."""
        heights = sorted(w["height"] for w in word_info if w["height"] > 5)
        if not heights:
            return 30.0
        # Use median-area heights (drop top/bottom 10%) for robust average
        trim = max(1, len(heights) // 10)
        trimmed = heights[trim:-trim] if len(heights) > 20 else heights
        return sum(trimmed) / len(trimmed)

    def _split_headers_by_font_size(
        self, lines: list[list[dict]], avg_height: float
    ) -> tuple[list[dict], list[list[dict]]]:
        """
        Split lines into header lines and body lines based on font size.

        A line is a header if most of its words are significantly taller
        than the average word height (large font = header).

        Returns:
            (header_lines, body_lines) where header_lines is a list of
            dicts with 'text' and 'y' keys, body_lines is the remaining lines.
        """
        threshold = avg_height * self.HEADER_HEIGHT_FACTOR
        header_lines = []
        body_lines = []

        logger.debug(
            f"  גופן | סף_כותרת={threshold:.0f}px (ממוצע={avg_height:.0f} × {self.HEADER_HEIGHT_FACTOR})"
        )

        for line in lines:
            # Count how many words in this line have "large" height
            large_words = [w for w in line if w["height"] >= threshold]
            large_ratio = len(large_words) / len(line) if line else 0
            line_text = " ".join(w["text"] for w in sorted(line, key=lambda w: -w["x_center"]))

            if large_ratio >= 0.6:
                # Most words are large -> this is a header line
                sorted_line = sorted(line, key=lambda w: -w["x_center"])  # RTL
                text = _join_words(sorted_line)
                y = sum(w["y_center"] for w in line) / len(line)
                avg_h = sum(w["height"] for w in line) / len(line)
                header_lines.append({"text": text, "y": y, "avg_height": avg_h})
                logger.debug(
                    f"  גופן | כותרת ({large_ratio:.0%} גדולים, גובה_ממוצע={avg_h:.0f}px): '{line_text}'"
                )
            else:
                body_lines.append(line)

        # Sort headers by y position (top to bottom)
        header_lines.sort(key=lambda h: h["y"])
        return header_lines, body_lines

    # Marker wrapping font-size headers so post-processor won't remove them
    HEADER_MARKER_START = "[כותרת]"
    HEADER_MARKER_END = "[/כותרת]"

    def _prepend_headers(self, header_lines: list[dict], body_text: str) -> str:
        """Prepend extracted header lines before the body text, wrapped in markers."""
        if not header_lines:
            return body_text
        marked = [
            f"{self.HEADER_MARKER_START} {h['text']} {self.HEADER_MARKER_END}"
            for h in header_lines
        ]
        header_text = "\n".join(marked)
        if not body_text:
            return header_text
        return header_text + "\n\n" + body_text

    def _group_into_lines(self, word_info: list[dict]) -> list[list[dict]]:
        """Group words into text lines based on y-proximity.

        Uses a running-mean y anchor (updated as words join a line) and an
        adaptive threshold based on the tallest word in the current line,
        so that large title words (which span a wider y-range) are not
        split across lines.  A second merge pass re-attaches short
        fragments that drifted into a neighbouring line due to scan
        distortion or page curvature.
        """
        # Estimate baseline threshold from global word heights
        heights = [w["height"] for w in word_info if w["height"] > 5]
        avg_height = sum(heights) / len(heights) if heights else 30
        base_threshold = avg_height * 0.7

        # Sort by y_center
        sorted_words = sorted(word_info, key=lambda w: w["y_center"])

        lines: list[list[dict]] = []
        current_line = [sorted_words[0]]
        line_y_sum = sorted_words[0]["y_center"]

        for w in sorted_words[1:]:
            # Adaptive: use the tallest word in the current line as threshold
            # so large title words (h~90) get a wider tolerance than body (h~50)
            current_max_h = max(cw["height"] for cw in current_line)
            effective_threshold = max(base_threshold, current_max_h * 0.7)

            # Running mean keeps the anchor centred on the *actual* line
            line_y_mean = line_y_sum / len(current_line)

            if abs(w["y_center"] - line_y_mean) <= effective_threshold:
                current_line.append(w)
                line_y_sum += w["y_center"]
            else:
                current_line.sort(key=lambda w: -w["x_center"])  # RTL
                lines.append(current_line)
                current_line = [w]
                line_y_sum = w["y_center"]

        current_line.sort(key=lambda w: -w["x_center"])
        lines.append(current_line)

        # --- Second pass: merge short fragments into neighbours ----------
        # A fragment (≤3 words) whose y-range overlaps with an adjacent
        # line likely drifted there due to OCR / scan distortion.
        lines = self._merge_short_fragments(lines, avg_height)

        return lines

    @staticmethod
    def _merge_short_fragments(
        lines: list[list[dict]], avg_height: float
    ) -> list[list[dict]]:
        """Merge very short line fragments into their best vertical neighbour."""
        MAX_FRAGMENT_WORDS = 3
        # A fragment is worth merging only if a neighbour's y-range overlaps
        # within a generous tolerance (half the average word height).
        tolerance = avg_height * 0.5

        merged = True
        while merged:
            merged = False
            new_lines: list[list[dict]] = []
            skip: set[int] = set()
            for i, line in enumerate(lines):
                if i in skip:
                    continue
                if len(line) > MAX_FRAGMENT_WORDS:
                    new_lines.append(line)
                    continue

                # Fragment – find the closest neighbouring line by y-mean
                frag_y = sum(w["y_center"] for w in line) / len(line)
                best_idx = -1
                best_dist = float("inf")
                for j, other in enumerate(lines):
                    if j == i or j in skip or len(other) == 0:
                        continue
                    other_y = sum(w["y_center"] for w in other) / len(other)
                    dist = abs(frag_y - other_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = j

                if best_idx == -1 or best_dist > avg_height * 1.2:
                    # No suitable neighbour – keep as-is
                    new_lines.append(line)
                    continue

                # Merge fragment into the neighbour
                target = lines[best_idx]
                frag_count = len(line)
                target.extend(line)
                target.sort(key=lambda w: -w["x_center"])  # RTL
                line.clear()  # mark fragment as consumed
                merged = True
                logger.debug(
                    f"  קיבוץ | מיזוג פרגמנט ({frag_count} מילים) לשורה שכנה "
                    f"(Δy={best_dist:.0f}px)"
                )

            # Re-collect non-skipped lines (merged fragments already inside
            # their target which is added via the normal branch above).
            if merged:
                lines = [l for l in lines if len(l) > 0]
                # Re-sort each line RTL just in case
                for l in lines:
                    l.sort(key=lambda w: -w["x_center"])
                break  # restart outer loop
            else:
                lines = new_lines

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
        for i in range(third_start, third_end):
            if bins[i] < min_count:
                min_count = bins[i]

        # Among all bins with min_count, pick the one closest to the page
        # center.  This avoids selecting a small gap between words within
        # one column when the real gutter (also empty) sits near center.
        center_bin = num_bins / 2
        min_idx = -1
        best_dist = float("inf")
        for i in range(third_start, third_end):
            if bins[i] == min_count:
                dist = abs(i - center_bin)
                if dist < best_dist:
                    best_dist = dist
                    min_idx = i

        # The gutter bin should have very few words compared to average
        total_words = sum(bins)
        avg_per_bin = total_words / num_bins if num_bins > 0 else 0

        logger.debug(
            f"  עמודות | היסטוגרמה: מרזב={min_idx} ({min_count} מילים), "
            f"ממוצע={avg_per_bin:.1f}/תא, סף={max(1, avg_per_bin * 0.15):.1f}"
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
            logger.debug(
                f"    ממורכזת | כן: רוחב={width_ratio:.0%} (<40%), "
                f"היסט_מרכז={center_offset:.0%} (<15%)"
            )
            return True

        return False

    @staticmethod
    def _find_gutter_split(line: list[dict], boundary: float, page_width: float = 0) -> int | None:
        """Find where to split a line at the column gutter.

        Looks for the largest inter-word gap near the boundary.  If it is
        significantly wider than the other gaps, returns the split index
        (in the x-sorted word list) so that words[:idx] are the left side
        and words[idx:] are the right side.  Returns None if no real
        gutter gap is found.
        """
        line_text = " ".join(w["text"] for w in sorted(line, key=lambda w: -w["x_center"]))

        if len(line) < 3:
            logger.debug(f"    מרזב | דילוג (פחות מ-3 מילים): '{line_text}'")
            return None

        # A line that fits within a single column should never be split.
        # Justified text can have large inter-word gaps that mimic a gutter,
        # but the line still only spans ~45% of the page.  Require >55% to
        # even consider splitting.
        if page_width > 0:
            line_x_min = min(w["x_min"] for w in line)
            line_x_max = max(w["x_max"] for w in line)
            line_width = line_x_max - line_x_min
            width_ratio = line_width / page_width
            if line_width < page_width * 0.55:
                logger.debug(
                    f"    מרזב | דילוג (שורה צרה {width_ratio:.0%} < 55%): '{line_text}'"
                )
                return None

        # Sort words left-to-right by x_min
        sorted_words = sorted(line, key=lambda w: w["x_min"])

        # If the boundary runs through a word's bbox, it's not a gutter
        for w in sorted_words:
            if w["x_min"] <= boundary <= w["x_max"]:
                logger.debug(
                    f"    מרזב | דילוג (גבול חוצה מילה '{w['text']}'): '{line_text}'"
                )
                return None

        # Compute all inter-word gaps
        gaps = []
        for j in range(1, len(sorted_words)):
            gap = sorted_words[j]["x_min"] - sorted_words[j - 1]["x_max"]
            gaps.append((gap, j))

        if not gaps:
            return None

        # Log all gaps for debugging
        gap_details = [
            f"'{sorted_words[j-1]['text']}' ←{g:.0f}px→ '{sorted_words[j]['text']}'"
            for g, j in gaps
        ]
        logger.debug(f"    מרזב | רווחים בשורה: {', '.join(gap_details)}")

        # Find the gap that straddles the boundary
        boundary_gap = None
        boundary_j = None
        for gap_size, j in gaps:
            left_edge = sorted_words[j - 1]["x_max"]
            right_edge = sorted_words[j]["x_min"]
            if left_edge <= boundary <= right_edge:
                boundary_gap = gap_size
                boundary_j = j
                break

        if boundary_gap is None:
            logger.debug(f"    מרזב | דילוג (אין רווח על הגבול x={boundary:.0f}): '{line_text}'")
            return None

        # Compare to the MAXIMUM gap in the rest of the line.
        # A real gutter must be clearly larger than normal word gaps.
        other_gaps = [g for g, j2 in gaps if j2 != boundary_j]
        if not other_gaps:
            logger.debug(
                f"    מרזב | פיצול (רווח יחיד על הגבול, {boundary_gap:.0f}px): '{line_text}'"
            )
            return boundary_j  # only one gap and it's at the boundary

        max_other_gap = max(other_gaps)
        ratio = boundary_gap / max_other_gap if max_other_gap > 0 else float("inf")
        if boundary_gap > max_other_gap * 1.8:
            logger.debug(
                f"    מרזב | ✂ פיצול! רווח_גבול={boundary_gap:.0f}px, "
                f"מקס_אחר={max_other_gap:.0f}px, יחס={ratio:.1f}x (סף=1.8x): '{line_text}'"
            )
            return boundary_j

        # Fallback: if one outlier gap inside a column is masking the real
        # gutter, compare against the MEDIAN gap instead.  A single OCR
        # artefact (e.g. extra whitespace between two words) can inflate
        # max_other_gap and defeat the 1.8x check above.  The median is
        # robust to such outliers.
        sorted_others = sorted(other_gaps)
        median_other = sorted_others[len(sorted_others) // 2]
        avg_word_h = sum(w["y_max"] - w["y_min"] for w in line) / len(line)
        median_ratio = boundary_gap / median_other if median_other > 0 else float("inf")
        if boundary_gap > median_other * 2.0 and boundary_gap > avg_word_h:
            logger.debug(
                f"    מרזב | ✂ פיצול (fallback מדיאנה)! רווח_גבול={boundary_gap:.0f}px, "
                f"מדיאנה={median_other:.0f}px, יחס={median_ratio:.1f}x (סף=2.0x), "
                f"מקס_אחר={max_other_gap:.0f}px: '{line_text}'"
            )
            return boundary_j

        logger.debug(
            f"    מרזב | לא פוצל: רווח_גבול={boundary_gap:.0f}px, "
            f"מקס_אחר={max_other_gap:.0f}px, יחס={ratio:.1f}x (צריך >1.8x), "
            f"מדיאנה={median_other:.0f}px, יחס_מדיאנה={median_ratio:.1f}x (צריך >2.0x): '{line_text}'"
        )
        return None

    def _lines_to_text(self, lines: list[list[dict]]) -> str:
        """Convert lines of words to text (single column)."""
        return "\n".join(_join_words(line) for line in lines)
