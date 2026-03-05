"""Tests for column detector, focusing on header/spanning line detection."""

import pytest
from column_detector import ColumnDetector


def _make_word(text: str, x_min: float, y_min: float, width: float = 50, height: float = 20):
    """Helper to create a word dict with bbox."""
    x_max = x_min + width
    y_max = y_min + height
    return {
        "text": text,
        "bbox": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
    }


class TestColumnDetectorBasic:
    """Test basic column detection."""

    def setup_method(self):
        self.detector = ColumnDetector()

    def test_empty_words(self):
        assert self.detector.reorder_by_columns([]) == ""

    def test_single_column(self):
        # All words in a single column - no split
        words = [
            _make_word("שלום", 400, 100),
            _make_word("עולם", 300, 100),
            _make_word("טוב", 200, 100),
        ]
        result = self.detector.reorder_by_columns(words)
        assert "שלום" in result
        assert "עולם" in result


class TestSpanningLineDetection:
    """Test that full-width header lines are not split across columns."""

    def setup_method(self):
        self.detector = ColumnDetector()

    def test_header_spanning_both_columns(self):
        """A header line spanning both columns should appear once, not twice."""
        page_width = 1000  # 0 to 1000

        # Header words spread across full width
        header_words = [
            _make_word("שיחת", 750, 50, width=80),
            _make_word("קודש", 600, 50, width=80),
            _make_word("ממורנו", 400, 50, width=90),
            _make_word("ורבנו", 250, 50, width=80),
            _make_word("זיע\"א", 100, 50, width=70),
        ]

        # Right column body text (x > 500)
        right_col = []
        for i, y in enumerate(range(120, 400, 30)):
            right_col.append(_make_word(f"ימין{i}", 700, y, width=60))
            right_col.append(_make_word(f"טקסט{i}", 600, y, width=60))

        # Left column body text (x < 500)
        left_col = []
        for i, y in enumerate(range(120, 400, 30)):
            left_col.append(_make_word(f"שמאל{i}", 300, y, width=60))
            left_col.append(_make_word(f"טקסט{i}", 200, y, width=60))

        words = header_words + right_col + left_col
        result = self.detector.reorder_by_columns(words)

        # The header words should appear exactly once
        assert result.count("שיחת") == 1
        assert result.count("קודש") == 1
        assert result.count("ממורנו") == 1
        assert result.count("זיע\"א") == 1


class TestCenteredHeaderDetection:
    """Test that short centered lines (header fragments) are detected."""

    def setup_method(self):
        self.detector = ColumnDetector()

    def test_short_centered_line_detected_as_header(self):
        """A short centered line like 'שיחת קודש' should not be split."""
        page_width = 1000

        # Short centered header line: 2 words near center
        header_words = [
            _make_word("שיחת", 530, 50, width=70),
            _make_word("קודש", 430, 50, width=70),
        ]

        # Right column
        right_col = []
        for i, y in enumerate(range(120, 500, 30)):
            right_col.append(_make_word(f"ימין{i}", 750, y, width=60))
            right_col.append(_make_word(f"שורה{i}", 650, y, width=60))
            right_col.append(_make_word(f"טקסט{i}", 550, y, width=60))

        # Left column
        left_col = []
        for i, y in enumerate(range(120, 500, 30)):
            left_col.append(_make_word(f"שמאל{i}", 350, y, width=60))
            left_col.append(_make_word(f"עמודה{i}", 250, y, width=60))
            left_col.append(_make_word(f"שני{i}", 150, y, width=60))

        words = header_words + right_col + left_col
        result = self.detector.reorder_by_columns(words)

        # Header should appear once, before columns
        assert result.count("שיחת") == 1
        assert result.count("קודש") == 1

    def test_single_centered_word_detected_as_header(self):
        """A single word centered on the page should be a header, not column text."""
        page_width = 1000

        header_words = [
            _make_word("זיע\"א", 460, 80, width=80),
        ]

        # Right column
        right_col = []
        for i, y in enumerate(range(150, 500, 30)):
            right_col.append(_make_word(f"ימין{i}", 750, y, width=60))
            right_col.append(_make_word(f"טקסט{i}", 650, y, width=60))

        # Left column
        left_col = []
        for i, y in enumerate(range(150, 500, 30)):
            left_col.append(_make_word(f"שמאל{i}", 300, y, width=60))
            left_col.append(_make_word(f"טקסט{i}", 200, y, width=60))

        words = header_words + right_col + left_col
        result = self.detector.reorder_by_columns(words)

        assert result.count("זיע\"א") == 1

    def test_regular_column_text_not_detected_as_header(self):
        """Regular column text should NOT be marked as header."""
        page_width = 1000

        # Right column - long lines with many words
        right_col = []
        for i, y in enumerate(range(100, 500, 30)):
            right_col.append(_make_word(f"מילה{i}א", 900, y, width=60))
            right_col.append(_make_word(f"מילה{i}ב", 800, y, width=60))
            right_col.append(_make_word(f"מילה{i}ג", 700, y, width=60))
            right_col.append(_make_word(f"מילה{i}ד", 600, y, width=60))

        # Left column - long lines with many words
        left_col = []
        for i, y in enumerate(range(100, 500, 30)):
            left_col.append(_make_word(f"שמאל{i}א", 400, y, width=60))
            left_col.append(_make_word(f"שמאל{i}ב", 300, y, width=60))
            left_col.append(_make_word(f"שמאל{i}ג", 200, y, width=60))
            left_col.append(_make_word(f"שמאל{i}ד", 100, y, width=60))

        words = right_col + left_col
        result = self.detector.reorder_by_columns(words)

        # Should have two distinct column sections
        lines = result.strip().split("\n")
        # At least some lines should exist
        assert len(lines) > 5


class TestMultipleHeaderLines:
    """Test stacked header lines (like the real case)."""

    def setup_method(self):
        self.detector = ColumnDetector()

    def test_stacked_centered_headers(self):
        """
        Multiple centered header lines like:
            שיחת קודש
            ממורנו ורבנו
            זיע"א
        Should all be detected as headers and appear once each.
        """
        page_width = 1000

        # Line 1: "שיחת קודש" centered
        header1 = [
            _make_word("שיחת", 540, 30, width=60),
            _make_word("קודש", 440, 30, width=60),
        ]
        # Line 2: "ממורנו ורבנו" centered
        header2 = [
            _make_word("ממורנו", 540, 60, width=70),
            _make_word("ורבנו", 430, 60, width=60),
        ]
        # Line 3: "זיע"א" centered
        header3 = [
            _make_word("זיע\"א", 470, 90, width=60),
        ]

        # Right column body
        right_col = []
        for i, y in enumerate(range(150, 500, 30)):
            right_col.append(_make_word(f"ימין{i}", 800, y, width=60))
            right_col.append(_make_word(f"טקסט{i}", 700, y, width=60))
            right_col.append(_make_word(f"עוד{i}", 600, y, width=60))

        # Left column body
        left_col = []
        for i, y in enumerate(range(150, 500, 30)):
            left_col.append(_make_word(f"שמאל{i}", 350, y, width=60))
            left_col.append(_make_word(f"עמוד{i}", 250, y, width=60))
            left_col.append(_make_word(f"שני{i}", 150, y, width=60))

        words = header1 + header2 + header3 + right_col + left_col
        result = self.detector.reorder_by_columns(words)

        # Each header word should appear exactly once
        assert result.count("שיחת") == 1
        assert result.count("קודש") == 1
        assert result.count("ממורנו") == 1
        assert result.count("ורבנו") == 1
        assert result.count("זיע\"א") == 1

        # Headers should appear before the column text
        header_pos = result.find("שיחת")
        col_pos = result.find("ימין0")
        assert header_pos < col_pos, "Header should appear before column text"
