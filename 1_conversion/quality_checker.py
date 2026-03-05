"""
Quality checker for OCR results.
Level 1: Basic sanity checks (Hebrew ratio, weird chars, text length, spacing)
Level 2: Linguistic checks (dictionary, final letters, nikud, grammar)
"""

import re
import unicodedata


class QualityChecker:
    """Two-level quality checking for OCR text output."""

    # Hebrew Unicode range
    HEBREW_RANGE = re.compile(r'[\u0590-\u05FF]')
    # Hebrew final letters mapping: regular -> final
    FINAL_LETTERS = {'כ': 'ך', 'מ': 'ם', 'נ': 'ן', 'פ': 'ף', 'צ': 'ץ'}
    # Reverse mapping: final -> regular
    REGULAR_LETTERS = {v: k for k, v in FINAL_LETTERS.items()}
    # Characters that shouldn't appear in historical Hebrew newspapers
    WEIRD_CHARS = set('@#$©™®□■◊◆●○►◄▲▼')

    def __init__(self, config: dict):
        self.hebrew_ratio_min = config.get("hebrew_ratio_min", 0.85)
        self.weird_chars_max = config.get("weird_chars_max", 10)
        self.chars_per_page_min = config.get("chars_per_page_min", 500)
        self.chars_per_page_max = config.get("chars_per_page_max", 5000)

    def check_all(self, text: str, num_pages: int = 1) -> dict:
        """
        Run all quality checks on text.

        Args:
            text: The OCR extracted text
            num_pages: Number of pages (for per-page averages)

        Returns:
            Full quality report dict
        """
        level1 = self.check_level1(text, num_pages)
        level2 = self.check_level2(text)

        # Calculate overall score (0-100)
        score = self._calculate_score(level1, level2)

        return {
            "level1": level1,
            "level2": level2,
            "score": score,
            "recommendation": self._get_recommendation(score),
        }

    def check_level1(self, text: str, num_pages: int = 1) -> dict:
        """Level 1: Basic sanity checks."""
        return {
            "hebrew_ratio": self._check_hebrew_ratio(text),
            "weird_chars": self._check_weird_chars(text),
            "text_length": self._check_text_length(text, num_pages),
            "spacing": self._check_spacing(text),
        }

    def check_level2(self, text: str) -> dict:
        """Level 2: Linguistic checks."""
        return {
            "final_letters": self._check_final_letters(text),
            "nikud": self._check_nikud(text),
            "grammar": self._check_basic_grammar(text),
        }

    # --- Level 1 Checks ---

    def _check_hebrew_ratio(self, text: str) -> dict:
        """Check ratio of Hebrew characters to total alphabetic characters."""
        if not text.strip():
            return {"ratio": 0.0, "passed": False, "detail": "empty text"}

        hebrew_count = len(self.HEBREW_RANGE.findall(text))
        # Count all alphabetic characters (Hebrew + Latin + other)
        alpha_count = sum(1 for c in text if unicodedata.category(c).startswith('L'))

        if alpha_count == 0:
            return {"ratio": 0.0, "passed": False, "detail": "no alphabetic chars"}

        ratio = hebrew_count / alpha_count
        return {
            "ratio": round(ratio, 4),
            "hebrew_chars": hebrew_count,
            "total_alpha": alpha_count,
            "passed": ratio >= self.hebrew_ratio_min,
        }

    def _check_weird_chars(self, text: str) -> dict:
        """Check for characters that shouldn't appear."""
        found = []
        for i, char in enumerate(text):
            if char in self.WEIRD_CHARS:
                found.append({"char": char, "position": i})

        return {
            "count": len(found),
            "passed": len(found) <= self.weird_chars_max,
            "found": found[:20],  # limit output
        }

    def _check_text_length(self, text: str, num_pages: int) -> dict:
        """Check average characters per page."""
        total_chars = len(text.strip())
        avg_per_page = total_chars / max(num_pages, 1)

        return {
            "total_chars": total_chars,
            "avg_per_page": round(avg_per_page),
            "num_pages": num_pages,
            "passed": self.chars_per_page_min <= avg_per_page <= self.chars_per_page_max,
        }

    def _check_spacing(self, text: str) -> dict:
        """Check for spacing anomalies."""
        double_spaces = len(re.findall(r'  ', text))
        triple_spaces = len(re.findall(r'   ', text))
        # Lines with no spaces (likely OCR error - words merged)
        lines = text.split('\n')
        no_space_lines = sum(1 for line in lines if len(line) > 30 and ' ' not in line)

        errors = double_spaces + triple_spaces * 2 + no_space_lines * 5

        return {
            "double_spaces": double_spaces,
            "triple_spaces": triple_spaces,
            "merged_lines": no_space_lines,
            "error_score": errors,
            "passed": errors < 50,
        }

    # --- Level 2 Checks ---

    def _check_final_letters(self, text: str) -> dict:
        """
        Check that final letters (ך, ם, ן, ף, ץ) are used correctly.
        Final letters should appear at end of words, regular at middle/start.
        """
        errors = []
        words = re.findall(r'[\u0590-\u05FF]+', text)

        for word in words:
            if len(word) < 2:
                continue

            # Check: final letter in middle of word
            for i, char in enumerate(word[:-1]):  # exclude last char
                if char in self.REGULAR_LETTERS:  # it's a final letter
                    errors.append({
                        "word": word,
                        "type": "final_in_middle",
                        "char": char,
                        "expected": self.REGULAR_LETTERS[char],
                    })

            # Check: regular letter at end of word (should be final)
            last_char = word[-1]
            if last_char in self.FINAL_LETTERS:
                errors.append({
                    "word": word,
                    "type": "regular_at_end",
                    "char": last_char,
                    "expected": self.FINAL_LETTERS[last_char],
                })

        return {
            "errors": len(errors),
            "passed": len(errors) < 10,
            "details": errors[:20],  # limit output
        }

    def _check_nikud(self, text: str) -> dict:
        """Check for nikud (vowel marks) issues."""
        # Hebrew nikud range: U+05B0 to U+05BD, U+05BF, U+05C1, U+05C2
        nikud_pattern = re.compile(r'[\u05B0-\u05BD\u05BF\u05C1\u05C2]')
        nikud_chars = nikud_pattern.findall(text)

        # Check for double nikud (two nikud marks in a row)
        double_nikud = len(re.findall(r'[\u05B0-\u05BD\u05BF\u05C1\u05C2]{2,}', text))

        # Check nikud on non-Hebrew chars
        nikud_on_non_hebrew = 0
        for i, char in enumerate(text):
            if nikud_pattern.match(char) and i > 0:
                prev = text[i - 1]
                if not self.HEBREW_RANGE.match(prev) and not nikud_pattern.match(prev):
                    nikud_on_non_hebrew += 1

        return {
            "nikud_count": len(nikud_chars),
            "double_nikud": double_nikud,
            "nikud_on_non_hebrew": nikud_on_non_hebrew,
            "passed": double_nikud < 5 and nikud_on_non_hebrew < 3,
        }

    def _check_basic_grammar(self, text: str) -> dict:
        """Basic grammar checks for common OCR errors."""
        words = re.findall(r'[\u0590-\u05FF]+', text)

        # Single-char Hebrew words (valid: ב, ה, ו, כ, ל, מ, ש, א)
        valid_single = set("בהוכלמשא")
        single_char_errors = sum(
            1 for w in words if len(w) == 1 and w not in valid_single
        )

        # Repeated chars (3+ same char in a row, like "ששש")
        repeated = len(re.findall(r'([\u0590-\u05FF])\1{2,}', text))

        return {
            "single_char_errors": single_char_errors,
            "repeated_chars": repeated,
            "passed": single_char_errors < 10 and repeated < 5,
        }

    # --- Scoring ---

    def _calculate_score(self, level1: dict, level2: dict) -> int:
        """Calculate overall quality score (0-100)."""
        score = 100

        # Level 1 deductions
        if not level1["hebrew_ratio"]["passed"]:
            ratio = level1["hebrew_ratio"]["ratio"]
            score -= int((self.hebrew_ratio_min - ratio) * 100)

        if not level1["weird_chars"]["passed"]:
            score -= min(15, level1["weird_chars"]["count"])

        if not level1["text_length"]["passed"]:
            score -= 10

        if not level1["spacing"]["passed"]:
            score -= min(10, level1["spacing"]["error_score"] // 5)

        # Level 2 deductions
        if not level2["final_letters"]["passed"]:
            score -= min(10, level2["final_letters"]["errors"])

        if not level2["nikud"]["passed"]:
            score -= 5

        if not level2["grammar"]["passed"]:
            score -= 5

        return max(0, min(100, score))

    def _get_recommendation(self, score: int) -> str:
        """Get recommendation based on quality score."""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "needs_review"
        else:
            return "poor_quality"
