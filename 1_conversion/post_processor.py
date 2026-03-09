"""
Post-processor for OCR text output.

Handles:
1. Removing interstitial text (continuation markers between pages)
2. Removing repeating headers that appear on every page
3. Adding a structured document header template
"""

import re
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class PostProcessor:
    """Cleans and structures OCR output text."""

    def __init__(self, config: dict):
        """
        Args:
            config: Post-processing config section from config.yaml
        """
        self.interstitial_phrases = config.get("interstitial_phrases", [])
        self.repeating_headers = config.get("repeating_headers", [])
        self.header_template = config.get("header_template", "")
        self.header_scan_lines = config.get("header_scan_lines", 3)
        self.header_min_page_ratio = config.get("header_min_page_ratio", 0.5)

    def process_document(self, pages_data: list, filename: str) -> list:
        """
        Apply all post-processing steps to a list of page results.

        Args:
            pages_data: List of dicts with 'text', 'page_number', etc.
            filename: PDF filename (used to extract issue number/date)

        Returns:
            Updated pages_data with cleaned text
        """
        if not pages_data:
            return pages_data

        # Step 1: Detect repeating headers across pages
        auto_headers = self._detect_repeating_headers(pages_data)
        all_headers = auto_headers + self.repeating_headers
        if all_headers:
            logger.info(f"  עיבוד-אחר | כותרות חוזרות: {all_headers}")

        # Step 2: Clean each page
        # Track font-size headers seen so far (keep first occurrence only)
        seen_headers = set()
        for page in pages_data:
            if not page.get("text"):
                continue
            page["text"] = self._clean_page(page["text"], all_headers, seen_headers)

        # Step 3: Add structured header to first page
        issue_num, issue_date = self._extract_issue_info(filename)
        if self.header_template and issue_num:
            header = self._build_header(issue_num, issue_date)
            if pages_data[0].get("text"):
                pages_data[0]["text"] = header + "\n\n" + pages_data[0]["text"]
            else:
                pages_data[0]["text"] = header

        return pages_data

    def _clean_page(self, text: str, repeating_headers: list, seen_headers: set = None) -> str:
        """Remove interstitial text and repeating headers from a single page."""
        if seen_headers is None:
            seen_headers = set()
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines (preserve them)
            if not stripped:
                cleaned_lines.append(line)
                continue

            # Font-size detected headers: keep first occurrence, remove duplicates
            if "[כותרת]" in stripped:
                header_text = stripped.replace("[כותרת]", "").replace("[/כותרת]", "").strip()
                if header_text in seen_headers:
                    logger.debug(f"  עיבוד-אחר | הוסרה כותרת כפולה: {header_text}")
                    continue
                seen_headers.add(header_text)
                cleaned_lines.append(line)
                continue

            # Check interstitial phrases
            if self._is_interstitial(stripped):
                logger.debug(f"  עיבוד-אחר | הוסר טקסט ביניים: {stripped}")
                continue

            # Check repeating headers
            if self._is_repeating_header(stripped, repeating_headers):
                logger.debug(f"  עיבוד-אחר | הוסרה כותרת חוזרת: {stripped}")
                continue

            cleaned_lines.append(line)

        # Remove leading/trailing blank lines
        result = "\n".join(cleaned_lines).strip()

        # Fix duplicate punctuation from OCR (e.g. ",," -> ",", "--" -> "-")
        result = re.sub(r"([,\.;:!?\-])\1+", r"\1", result)

        return result

    def _is_interstitial(self, line: str) -> bool:
        """Check if a line is an interstitial continuation marker."""
        # Normalize spaces around geresh/quotes for OCR variants like "בעמ ' הבא"
        normalized = re.sub(r"\s*['׳\"״]\s*", "'", line)

        for phrase in self.interstitial_phrases:
            if phrase in line or phrase in normalized:
                return True

        # Generic patterns for continuation markers
        if re.search(r"המשך\s+.+\s+בעמוד", line):
            return True
        if re.search(r"המשך\s+מעמ", line):
            return True
        if re.search(r"המשך\s+בעמ", normalized):
            return True
        # "המשך יבוא אי"ה" and variants
        if re.search(r"המשך\s+יבוא", line):
            return True

        return False

    def _is_repeating_header(self, line: str, headers: list) -> bool:
        """Check if a line matches a known repeating header."""
        for header in headers:
            # Fuzzy match: allow minor OCR differences
            if header in line or line in header:
                return True
            # Similarity check for OCR errors
            if self._similar(line, header, threshold=0.85):
                return True
        return False

    def _similar(self, a: str, b: str, threshold: float = 0.85) -> float:
        """Simple character-level similarity ratio."""
        if not a or not b:
            return 0.0
        # Quick length check
        if abs(len(a) - len(b)) / max(len(a), len(b)) > (1 - threshold):
            return False
        # Character overlap
        common = sum(1 for ca, cb in zip(a, b) if ca == cb)
        ratio = common / max(len(a), len(b))
        return ratio >= threshold

    def _detect_repeating_headers(self, pages_data: list) -> list:
        """
        Automatically detect text that repeats at the top of most pages.

        Returns:
            List of header strings found
        """
        if len(pages_data) < 3:
            return []

        # Collect top N lines from each page
        top_lines = Counter()
        for page in pages_data:
            text = page.get("text", "")
            if not text:
                continue
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines[: self.header_scan_lines]:
                if len(line) > 3:  # Skip very short lines
                    top_lines[line] += 1

        # Lines appearing in more than half the pages are likely headers
        min_count = max(2, int(len(pages_data) * self.header_min_page_ratio))
        headers = [line for line, count in top_lines.items() if count >= min_count]

        return headers

    def _extract_issue_info(self, filename: str) -> tuple[str, str]:
        """
        Extract issue number and date from PDF filename.

        Expected patterns:
            - "החברותא_123_תשפה.pdf"
            - "hachavruta_123_5785.pdf"
            - Various combinations with numbers and dates

        Returns:
            (issue_number, date_string) - either may be empty string
        """
        stem = Path(filename).stem

        # Extract numbers from filename
        numbers = re.findall(r"\d+", stem)

        # Extract Hebrew date patterns (תשפ"ה, תשפ״ה, etc.)
        hebrew_date = re.search(r"[תש][שׁ][פצ][\"״׳']?[א-ת]", stem)

        issue_num = numbers[0] if numbers else ""
        issue_date = hebrew_date.group(0) if hebrew_date else ""

        # If no Hebrew date, try to find any remaining text as date
        if not issue_date:
            parts = re.split(r"[_\-\s]+", stem)
            for part in parts:
                if part and not part.isdigit() and part not in ("החברותא", "hachavruta"):
                    issue_date = part
                    break

        return issue_num, issue_date

    def _build_header(self, issue_num: str, issue_date: str) -> str:
        """Build the structured document header from template."""
        header = self.header_template
        header = header.replace("{issue_number}", issue_num)
        header = header.replace("{issue_date}", issue_date)
        return header
