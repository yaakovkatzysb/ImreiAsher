"""
Hebrew dictionary checker for OCR quality validation.
Checks words against a Hebrew dictionary and suggests corrections.
"""

import re
from pathlib import Path


class HebrewDictionaryChecker:
    """Check OCR text against a Hebrew word dictionary."""

    def __init__(self, dictionary_path: str = ""):
        self.words = set()
        if dictionary_path and Path(dictionary_path).exists():
            self._load_dictionary(dictionary_path)

    def _load_dictionary(self, path: str):
        """Load Hebrew words from a text file (one word per line)."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self.words.add(word)

    def has_dictionary(self) -> bool:
        return len(self.words) > 0

    def check_text(self, text: str) -> dict:
        """
        Check all Hebrew words in text against the dictionary.

        Returns:
            dict with valid_words, unknown_words, unknown_list, suggestions
        """
        hebrew_words = self._extract_hebrew_words(text)

        if not hebrew_words:
            return {
                "valid_words": 0,
                "unknown_words": 0,
                "total_words": 0,
                "accuracy": 0.0,
                "unknown_list": [],
                "suggestions": [],
            }

        valid = []
        unknown = []

        for word in hebrew_words:
            if self._is_known(word):
                valid.append(word)
            else:
                unknown.append(word)

        # Generate suggestions for unknown words
        suggestions = []
        for word in unknown[:50]:  # limit to first 50 to avoid slow processing
            suggestion = self._find_closest(word)
            if suggestion:
                suggestions.append(suggestion)

        total = len(hebrew_words)
        return {
            "valid_words": len(valid),
            "unknown_words": len(unknown),
            "total_words": total,
            "accuracy": round(len(valid) / total, 4) if total > 0 else 0.0,
            "unknown_list": unknown[:100],  # limit output size
            "suggestions": suggestions,
        }

    def _extract_hebrew_words(self, text: str) -> list[str]:
        """Extract all Hebrew words (2+ chars) from text."""
        # Match sequences of Hebrew characters (including final letters)
        pattern = re.compile(r'[\u0590-\u05FF]{2,}')
        return pattern.findall(text)

    def _is_known(self, word: str) -> bool:
        """Check if word is in dictionary (with common variations)."""
        if word in self.words:
            return True
        # Try without prefix letters (ב, ה, ו, כ, ל, מ, ש)
        prefixes = "בהוכלמש"
        if len(word) > 2 and word[0] in prefixes:
            if word[1:] in self.words:
                return True
        # Two-letter prefix (מה, של, etc.)
        if len(word) > 3 and word[:2] in ("מה", "של", "בה", "לה", "וה", "כש", "מש"):
            if word[2:] in self.words:
                return True
        return False

    def _find_closest(self, word: str) -> tuple | None:
        """
        Find the closest dictionary word using Levenshtein distance.

        Returns:
            (original, suggestion, confidence) or None
        """
        if not self.words:
            return None

        best_match = None
        best_distance = 2  # max distance to consider

        for dict_word in self.words:
            # Quick length filter
            if abs(len(dict_word) - len(word)) > 1:
                continue
            dist = self._levenshtein(word, dict_word)
            if dist < best_distance:
                best_distance = dist
                best_match = dict_word
            if dist == 1:
                break  # good enough

        if best_match and best_distance <= 1:
            confidence = 1.0 - (best_distance * 0.1)
            return (word, best_match, round(confidence, 2))
        return None

    def _levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]
