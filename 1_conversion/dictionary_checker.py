"""
Hebrew dictionary checker for OCR quality validation.
Checks words against a Hebrew dictionary and suggests corrections.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Visually similar Hebrew letter pairs (common OCR confusions)
_CONFUSION_MAP: dict[str, list[str]] = {
    "כ": ["נ"],
    "נ": ["כ"],
    "ד": ["ר"],
    "ר": ["ד"],
    "ב": ["כ"],
    "ה": ["ח", "ת"],
    "ח": ["ה", "ת"],
    "ת": ["ה", "ח"],
    "ו": ["ז"],
    "ז": ["ו"],
    "ם": ["ס"],
    "ס": ["ם"],
    "ע": ["צ"],
    "צ": ["ע"],
    "ן": ["ו"],
    "ף": ["ק"],
    "ק": ["ף"],
}

# Common Hebrew abbreviations found in religious texts.
# Used as a fallback when no external dictionary is available.
_BUILTIN_ABBREVIATIONS = {
    'כמש"כ', 'כמש"נ', 'כמ"ש', 'עי"ש', 'וע"ש', 'ע"ש',
    'ז"ל', 'זצ"ל', 'זי"ע', 'זיע"א', 'שליט"א',
    'ע"ה', 'ע"א', 'ע"ב', 'ע"פ', 'ע"ד', 'ע"כ', 'ע"ז', 'ע"י',
    'ב"ה', 'בע"ה', 'בעז"ה', 'אי"ה', 'בס"ד', 'ה"ה',
    'ד"ה', 'וד"ה', 'ס"ק', 'שם',
    'הקב"ה', 'כביכ"ל',
    'מש"כ', 'שנ"ל',
    'אע"פ', 'אעפ"כ', 'אא"כ',
    'שבת"ל', 'ר"ל', 'ח"ו', 'חלי"ל',
    'א"כ', 'א"ל', 'ב"ד', 'ב"ב', 'ג"כ',
    'י"ל', 'צ"ל', 'צ"ע', 'נ"ל', 'נ"מ',
    'ר"ת', 'ס"ת', 'ר"י', 'ר"ה',
    'תוס"ד', 'וכה"ג', 'כה"ג',
    'פ"א', 'פ"ב', 'פ"ג', 'פ"ד',
    'ס"א', 'ס"ב', 'ס"ג', 'ס"ד',
    'אות"ו', 'אות"ה', 'אות"ן', 'אות"ם',
    'דב"ק', 'ספה"ק',
}


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

    def fix_confused_letters(self, text: str, words_data: list[dict] = None) -> str:
        """Try to fix OCR confusion between visually similar Hebrew letters.

        For each word that is NOT in the dictionary/abbreviation set, try
        swapping each letter with its visual confusion partners. If exactly
        one swap produces a known word, apply it.

        When symbol_confidences are available (from words_data), also fixes
        ambiguous cases by preferring alternatives where the swapped letter
        had low OCR confidence.

        Args:
            text: The text to fix.
            words_data: Optional list of word dicts with 'text' and
                'symbol_confidences' keys (from Google Vision).
        """
        known = self.words | _BUILTIN_ABBREVIATIONS

        # Build a lookup from word text to symbol confidences
        sym_conf_map = {}
        if words_data:
            for w in words_data:
                sc = w.get("symbol_confidences")
                if sc and w.get("text"):
                    sym_conf_map[w["text"]] = sc

        # Split preserving non-word separators
        tokens = re.split(r'(\s+)', text)
        changed = False

        for i, token in enumerate(tokens):
            if not token or token.isspace():
                continue

            clean = token.strip("'׳")
            sym_conf = sym_conf_map.get(token) or sym_conf_map.get(clean)
            is_known = self._is_known_or_abbrev(clean, known)

            # If word is known and we have no confidence data, skip it
            if is_known and not sym_conf:
                continue

            # If word is known but has confidence data, only check if any
            # symbol has low confidence (might be a confusion error)
            if is_known and sym_conf:
                has_low_conf = any(
                    s.get("confidence", 1.0) < self._LOW_CONFIDENCE_THRESHOLD
                    for s in sym_conf
                )
                if not has_low_conf:
                    continue

            # Try single-letter swaps
            best = self._try_confusion_swaps(clean, known, sym_conf)
            if best and best != clean:
                # Preserve any stripped characters
                prefix = token[:len(token) - len(token.lstrip("'׳"))]
                suffix = token[len(token.rstrip("'׳")):]
                tokens[i] = prefix + best + suffix
                logger.info(f"    תיקון בלבול אותיות: {token} → {tokens[i]}")
                changed = True

        return "".join(tokens) if changed else text

    def _is_known_or_abbrev(self, word: str, known: set) -> bool:
        """Check if word is in known set or dictionary (with prefix stripping)."""
        if word in known:
            return True
        if self._is_known(word):
            return True
        return False

    # Below this confidence, a symbol is considered uncertain
    _LOW_CONFIDENCE_THRESHOLD = 0.85

    def _try_confusion_swaps(
        self, word: str, known: set, sym_conf: list[dict] = None
    ) -> str | None:
        """Try swapping each letter with confusion partners, return match or None.

        When sym_conf is provided and multiple candidates exist, uses
        per-symbol confidence to pick the best one: prefer the candidate
        where the swapped position had the lowest OCR confidence.
        """
        candidates = []
        for pos, ch in enumerate(word):
            alternatives = _CONFUSION_MAP.get(ch, [])
            for alt in alternatives:
                candidate = word[:pos] + alt + word[pos + 1:]
                if candidate in known or self._is_known(candidate):
                    candidates.append((candidate, pos))

        if not candidates:
            return None

        # Unambiguous: exactly one candidate
        if len(candidates) == 1:
            return candidates[0][0]

        # Ambiguous: multiple candidates. Use symbol confidence to pick.
        if not sym_conf:
            return None

        # Find the candidate whose swapped position has the lowest confidence
        best_candidate = None
        lowest_conf = 1.0
        for candidate, pos in candidates:
            if pos < len(sym_conf):
                conf = sym_conf[pos].get("confidence", 1.0)
                if conf < lowest_conf:
                    lowest_conf = conf
                    best_candidate = candidate

        # Only correct if the symbol's confidence is genuinely low
        if best_candidate and lowest_conf < self._LOW_CONFIDENCE_THRESHOLD:
            return best_candidate
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
