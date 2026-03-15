"""
Google Cloud Vision API wrapper for Hebrew OCR.
Handles text detection with confidence scores and caching.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

from google.cloud import vision

logger = logging.getLogger(__name__)


class GoogleVisionOCR:
    """Wrapper around Google Cloud Vision API for text detection."""

    def __init__(self, credentials_path: str = "", cache_dir: str = "data/cache"):
        if credentials_path:
            abs_path = str(Path(credentials_path).resolve())
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
        self.client = vision.ImageAnnotatorClient()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes).hexdigest()

    def _load_from_cache(self, cache_key: str) -> dict | None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, result: dict):
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def detect_text(self, image_bytes: bytes) -> dict:
        """
        Send image to Google Vision API for text detection.

        Args:
            image_bytes: Raw image bytes (PNG/JPEG)

        Returns:
            dict with keys: text, confidence, languages, blocks
        """
        cache_key = self._get_cache_key(image_bytes)
        cached = self._load_from_cache(cache_key)
        if cached and "words" in cached:
            # Check if cache has per-symbol confidence data (new format).
            # Old cache entries lack this, so re-run OCR to get full data.
            words = cached["words"]
            if words and "symbol_confidences" in words[0]:
                # Apply dedup and quote detachment on cached data too
                cached["words"] = self._deduplicate_words(words)
                self._detach_trailing_quotes(cached["words"])
                return cached

        image = vision.Image(content=image_bytes)
        response = self.client.document_text_detection(
            image=image,
            image_context=vision.ImageContext(
                language_hints=["he"]
            ),
        )

        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        result = self._parse_response(response)
        self._save_to_cache(cache_key, result)
        return result

    def _parse_response(self, response) -> dict:
        """Parse the API response into a clean dict."""
        annotation = response.full_text_annotation

        if not annotation.text:
            return {"text": "", "confidence": 0.0, "languages": [], "words": []}

        # Calculate average confidence from pages
        confidences = []
        for page in annotation.pages:
            for block in page.blocks:
                confidences.append(block.confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Detect languages
        languages = set()
        for page in annotation.pages:
            if page.property and page.property.detected_languages:
                for lang in page.property.detected_languages:
                    languages.add(lang.language_code)

        # Log block structure to understand Vision's layout detection
        for page in annotation.pages:
            for block_idx, block in enumerate(page.blocks):
                verts = block.bounding_box.vertices
                xs = [v.x for v in verts]
                ys = [v.y for v in verts]
                para_count = len(block.paragraphs)
                word_count = sum(len(p.words) for p in block.paragraphs)
                first_words = []
                for p in block.paragraphs:
                    for w in p.words[:3]:
                        first_words.append("".join(s.text for s in w.symbols))
                logger.debug(
                    f"  vision block {block_idx} | "
                    f"x=[{min(xs)}-{max(xs)}], y=[{min(ys)}-{max(ys)}] | "
                    f"{para_count} paragraphs, {word_count} words | "
                    f"preview: {' '.join(first_words)}"
                )

        # Extract individual words with positions (finest granularity for column detection)
        words = []
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = self._build_word_text(word.symbols)
                        if word_text.strip():
                            # Per-symbol confidence for confusion correction
                            sym_conf = [
                                {"char": s.text, "confidence": s.confidence}
                                for s in word.symbols
                            ]
                            words.append({
                                "text": word_text,
                                "confidence": word.confidence,
                                "bbox": self._extract_bbox(word.bounding_box),
                                "symbol_confidences": sym_conf,
                            })

        # Remove duplicate words from overlapping blocks
        words = self._deduplicate_words(words)

        # Post-process: detach trailing quotes that belong to the next word
        self._detach_trailing_quotes(words)

        return {
            "text": annotation.text,
            "confidence": round(avg_confidence, 4),
            "languages": sorted(languages),
            "words": words,
        }

    @staticmethod
    def _deduplicate_words(words: list[dict]) -> list[dict]:
        """Remove duplicate words that Google Vision returns from overlapping blocks.

        When a word sits near a block boundary, Vision may include it in
        both adjacent blocks.  We detect this by checking for words with
        the same text whose bounding boxes overlap significantly.
        """
        if not words:
            return words

        def _bbox_overlap(a, b) -> bool:
            """Check if two bounding boxes overlap by more than 50%."""
            ax = [p[0] for p in a]
            ay = [p[1] for p in a]
            bx = [p[0] for p in b]
            by = [p[1] for p in b]

            a_min_x, a_max_x = min(ax), max(ax)
            a_min_y, a_max_y = min(ay), max(ay)
            b_min_x, b_max_x = min(bx), max(bx)
            b_min_y, b_max_y = min(by), max(by)

            # Intersection
            ix = max(0, min(a_max_x, b_max_x) - max(a_min_x, b_min_x))
            iy = max(0, min(a_max_y, b_max_y) - max(a_min_y, b_min_y))
            intersection = ix * iy

            # Smaller area
            a_area = (a_max_x - a_min_x) * (a_max_y - a_min_y)
            b_area = (b_max_x - b_min_x) * (b_max_y - b_min_y)
            smaller = min(a_area, b_area)

            return smaller > 0 and intersection > smaller * 0.5

        kept = []
        for w in words:
            is_dup = False
            for k in kept:
                if k["text"] == w["text"] and _bbox_overlap(k["bbox"], w["bbox"]):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(w)
        return kept

    # Yod is often misread by OCR when the actual glyph is a geresh (׳).
    # A geresh is much smaller than a regular letter, so we detect this by
    # comparing the last symbol's height to the rest of the word.
    _GERESH_HEIGHT_RATIO = 0.6  # last symbol must be shorter than this ratio of avg

    _QUOTE_CHARS = set('"״""')

    def _detach_trailing_quotes(self, words: list[dict]):
        """Move quote marks to the correct word when they are opening quotes.

        Handles three cases:
        1. Trailing quote on abbreviation: כמש"כ" → כמש"כ + "next
        2. Standalone quote after abbreviation: כמש"כ, ", next → כמש"כ, "next
        3. Standalone/trailing quote using spatial proximity (bbox):
           attach to whichever neighbor is physically closer.
        """
        i = 0
        while i < len(words) - 1:
            text = words[i]["text"]

            # Case 1: word ends with a quote and already has an internal quote
            if len(text) > 2 and text[-1] in self._QUOTE_CHARS:
                inner = text[1:-1]
                if any(c in self._QUOTE_CHARS for c in inner):
                    words[i]["text"] = text[:-1]
                    words[i + 1]["text"] = text[-1] + words[i + 1]["text"]
                    i += 1
                    continue

            # Case 2: standalone quote word after an abbreviation (word with internal quote)
            # → it's an opening quote, attach to the next word
            if text in self._QUOTE_CHARS and i > 0 and i + 1 < len(words):
                prev = words[i - 1]["text"]
                has_internal_quote = (
                    len(prev) > 2
                    and any(c in self._QUOTE_CHARS for c in prev[1:-1])
                )
                if has_internal_quote:
                    words[i + 1]["text"] = text + words[i + 1]["text"]
                    words.pop(i)
                    continue

            # Case 3a: standalone quote — use bbox to decide opening vs closing
            if text in self._QUOTE_CHARS and i + 1 < len(words):
                if self._quote_closer_to_next(words, i):
                    words[i + 1]["text"] = text + words[i + 1]["text"]
                    words.pop(i)
                    continue

            # Case 3b: trailing quote on a regular word (no internal quotes) —
            # use bbox gap to decide if it's an opening quote for the next word
            if (len(text) > 2 and text[-1] in self._QUOTE_CHARS
                    and not any(c in self._QUOTE_CHARS for c in text[1:-1])
                    and i + 1 < len(words)):
                if self._trailing_quote_is_opener(words, i):
                    words[i]["text"] = text[:-1]
                    words[i + 1]["text"] = text[-1] + words[i + 1]["text"]
                    i += 1
                    continue

            i += 1

    @staticmethod
    def _quote_closer_to_next(words: list[dict], i: int) -> bool:
        """Check if a standalone quote word is spatially closer to the next word.

        Uses edge-to-edge distances (not centers) so that word width
        doesn't skew the result.  In RTL Hebrew:
        - gap to prev word  = prev word's LEFT edge − quote's RIGHT edge
        - gap to next word  = quote's LEFT edge − next word's RIGHT edge
        """
        curr = words[i].get("bbox")
        next_w = words[i + 1].get("bbox") if i + 1 < len(words) else None
        prev_w = words[i - 1].get("bbox") if i > 0 else None

        if not curr or not next_w:
            return False

        quote_right = max(p[0] for p in curr)
        quote_left = min(p[0] for p in curr)
        next_right = max(p[0] for p in next_w)

        if prev_w:
            prev_left = min(p[0] for p in prev_w)
            gap_to_prev = abs(prev_left - quote_right)
            gap_to_next = abs(quote_left - next_right)
            return gap_to_next < gap_to_prev

        # No previous word — default to attaching to next
        return True

    _HEBREW_RE = __import__("re").compile(r"[\u0590-\u05FF]")

    @classmethod
    def _trailing_quote_is_opener(cls, words: list[dict], i: int) -> bool:
        """Check if a trailing quote on a word is an opening quote for the next word.

        Uses a content-based heuristic: if the next word starts with a
        Hebrew letter, the quote is likely an opening quote.  If the next
        word is punctuation, it's a closing quote that should stay.
        """
        if i + 1 >= len(words):
            return False

        next_text = words[i + 1]["text"]
        if not next_text:
            return False

        # Opening quote → next word starts with Hebrew letter
        return bool(cls._HEBREW_RE.match(next_text[0]))

    def _build_word_text(self, symbols) -> str:
        """Build word text from symbols, fixing yod-that-is-actually-geresh."""
        if len(symbols) < 2:
            return "".join(s.text for s in symbols)

        last = symbols[-1]
        if last.text != "י":
            return "".join(s.text for s in symbols)

        # Compare last symbol height to the average of the others
        def _sym_height(sym):
            if not sym.bounding_box or not sym.bounding_box.vertices:
                return None
            ys = [v.y for v in sym.bounding_box.vertices]
            return max(ys) - min(ys)

        last_h = _sym_height(last)
        if last_h is None:
            return "".join(s.text for s in symbols)

        other_heights = [_sym_height(s) for s in symbols[:-1]]
        other_heights = [h for h in other_heights if h is not None and h > 0]
        if not other_heights:
            return "".join(s.text for s in symbols)

        avg_h = sum(other_heights) / len(other_heights)
        if avg_h > 0 and last_h / avg_h < self._GERESH_HEIGHT_RATIO:
            return "".join(s.text for s in symbols[:-1]) + "'"

        return "".join(s.text for s in symbols)

    def _extract_bbox(self, bounding_box) -> list:
        """Extract bounding box as list of [x, y] points."""
        return [
            [vertex.x, vertex.y]
            for vertex in bounding_box.vertices
        ]
