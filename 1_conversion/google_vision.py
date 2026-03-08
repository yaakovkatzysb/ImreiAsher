"""
Google Cloud Vision API wrapper for Hebrew OCR.
Handles text detection with confidence scores and caching.
"""

import hashlib
import json
import os
from pathlib import Path

from google.cloud import vision


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

        # Post-process: detach trailing quotes that belong to the next word
        self._detach_trailing_quotes(words)

        return {
            "text": annotation.text,
            "confidence": round(avg_confidence, 4),
            "languages": sorted(languages),
            "words": words,
        }

    # Yod is often misread by OCR when the actual glyph is a geresh (׳).
    # A geresh is much smaller than a regular letter, so we detect this by
    # comparing the last symbol's height to the rest of the word.
    _GERESH_HEIGHT_RATIO = 0.6  # last symbol must be shorter than this ratio of avg

    _QUOTE_CHARS = set('"״""')

    def _detach_trailing_quotes(self, words: list[dict]):
        """Move trailing quote marks to the next word when they are opening quotes.

        If a word already contains an internal quote (abbreviation mark like כמש"כ)
        and ends with another quote, the trailing quote is likely the opening mark
        of the following quoted text and should attach to the next word.
        """
        i = 0
        while i < len(words) - 1:
            text = words[i]["text"]
            if len(text) > 2 and text[-1] in self._QUOTE_CHARS:
                # Check if there's already a quote inside the word (not at edges)
                inner = text[1:-1]
                if any(c in self._QUOTE_CHARS for c in inner):
                    # Detach the trailing quote
                    words[i]["text"] = text[:-1]
                    words[i + 1]["text"] = text[-1] + words[i + 1]["text"]
            i += 1

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
