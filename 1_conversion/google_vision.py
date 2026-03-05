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
        if cached:
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
            return {"text": "", "confidence": 0.0, "languages": [], "blocks": []}

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

        # Extract text blocks with positions
        blocks = []
        for page in annotation.pages:
            for block in page.blocks:
                block_text = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join(
                            symbol.text for symbol in word.symbols
                        )
                        block_text += word_text + " "
                    block_text += "\n"

                blocks.append({
                    "text": block_text.strip(),
                    "confidence": block.confidence,
                    "bbox": self._extract_bbox(block.bounding_box),
                })

        return {
            "text": annotation.text,
            "confidence": round(avg_confidence, 4),
            "languages": sorted(languages),
            "blocks": blocks,
        }

    def _extract_bbox(self, bounding_box) -> list:
        """Extract bounding box as list of [x, y] points."""
        return [
            [vertex.x, vertex.y]
            for vertex in bounding_box.vertices
        ]
