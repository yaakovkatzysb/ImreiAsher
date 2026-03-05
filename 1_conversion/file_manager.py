"""
File manager for PDF reading and output file management.
Handles PDF-to-image conversion and organizing output files.
"""

import json
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF


class FileManager:
    """Manages input PDFs and output files."""

    def __init__(self, config: dict):
        self.input_dir = Path(config["paths"]["input_dir"])
        self.txt_dir = Path(config["paths"]["output_txt_dir"])
        self.json_dir = Path(config["paths"]["output_json_dir"])
        self.reports_dir = Path(config["paths"]["output_reports_dir"])

        # Create output directories
        for d in [self.txt_dir, self.json_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_pdf_list(self) -> list[Path]:
        """Get all PDF files from input directory, sorted by name."""
        pdfs = sorted(self.input_dir.glob("*.pdf"))
        return pdfs

    def pdf_to_images(self, pdf_path: Path, dpi: int = 300) -> list[bytes]:
        """
        Convert PDF pages to image bytes.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering

        Returns:
            List of PNG image bytes, one per page
        """
        doc = fitz.open(str(pdf_path))
        images = []

        zoom = dpi / 72  # default PDF resolution is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))

        doc.close()
        return images

    def get_pdf_metadata(self, pdf_path: Path) -> dict:
        """Extract basic metadata from PDF."""
        doc = fitz.open(str(pdf_path))
        metadata = {
            "filename": pdf_path.name,
            "pages": len(doc),
            "pdf_metadata": doc.metadata or {},
        }
        doc.close()
        return metadata

    def save_txt(self, filename: str, text: str):
        """Save extracted text to TXT file."""
        stem = Path(filename).stem
        output_path = self.txt_dir / f"{stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        return output_path

    def save_json(self, filename: str, data: dict):
        """Save result data to JSON file."""
        stem = Path(filename).stem
        output_path = self.json_dir / f"{stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return output_path

    def is_already_processed(self, filename: str) -> bool:
        """Check if a file was already processed (JSON output exists)."""
        stem = Path(filename).stem
        return (self.json_dir / f"{stem}.json").exists()

    def build_result(self, pdf_path: Path, pages_data: list, quality_report: dict) -> dict:
        """
        Build the full result dict for a processed PDF.

        Args:
            pdf_path: Original PDF path
            pages_data: List of OCR results per page
            quality_report: Quality check results

        Returns:
            Complete result dict ready for JSON output
        """
        full_text = "\n\n--- עמוד ---\n\n".join(
            page["text"] for page in pages_data if page.get("text")
        )

        avg_confidence = 0.0
        if pages_data:
            confidences = [p["confidence"] for p in pages_data if p.get("confidence")]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)

        metadata = self.get_pdf_metadata(pdf_path)

        return {
            "filename": pdf_path.name,
            "date_processed": datetime.now().isoformat(),
            "pages": metadata["pages"],
            "source": self._extract_source(pdf_path.name),
            "confidence": round(avg_confidence, 4),
            "text": full_text,
            "pages_data": [
                {
                    "page_number": i + 1,
                    "text": p.get("text", ""),
                    "confidence": p.get("confidence", 0),
                }
                for i, p in enumerate(pages_data)
            ],
            "quality_report": quality_report,
        }

    def _extract_source(self, filename: str) -> str:
        """Try to extract publication name from filename."""
        stem = Path(filename).stem
        # Remove common patterns like numbers, underscores
        parts = stem.replace("_", " ").replace("-", " ").split()
        # Return first non-numeric part as source name
        for part in parts:
            if not part.isdigit():
                return part
        return stem
