"""
Main OCR Pipeline - Adaptive processing for Hebrew historical newspapers.

Usage:
    python main.py                    # Process all PDFs in input dir
    python main.py --file path.pdf    # Process a single file
    python main.py --pilot 5          # Process first 5 files only
    python main.py --reprocess        # Force reprocess (ignore cache)
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from bidi.algorithm import get_display

from column_detector import ColumnDetector
from database import Database
from dictionary_checker import HebrewDictionaryChecker
from file_manager import FileManager
from google_vision import GoogleVisionOCR
from preprocessor import ImagePreprocessor
from quality_checker import QualityChecker
from post_processor import PostProcessor
from reporter import Reporter


class BidiFormatter(logging.Formatter):
    """Logging formatter that applies BiDi algorithm for correct Hebrew display in terminals."""

    def format(self, record):
        result = super().format(record)
        return get_display(result)


# Console handler with BiDi support, file handler without (file is already correct)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(BidiFormatter("%(asctime)s [%(levelname)s] %(message)s"))

_file_handler = logging.FileHandler("ocr_pipeline.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_console_handler, _file_handler],
)
logger = logging.getLogger(__name__)


class AdaptiveOCRPipeline:
    """Main pipeline that processes PDFs through OCR with adaptive quality handling."""

    def __init__(self, config_path: str = None):
        # Auto-detect project root (parent of 1_conversion/)
        self.project_root = Path(__file__).resolve().parent.parent

        if config_path is None:
            config_path = self.project_root / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Resolve all relative paths to project root
        self._resolve_paths()

        self.ocr = GoogleVisionOCR(
            credentials_path=self.config["google_vision"]["credentials_path"],
            cache_dir=self.config["paths"]["cache_dir"],
        )
        self.preprocessor = ImagePreprocessor(self.config["preprocessing"])
        self.file_manager = FileManager(self.config)
        self.quality_checker = QualityChecker(self.config["quality"])
        self.dictionary_checker = HebrewDictionaryChecker(
            self.config["paths"]["dictionary_path"]
        )
        self.database = Database(self.config["paths"]["database_path"])
        self.reporter = Reporter(self.config["paths"]["output_reports_dir"])
        self.column_detector = ColumnDetector()
        self.post_processor = PostProcessor(self.config.get("post_processing", {}))

        self.good_threshold = self.config["quality_assessment"]["good_threshold"]
        self.medium_threshold = self.config["quality_assessment"]["medium_threshold"]

    def _resolve_paths(self):
        """Resolve all relative paths in config to be relative to project root."""
        root = self.project_root

        # Resolve paths section
        for key, value in self.config["paths"].items():
            p = Path(value)
            if not p.is_absolute():
                self.config["paths"][key] = str(root / p)

        # Resolve credentials path
        cred = Path(self.config["google_vision"]["credentials_path"])
        if not cred.is_absolute():
            self.config["google_vision"]["credentials_path"] = str(root / cred)

    def process_single(self, pdf_path: Path, force: bool = False) -> dict:
        """
        Process a single PDF file through the adaptive pipeline.

        Args:
            pdf_path: Path to PDF file
            force: If True, reprocess even if already done

        Returns:
            Result dict with text, confidence, quality report
        """
        filename = pdf_path.name
        logger.info(f"קובץ | מעבד: {filename}")

        # Check if already processed
        if not force and self.file_manager.is_already_processed(filename):
            logger.info(f"קובץ | מדלג (כבר עובד): {filename}")
            return None

        start_time = time.time()

        try:
            # Step 1: Convert PDF to images
            images = self.file_manager.pdf_to_images(
                pdf_path, dpi=self.config["processing"]["dpi"]
            )
            logger.info(f"  המרה | {len(images)} עמודים → תמונות")

            # Step 2: Process each page
            pages_data = []
            for i, image_bytes in enumerate(images):
                page_result = self._process_page(image_bytes, i + 1)
                pages_data.append(page_result)

            # Step 3: Post-processing (remove interstitials, headers, add template)
            pages_data = self.post_processor.process_document(pages_data, filename)

            # Step 3b: Remove header markers (used to protect headers during post-processing)
            for page in pages_data:
                if page.get("text"):
                    page["text"] = (page["text"]
                        .replace("[כותרת] ", "")
                        .replace(" [/כותרת]", "")
                        .replace("[כותרת]", "")
                        .replace("[/כותרת]", ""))

            # Step 3c: Apply manual corrections from corrections file
            corrections = self.config.get("post_processing", {}).get("corrections", {})
            if corrections:
                for page in pages_data:
                    if page.get("text"):
                        for wrong, right in corrections.items():
                            page["text"] = page["text"].replace(wrong, right)

            # Step 3d: Flag ambiguous words (confusion swaps that produce valid alternatives)
            # Runs even without external dictionary — built-in abbreviations are checked too
            for page in pages_data:
                    if page.get("text"):
                        page["text"] = self.dictionary_checker.flag_ambiguous_words(
                            page["text"], page.get("words_data")
                        )

            logger.debug("  עיבוד-אחר | הושלם")

            # Step 4: Quality checks on combined text
            full_text = "\n".join(p["text"] for p in pages_data if p.get("text"))
            quality_report = self.quality_checker.check_all(full_text, len(images))

            # Step 5: Dictionary check (if dictionary available)
            if self.dictionary_checker.has_dictionary():
                dict_report = self.dictionary_checker.check_text(full_text)
                quality_report["dictionary"] = dict_report

            # Step 6: Build and save result
            result = self.file_manager.build_result(pdf_path, pages_data, quality_report)

            elapsed = time.time() - start_time
            result["processing_time_seconds"] = round(elapsed, 2)

            # Save outputs
            self.file_manager.save_txt(filename, result["text"])
            self.file_manager.save_json(filename, result)
            self.database.insert_document(result)
            self.reporter.save_file_report(result)

            score = quality_report.get("score", 0)
            logger.info(
                f"קובץ | הושלם: {filename} | "
                f"ביטחון={result['confidence']:.2%} | ציון={score}/100 | {elapsed:.1f}ש"
            )
            return result

        except Exception as e:
            logger.error(f"קובץ | נכשל: {filename} — {e}")
            return {
                "filename": filename,
                "error": str(e),
                "confidence": 0,
                "quality_report": {"score": 0, "recommendation": "failed"},
            }

    def _process_page(self, image_bytes: bytes, page_num: int) -> dict:
        """Process a single page image through OCR with adaptive pre-processing."""
        # Assess quality
        quality_level, quality_score = self.preprocessor.assess_quality(image_bytes)
        logger.debug(f"  עמוד {page_num} | איכות={quality_level} ({quality_score:.2f})")

        # Pre-process if needed
        if quality_level != "good":
            image_bytes = self.preprocessor.process(image_bytes, quality_level)
            logger.debug(f"  עמוד {page_num} | עיבוד מקדים ({quality_level})")

        # OCR
        ocr_result = self.ocr.detect_text(image_bytes)

        # Reorder text by columns (right-to-left for Hebrew)
        words = ocr_result.get("words", [])
        if words:
            text = self.column_detector.reorder_by_columns(words)
        else:
            text = ocr_result["text"]

        return {
            "page_number": page_num,
            "text": text,
            "confidence": ocr_result["confidence"],
            "quality_level": quality_level,
            "quality_score": quality_score,
            "languages": ocr_result.get("languages", []),
            "words_data": words,  # includes symbol_confidences for correction
        }

    def process_all(self, limit: int = 0, force: bool = False) -> list[dict]:
        """
        Process all PDFs in the input directory.

        Args:
            limit: Max files to process (0 = all)
            force: Reprocess already-processed files

        Returns:
            List of result dicts
        """
        pdfs = self.file_manager.get_pdf_list()
        if not pdfs:
            logger.warning("אצווה | לא נמצאו קבצי PDF בתיקיית הקלט")
            return []

        if limit > 0:
            pdfs = pdfs[:limit]

        logger.info(f"אצווה | מתחיל עיבוד של {len(pdfs)} קבצי PDF")
        start_time = time.time()

        results = []
        workers = self.config["processing"]["parallel_workers"]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.process_single, pdf, force): pdf
                for pdf in pdfs
            }

            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"קובץ | שגיאה לא צפויה: {pdf.name} — {e}")

        # Generate aggregate report
        successful = [r for r in results if "error" not in r]
        if successful:
            self.reporter.save_aggregate_report(successful)

        elapsed = time.time() - start_time
        logger.info(
            f"אצווה | הושלם: {len(successful)}/{len(pdfs)} הצליחו | "
            f"זמן כולל: {elapsed:.1f}ש"
        )

        # Print summary
        self._print_summary(results, elapsed)

        return results

    def _print_summary(self, results: list[dict], elapsed: float):
        """Print processing summary to console."""
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]

        _bprint = lambda s: print(get_display(s))

        _bprint("\n" + "=" * 50)
        _bprint(f"  עיבוד OCR הושלם")
        _bprint("=" * 50)
        _bprint(f"  הצליחו: {len(successful)}")
        _bprint(f"  נכשלו: {len(failed)}")
        _bprint(f"  זמן כולל: {elapsed:.1f}ש")

        if successful:
            avg_conf = sum(r["confidence"] for r in successful) / len(successful)
            avg_score = sum(
                r.get("quality_report", {}).get("score", 0) for r in successful
            ) / len(successful)
            _bprint(f"  ביטחון ממוצע: {avg_conf:.2%}")
            _bprint(f"  ציון איכות ממוצע: {avg_score:.0f}/100")

        if failed:
            _bprint("\n  קבצים שנכשלו:")
            for r in failed:
                _bprint(f"    - {r['filename']}: {r['error']}")

        _bprint("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Hebrew OCR Pipeline")
    parser.add_argument("--config", default=None, help="Path to config file (auto-detected if not set)")
    parser.add_argument("--file", help="Process a single PDF file")
    parser.add_argument("--pilot", type=int, default=0, help="Process only N files (pilot run)")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocess all files")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging (verbose)")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = AdaptiveOCRPipeline(config_path=args.config)

    if args.file:
        result = pipeline.process_single(Path(args.file), force=args.reprocess)
        if result:
            print(pipeline.reporter.file_report(result))
    else:
        pipeline.process_all(limit=args.pilot, force=args.reprocess)


if __name__ == "__main__":
    main()
