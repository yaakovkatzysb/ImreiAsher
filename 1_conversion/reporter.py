"""
Report generator for OCR processing results.
Generates per-file reports and aggregate summary reports.
"""

import json
from datetime import datetime
from pathlib import Path


class Reporter:
    """Generate quality reports for OCR results."""

    def __init__(self, reports_dir: str = "data/output/reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def file_report(self, result: dict) -> str:
        """
        Generate a text report for a single processed file.

        Args:
            result: The full result dict from processing

        Returns:
            Formatted report string
        """
        qr = result.get("quality_report", {})
        l1 = qr.get("level1", {})
        l2 = qr.get("level2", {})

        lines = [
            "=" * 50,
            f"  {result['filename']}",
            "=" * 50,
            "",
            f"  Pages: {result['pages']}",
            f"  Confidence: {result['confidence']:.2%}",
            f"  Score: {qr.get('score', 'N/A')}/100",
            f"  Recommendation: {qr.get('recommendation', 'N/A')}",
            "",
            "--- Level 1: Basic Checks ---",
        ]

        # Hebrew ratio
        hr = l1.get("hebrew_ratio", {})
        status = "PASS" if hr.get("passed") else "FAIL"
        lines.append(f"  [{status}] Hebrew ratio: {hr.get('ratio', 0):.1%}")

        # Weird chars
        wc = l1.get("weird_chars", {})
        status = "PASS" if wc.get("passed") else "FAIL"
        lines.append(f"  [{status}] Weird characters: {wc.get('count', 0)}")

        # Text length
        tl = l1.get("text_length", {})
        status = "PASS" if tl.get("passed") else "FAIL"
        lines.append(f"  [{status}] Avg chars/page: {tl.get('avg_per_page', 0)}")

        # Spacing
        sp = l1.get("spacing", {})
        status = "PASS" if sp.get("passed") else "FAIL"
        lines.append(f"  [{status}] Spacing errors: {sp.get('error_score', 0)}")

        lines.append("")
        lines.append("--- Level 2: Linguistic Checks ---")

        # Final letters
        fl = l2.get("final_letters", {})
        status = "PASS" if fl.get("passed") else "FAIL"
        lines.append(f"  [{status}] Final letter errors: {fl.get('errors', 0)}")

        # Nikud
        nk = l2.get("nikud", {})
        status = "PASS" if nk.get("passed") else "FAIL"
        lines.append(f"  [{status}] Double nikud: {nk.get('double_nikud', 0)}")

        # Grammar
        gr = l2.get("grammar", {})
        status = "PASS" if gr.get("passed") else "FAIL"
        lines.append(f"  [{status}] Single char errors: {gr.get('single_char_errors', 0)}")
        lines.append(f"  [{status}] Repeated chars: {gr.get('repeated_chars', 0)}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)

    def aggregate_report(self, results: list[dict]) -> str:
        """
        Generate an aggregate report for all processed files.

        Args:
            results: List of result dicts

        Returns:
            Formatted aggregate report string
        """
        if not results:
            return "No files processed."

        total = len(results)
        confidences = [r["confidence"] for r in results]
        scores = [r.get("quality_report", {}).get("score", 0) for r in results]

        # Categorize by confidence
        excellent = sum(1 for c in confidences if c >= 0.90)
        good = sum(1 for c in confidences if 0.80 <= c < 0.90)
        problematic = sum(1 for c in confidences if c < 0.80)

        # Categorize by recommendation
        recommendations = {}
        for r in results:
            rec = r.get("quality_report", {}).get("recommendation", "unknown")
            recommendations[rec] = recommendations.get(rec, 0) + 1

        lines = [
            "=" * 50,
            f"  OCR Processing Report - {total} files",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 50,
            "",
            "--- Statistics ---",
            f"  Total files: {total}",
            f"  Avg confidence: {sum(confidences)/total:.2%}",
            f"  Avg quality score: {sum(scores)/total:.0f}/100",
            "",
            "--- Confidence Distribution ---",
            f"  Excellent (>90%): {excellent} ({excellent/total:.0%})",
            f"  Good (80-90%): {good} ({good/total:.0%})",
            f"  Problematic (<80%): {problematic} ({problematic/total:.0%})",
            "",
            "--- Recommendations ---",
        ]

        for rec, count in sorted(recommendations.items()):
            lines.append(f"  {rec}: {count} ({count/total:.0%})")

        # List problematic files
        problem_files = [
            r for r in results
            if r.get("quality_report", {}).get("score", 100) < 75
        ]
        if problem_files:
            lines.append("")
            lines.append("--- Files Needing Attention ---")
            for r in sorted(problem_files, key=lambda x: x.get("quality_report", {}).get("score", 0)):
                score = r.get("quality_report", {}).get("score", 0)
                lines.append(f"  {r['filename']}: score={score}, conf={r['confidence']:.2%}")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def save_file_report(self, result: dict):
        """Save individual file report."""
        report = self.file_report(result)
        stem = Path(result["filename"]).stem
        output = self.reports_dir / f"{stem}_report.txt"
        output.write_text(report, encoding="utf-8")
        return output

    def save_aggregate_report(self, results: list[dict]):
        """Save aggregate report."""
        report = self.aggregate_report(results)
        output = self.reports_dir / "aggregate_report.txt"
        output.write_text(report, encoding="utf-8")
        return output
