"""
SQLite database for storing OCR results with full-text search support.
Uses FTS5 for Hebrew text search.
"""

import json
import sqlite3
from pathlib import Path


class Database:
    """SQLite database with FTS5 full-text search for OCR results."""

    def __init__(self, db_path: str = "data/output/database.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables and FTS index."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                source TEXT,
                year TEXT,
                pages INTEGER,
                confidence REAL,
                quality_score INTEGER,
                recommendation TEXT,
                date_processed TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT,
                confidence REAL,
                FOREIGN KEY (document_id) REFERENCES documents(id),
                UNIQUE(document_id, page_number)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                text,
                content='pages',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
                INSERT INTO pages_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
                INSERT INTO pages_fts(rowid, text) VALUES (new.id, new.text);
            END;
        """)
        self.conn.commit()

    def insert_document(self, result: dict) -> int:
        """
        Insert a processed document and its pages.

        Args:
            result: The full result dict from FileManager.build_result()

        Returns:
            Document ID
        """
        quality = result.get("quality_report", {})

        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO documents
                (filename, source, year, pages, confidence, quality_score,
                 recommendation, date_processed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["filename"],
            result.get("source", ""),
            result.get("year", ""),
            result["pages"],
            result["confidence"],
            quality.get("score", 0),
            quality.get("recommendation", ""),
            result["date_processed"],
            json.dumps({k: v for k, v in result.items()
                        if k not in ("text", "pages_data", "quality_report")},
                       ensure_ascii=False),
        ))

        doc_id = cursor.lastrowid

        # Insert pages
        for page in result.get("pages_data", []):
            self.conn.execute("""
                INSERT OR REPLACE INTO pages
                    (document_id, page_number, text, confidence)
                VALUES (?, ?, ?, ?)
            """, (
                doc_id,
                page["page_number"],
                page["text"],
                page.get("confidence", 0),
            ))

        self.conn.commit()
        return doc_id

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """
        Full-text search across all pages.

        Args:
            query: Search text (Hebrew)
            limit: Max results

        Returns:
            List of matching results with document info
        """
        rows = self.conn.execute("""
            SELECT
                d.filename, d.source, d.year, d.confidence,
                p.page_number,
                snippet(pages_fts, 0, '>>>', '<<<', '...', 30) as snippet
            FROM pages_fts
            JOIN pages p ON p.id = pages_fts.rowid
            JOIN documents d ON d.id = p.document_id
            WHERE pages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()

        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Get database statistics."""
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        pages = self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

        avg_conf = self.conn.execute(
            "SELECT AVG(confidence) FROM documents"
        ).fetchone()[0] or 0

        quality_dist = {}
        for row in self.conn.execute(
            "SELECT recommendation, COUNT(*) as cnt FROM documents GROUP BY recommendation"
        ):
            quality_dist[row[0]] = row[1]

        return {
            "total_documents": docs,
            "total_pages": pages,
            "avg_confidence": round(avg_conf, 4),
            "quality_distribution": quality_dist,
        }

    def get_problematic_documents(self, max_score: int = 75) -> list[dict]:
        """Get documents with quality score below threshold."""
        rows = self.conn.execute("""
            SELECT filename, source, confidence, quality_score, recommendation
            FROM documents
            WHERE quality_score < ?
            ORDER BY quality_score ASC
        """, (max_score,)).fetchall()

        return [dict(row) for row in rows]

    def close(self):
        self.conn.close()
