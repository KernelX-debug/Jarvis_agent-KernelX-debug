from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from jarvis_agent.models import IngestReport, SearchHit

_STOPWORDS = {
    "a",
    "al",
    "algo",
    "como",
    "con",
    "cual",
    "de",
    "del",
    "donde",
    "el",
    "ella",
    "en",
    "es",
    "esta",
    "este",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "su",
    "un",
    "una",
    "y",
}


class KnowledgeBase:
    """Indice local SQLite/FTS5 sin servicios externos ni embeddings obligatorios."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> bool:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY (source, chunk_index),
                    FOREIGN KEY (source) REFERENCES sources(source) ON DELETE CASCADE
                )
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        content,
                        source UNINDEXED,
                        chunk_index UNINDEXED,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
            except sqlite3.OperationalError:
                return False
        return True

    def ingest_text(
        self,
        source: str,
        text: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> IngestReport:
        normalized = text.replace("\x00", "").strip()
        if not normalized:
            return IngestReport(source, files=0, chunks=0, skipped=1)
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        chunks = _split_chunks(normalized)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT fingerprint FROM sources WHERE source = ?", (source,)
            ).fetchone()
            if existing and existing["fingerprint"] == fingerprint:
                return IngestReport(source, files=0, chunks=0, skipped=1)

            connection.execute("PRAGMA foreign_keys = ON")
            if self._fts_enabled:
                connection.execute("DELETE FROM chunks_fts WHERE source = ?", (source,))
            connection.execute("DELETE FROM chunks WHERE source = ?", (source,))
            connection.execute("DELETE FROM sources WHERE source = ?", (source,))
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (
                    source,
                    fingerprint,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                    len(chunks),
                ),
            )
            for index, chunk in enumerate(chunks):
                connection.execute(
                    "INSERT INTO chunks(source, chunk_index, content) VALUES (?, ?, ?)",
                    (source, index, chunk),
                )
                if self._fts_enabled:
                    connection.execute(
                        "INSERT INTO chunks_fts(content, source, chunk_index) VALUES (?, ?, ?)",
                        (chunk, source, index),
                    )
        return IngestReport(source, files=1, chunks=len(chunks))

    def search(self, query: str, *, limit: int = 4) -> list[SearchHit]:
        tokens = _query_tokens(query)
        if not tokens or limit < 1:
            return []
        if self._fts_enabled:
            fts_query = " OR ".join(f'"{token}"' for token in tokens)
            try:
                with self._connect() as connection:
                    rows = connection.execute(
                        """
                        SELECT source, chunk_index, content, bm25(chunks_fts) AS rank
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, limit),
                    ).fetchall()
                if rows:
                    return [
                        SearchHit(
                            source=row["source"],
                            content=row["content"],
                            score=1.0 / (1.0 + abs(float(row["rank"]))),
                            chunk_index=int(row["chunk_index"]),
                        )
                        for row in rows
                    ]
            except sqlite3.OperationalError:
                pass
        return self._fallback_search(tokens, limit)

    def stats(self) -> dict[str, int | bool]:
        with self._connect() as connection:
            sources = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"sources": sources, "chunks": chunks, "fts5": self._fts_enabled}

    def _fallback_search(self, tokens: list[str], limit: int) -> list[SearchHit]:
        with self._connect() as connection:
            rows = connection.execute("SELECT source, chunk_index, content FROM chunks").fetchall()
        scored: list[SearchHit] = []
        for row in rows:
            lowered = row["content"].lower()
            count = sum(lowered.count(token) for token in tokens)
            if count:
                scored.append(
                    SearchHit(
                        row["source"],
                        row["content"],
                        float(count),
                        int(row["chunk_index"]),
                    )
                )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[\wáéíóúüñ]{2,}", query.lower(), flags=re.UNICODE)
    unique: list[str] = []
    for token in tokens:
        if token not in _STOPWORDS and token not in unique:
            unique.append(token)
    return unique[:12]


def _split_chunks(text: str, *, target: int = 1_200, overlap: int = 160) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = _split_long_text(paragraph, target)
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > target:
                chunks.append(current)
                prefix = current[-overlap:].lstrip()
                current = f"{prefix}\n{piece}".strip()
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks or [text[:target]]


def _split_long_text(text: str, target: int) -> list[str]:
    if len(text) <= target:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > target:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[index : index + target] for index in range(0, len(sentence), target)
            )
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces
