from __future__ import annotations

from jarvis_agent.knowledge.store import KnowledgeBase


def test_empty_text_is_skipped(tmp_path) -> None:
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")

    report = knowledge.ingest_text("empty.txt", " \n\x00 ")

    assert report.files == 0
    assert report.chunks == 0
    assert report.skipped == 1
    assert knowledge.stats()["sources"] == 0


def test_ingest_deduplicates_unchanged_source(tmp_path) -> None:
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")

    first = knowledge.ingest_text("manual.md", "El ornitorrinco usa un sombrero azul.")
    duplicate = knowledge.ingest_text("manual.md", "El ornitorrinco usa un sombrero azul.")

    assert first.files == 1
    assert first.chunks == 1
    assert duplicate.files == 0
    assert duplicate.chunks == 0
    assert duplicate.skipped == 1
    assert knowledge.stats()["sources"] == 1
    assert knowledge.stats()["chunks"] == 1


def test_reingest_changed_source_replaces_old_chunks(tmp_path) -> None:
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    knowledge.ingest_text("manual.md", "Contenido antiguo sobre ornitorrincos.")

    report = knowledge.ingest_text("manual.md", "Contenido nuevo sobre telescopios.")

    assert report.files == 1
    assert knowledge.search("ornitorrincos") == []
    hits = knowledge.search("telescopios")
    assert len(hits) == 1
    assert hits[0].source == "manual.md"
    assert "telescopios" in hits[0].content


def test_search_returns_relevant_sources_and_respects_limit(tmp_path) -> None:
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    knowledge.ingest_text("uno.md", "Nebulosa violeta y estrellas lejanas.")
    knowledge.ingest_text("dos.md", "Otra nebulosa contiene polvo cósmico.")
    knowledge.ingest_text("tres.md", "Un huerto contiene tomates.")

    hits = knowledge.search("nebulosa", limit=1)

    assert len(hits) == 1
    assert hits[0].source in {"uno.md", "dos.md"}
    assert hits[0].chunk_index == 0
    assert hits[0].score > 0


def test_search_ignores_empty_stopword_only_and_invalid_limit(tmp_path) -> None:
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    knowledge.ingest_text("manual.md", "Texto de ejemplo")

    assert knowledge.search("") == []
    assert knowledge.search("de la y") == []
    assert knowledge.search("ejemplo", limit=0) == []


def test_long_content_is_split_into_searchable_chunks(tmp_path) -> None:
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    long_text = "primero " * 180 + "\n\n" + "xilófono " * 180

    report = knowledge.ingest_text("largo.md", long_text)

    assert report.chunks >= 2
    hits = knowledge.search("xilófono")
    assert hits
    assert all(hit.source == "largo.md" for hit in hits)
