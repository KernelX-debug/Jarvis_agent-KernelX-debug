from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest

from jarvis_agent.knowledge.ingest import ResourceIngestor
from jarvis_agent.knowledge.store import KnowledgeBase


@pytest.fixture
def ingestor(tmp_path) -> ResourceIngestor:
    knowledge = KnowledgeBase(tmp_path / "db" / "knowledge.sqlite3")
    return ResourceIngestor(knowledge)


def test_ingest_json_flattens_nested_values(ingestor: ResourceIngestor, tmp_path) -> None:
    source = tmp_path / "profile.json"
    source.write_text(
        json.dumps(
            {
                "persona": {"nombre": "Ada Lovelace"},
                "habilidades": ["matemática", "programación"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = ingestor.ingest_file(source)
    hits = ingestor.knowledge.search("Lovelace")

    assert report.files == 1
    assert report.chunks == 1
    assert len(hits) == 1
    assert "persona.nombre: Ada Lovelace" in hits[0].content
    assert "habilidades[1]: programación" in hits[0].content


def test_ingest_csv_preserves_rows_as_searchable_text(
    ingestor: ResourceIngestor,
    tmp_path,
) -> None:
    source = tmp_path / "inventory.csv"
    source.write_text(
        "nombre,categoria\nQuena,instrumento andino\nCacao,alimento\n",
        encoding="utf-8",
    )

    report = ingestor.ingest(source)
    hits = ingestor.knowledge.search("Quena")

    assert report.files == 1
    assert hits
    assert "Quena | instrumento andino" in hits[0].content


def test_ingest_directory_counts_supported_and_skips_unsupported(
    ingestor: ResourceIngestor,
    tmp_path,
) -> None:
    resources = tmp_path / "resources"
    (resources / "nested").mkdir(parents=True)
    (resources / ".git").mkdir()
    (resources / "notes.txt").write_text("alpaca", encoding="utf-8")
    (resources / "nested" / "manual.md").write_text("vicuña", encoding="utf-8")
    (resources / "records.json").write_text('{"animal": "llama"}', encoding="utf-8")
    (resources / "table.csv").write_text("animal\ncondor\n", encoding="utf-8")
    (resources / "image.bin").write_bytes(b"not supported")
    (resources / ".git" / "ignored.md").write_text("ignorado", encoding="utf-8")

    report = ingestor.ingest_directory(resources)

    assert report.files == 4
    assert report.chunks == 4
    assert report.skipped == 1
    assert ingestor.knowledge.stats()["sources"] == 4
    assert ingestor.knowledge.search("ignorado") == []


def test_ingest_directory_skips_invalid_supported_file(
    ingestor: ResourceIngestor,
    tmp_path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "valid.txt").write_text("válido", encoding="utf-8")
    (resources / "broken.json").write_text("{not-json", encoding="utf-8")

    report = ingestor.ingest_directory(resources)

    assert report.files == 1
    assert report.skipped == 1


def test_invalid_json_file_raises_value_error(ingestor: ResourceIngestor, tmp_path) -> None:
    source = tmp_path / "broken.json"
    source.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError):
        ingestor.ingest_file(source)


def test_missing_file_and_directory_are_rejected(
    ingestor: ResourceIngestor,
    tmp_path,
) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ingestor.ingest_file(missing / "file.txt")
    with pytest.raises(FileNotFoundError):
        ingestor.ingest_directory(missing)


def test_file_size_limit_is_checked_before_reading(tmp_path) -> None:
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    ingestor = ResourceIngestor(knowledge, max_file_mb=1)
    source = tmp_path / "large.txt"
    source.write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(ValueError, match="supera el limite"):
        ingestor.ingest_file(source)


def _install_fake_huggingface_hub(monkeypatch, snapshot_path):
    calls: list[dict[str, object]] = []
    module = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(snapshot_path)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_huggingface_ingest_keeps_logical_provenance(
    monkeypatch,
    tmp_path,
) -> None:
    repository = tmp_path / "cache" / "datasets--acme--manual"
    snapshot = repository / "snapshots" / "abc123"
    (snapshot / "nested").mkdir(parents=True)
    (snapshot / "README.md").write_text("Manual sobre quipus.", encoding="utf-8")
    (snapshot / "nested" / "facts.csv").write_text(
        "tema,valor\nastronomía,chankillo\n",
        encoding="utf-8",
    )
    (snapshot / "ignored.bin").write_bytes(b"not indexed")
    calls = _install_fake_huggingface_hub(monkeypatch, snapshot)
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    ingestor = ResourceIngestor(knowledge)

    report = ingestor.ingest_huggingface(
        "acme/manual",
        repo_type="dataset",
        revision="abc123",
    )

    source_root = "hf://dataset/acme/manual@abc123"
    assert report.source == source_root
    assert report.files == 2
    assert report.chunks == 2
    assert report.skipped == 0
    assert knowledge.search("quipus")[0].source == f"{source_root}/README.md"
    assert knowledge.search("chankillo")[0].source == f"{source_root}/nested/facts.csv"
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "acme/manual"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "abc123"
    assert calls[0]["token"] is None
    patterns = calls[0]["allow_patterns"]
    assert isinstance(patterns, list)
    assert "*.md" in patterns
    assert "*.bin" not in patterns

    with sqlite3.connect(knowledge.path) as connection:
        raw_metadata = connection.execute(
            "SELECT metadata_json FROM sources WHERE source = ?",
            (f"{source_root}/README.md",),
        ).fetchone()[0]
    metadata = json.loads(raw_metadata)
    assert metadata == {
        "kind": "huggingface",
        "path": "README.md",
        "repo_id": "acme/manual",
        "repo_type": "dataset",
        "revision": "abc123",
        "size": len(b"Manual sobre quipus."),
        "suffix": ".md",
    }


def test_huggingface_ingest_allows_internal_symlink_and_excludes_escape(
    monkeypatch,
    tmp_path,
) -> None:
    repository = tmp_path / "cache" / "models--acme--manual"
    snapshot = repository / "snapshots" / "commit1"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    internal_blob = blobs / "internal-content"
    internal_blob.write_text("contenido interno verificable", encoding="utf-8")
    json_blob = blobs / "json-content"
    json_blob.write_text('{"mensajero": "chaski"}', encoding="utf-8")
    external_file = tmp_path / "outside.md"
    external_file.write_text("secreto externo", encoding="utf-8")
    internal_link = snapshot / "README.md"
    json_link = snapshot / "facts.json"
    escape_link = snapshot / "escape.md"
    internal_link.write_text("marcador de enlace interno", encoding="utf-8")
    json_link.write_text("marcador json", encoding="utf-8")
    escape_link.write_text("marcador de enlace externo", encoding="utf-8")

    real_is_symlink = Path.is_symlink
    real_resolve = Path.resolve
    simulated_links = {
        internal_link: internal_blob,
        json_link: json_blob,
        escape_link: external_file,
    }

    def simulated_is_symlink(path: Path) -> bool:
        return path in simulated_links or real_is_symlink(path)

    def simulated_resolve(path: Path, strict: bool = False) -> Path:
        target = simulated_links.get(path, path)
        return real_resolve(target, strict=strict)

    monkeypatch.setattr(Path, "is_symlink", simulated_is_symlink)
    monkeypatch.setattr(Path, "resolve", simulated_resolve)
    _install_fake_huggingface_hub(monkeypatch, snapshot)
    knowledge = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    ingestor = ResourceIngestor(knowledge)

    report = ingestor.ingest_huggingface(
        "acme/manual",
        repo_type="model",
        revision="commit1",
    )

    assert report.files == 2
    assert report.chunks == 2
    hits = knowledge.search("verificable")
    assert [hit.source for hit in hits] == ["hf://model/acme/manual@commit1/README.md"]
    json_hits = knowledge.search("chaski")
    assert [hit.source for hit in json_hits] == ["hf://model/acme/manual@commit1/facts.json"]
    assert "mensajero: chaski" in json_hits[0].content
    assert knowledge.search("secreto externo") == []
