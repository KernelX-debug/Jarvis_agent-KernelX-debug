from __future__ import annotations

import csv
import io
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from jarvis_agent.models import IngestReport

from .store import KnowledgeBase

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".log",
    ".html",
    ".htm",
}
DATA_EXTENSIONS = {".json", ".jsonl", ".csv"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DATA_EXTENSIONS | {".pdf"}
IGNORED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".jarvis"}


class _HttpOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPError(newurl, code, "Redireccion no permitida", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_URL_OPENER = build_opener(_HttpOnlyRedirectHandler())


class ResourceIngestor:
    def __init__(self, knowledge: KnowledgeBase, *, max_file_mb: int = 20) -> None:
        self.knowledge = knowledge
        self.max_bytes = max_file_mb * 1024 * 1024

    def ingest(self, source: str | Path) -> IngestReport:
        raw = str(source)
        if raw.startswith(("http://", "https://")):
            return self.ingest_url(raw)
        path = Path(source).expanduser()
        if path.is_dir():
            return self.ingest_directory(path)
        return self.ingest_file(path)

    def ingest_directory(self, path: Path) -> IngestReport:
        resolved = path.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"No existe el directorio: {path}")
        totals = IngestReport(str(resolved), files=0, chunks=0, skipped=0)
        files = sorted(
            item
            for item in resolved.rglob("*")
            if item.is_file()
            and not item.is_symlink()
            and not any(part in IGNORED_DIRECTORIES for part in item.parts)
        )
        for file_path in files:
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                totals = _add_reports(totals, IngestReport(str(file_path), 0, 0, 1))
                continue
            try:
                totals = _add_reports(totals, self.ingest_file(file_path))
            except (OSError, UnicodeError, ValueError, RuntimeError):
                totals = _add_reports(totals, IngestReport(str(file_path), 0, 0, 1))
        return totals

    def ingest_file(
        self,
        path: Path,
        *,
        source_id: str | None = None,
        extra_metadata: dict[str, object] | None = None,
        allowed_symlink_root: Path | None = None,
    ) -> IngestReport:
        if path.is_symlink():
            resolved_target = path.resolve(strict=True)
            allowed_root = allowed_symlink_root.resolve() if allowed_symlink_root else None
            if allowed_root is None or not resolved_target.is_relative_to(allowed_root):
                raise ValueError(f"No se ingiere el enlace simbolico fuera de la fuente: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"La fuente no es un archivo regular: {path}")
        source_suffix = path.suffix.lower()
        size = resolved.stat().st_size
        if size > self.max_bytes:
            raise ValueError(
                f"{resolved.name} supera el limite de {self.max_bytes // (1024 * 1024)} MB."
            )
        text = self._read_file(resolved, suffix=source_suffix)
        metadata: dict[str, object] = {
            "kind": "file",
            "size": size,
            "suffix": source_suffix,
        }
        metadata.update(extra_metadata or {})
        return self.knowledge.ingest_text(source_id or str(resolved), text, metadata=metadata)

    def ingest_url(self, url: str) -> IngestReport:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Solo se permiten URLs http o https validas.")
        request = Request(url, headers={"User-Agent": "Jarvis-Agent-Local/0.1"})
        with _URL_OPENER.open(request, timeout=30) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > self.max_bytes:
                raise ValueError("La fuente remota supera el limite configurado.")
            payload = response.read(self.max_bytes + 1)
            if len(payload) > self.max_bytes:
                raise ValueError("La fuente remota supera el limite configurado.")
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"

        if content_type == "application/pdf" or parsed.path.lower().endswith(".pdf"):
            text = _read_pdf(io.BytesIO(payload))
        else:
            text = payload.decode(charset, errors="replace")
            if content_type == "text/html" or "<html" in text[:500].lower():
                text = _html_to_text(text)
        return self.knowledge.ingest_text(
            url,
            text,
            metadata={"kind": "url", "content_type": content_type, "size": len(payload)},
        )

    def ingest_huggingface(
        self,
        repo_id: str,
        *,
        repo_type: str = "dataset",
        revision: str | None = None,
    ) -> IngestReport:
        if repo_type not in {"dataset", "model", "space"}:
            raise ValueError("repo_type debe ser dataset, model o space.")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError('Instala el conector: pip install -e ".[hub]"') from exc

        local_path = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                allow_patterns=[
                    "*.txt",
                    "*.md",
                    "*.rst",
                    "*.json",
                    "*.jsonl",
                    "*.csv",
                    "*.pdf",
                    "*.html",
                    "*.htm",
                    "LICENSE*",
                    "README*",
                ],
                token=None,
            )
        )
        source_root = f"hf://{repo_type}/{repo_id}@{revision or 'latest'}"
        total = IngestReport(source_root, 0, 0, 0)
        cache_repo_root = local_path.parent.parent.resolve()
        files = sorted(
            item
            for item in local_path.rglob("*")
            if item.is_file()
            and item.suffix.lower() in SUPPORTED_EXTENSIONS
            and item.resolve().is_relative_to(cache_repo_root)
        )
        for file_path in files:
            relative = file_path.relative_to(local_path).as_posix()
            try:
                report = self.ingest_file(
                    file_path,
                    source_id=f"{source_root}/{relative}",
                    allowed_symlink_root=cache_repo_root,
                    extra_metadata={
                        "kind": "huggingface",
                        "repo_id": repo_id,
                        "repo_type": repo_type,
                        "revision": revision or "latest",
                        "path": relative,
                    },
                )
                total = _add_reports(total, report)
            except (OSError, UnicodeError, ValueError, RuntimeError):
                total = _add_reports(total, IngestReport(str(file_path), 0, 0, 1))
        return total

    def _read_file(self, path: Path, *, suffix: str | None = None) -> str:
        suffix = suffix or path.suffix.lower()
        if suffix == ".pdf":
            return _read_pdf(path)
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".json":
            return _json_to_text(json.loads(raw))
        if suffix == ".jsonl":
            values = []
            for line in raw.splitlines():
                if line.strip():
                    values.append(_json_to_text(json.loads(line)))
            return "\n\n".join(values)
        if suffix == ".csv":
            rows = csv.reader(io.StringIO(raw))
            return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        if suffix in {".html", ".htm"}:
            return _html_to_text(raw)
        return raw


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


def _html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return "\n".join(parser.parts)


def _json_to_text(value: object, prefix: str = "") -> str:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            lines.append(_json_to_text(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            lines.append(_json_to_text(item, f"{prefix}[{index}]"))
    elif value is not None:
        lines.append(f"{prefix}: {value}" if prefix else str(value))
    return "\n".join(line for line in lines if line)


def _read_pdf(source: object) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError('Para PDF instala: pip install -e ".[pdf]"') from exc
    reader = PdfReader(source)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _add_reports(left: IngestReport, right: IngestReport) -> IngestReport:
    return IngestReport(
        left.source,
        left.files + right.files,
        left.chunks + right.chunks,
        left.skipped + right.skipped,
    )
