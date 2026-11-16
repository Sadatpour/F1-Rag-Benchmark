from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from f1_rag.sources import JOLPICA_API_BASE, jolpica_source_url, wikipedia_source_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"
SOURCE_DATA = PROJECT_ROOT / "data"


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {error}") from error
            if isinstance(row, dict):
                yield row


def _flatten(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"url", "xmlns"}:
                continue
            label = f"{prefix}.{key}" if prefix else key
            yield from _flatten(child, label)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten(child, prefix)
    elif value is not None and str(value).strip():
        yield f"{prefix}: {value}"


def _wiki_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read Wikipedia source {path}: {error}") from error
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of Wikipedia pages in {path}")

    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("missing"):
            continue
        title = str(row.get("title") or row.get("query_title") or f"Page {index + 1}")
        source_title = title.replace(" ", "_")
        text_parts = [f"Title: {title}"]
        extract = row.get("extract")
        if isinstance(extract, str) and extract.strip():
            text_parts.append(extract.strip())
        paragraphs = row.get("paragraphs")
        if isinstance(paragraphs, list):
            text_parts.extend(str(item).strip() for item in paragraphs if str(item).strip())
        text = "\n".join(text_parts).strip()
        if len(text) < 20:
            continue
        yield {
            "id": f"wiki:{path.stem}:{index}",
            "text": text,
            "source": f"Wikipedia:{source_title}",
            "source_file": path.name,
            "title": title,
            "source_url": wikipedia_source_url(source_title),
        }


def _jolpica_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read Jolpica source {path}: {error}") from error
    root = payload.get("MRData", payload) if isinstance(payload, dict) else payload
    endpoint = re.sub(r"^\d{4}_", "", path.stem)
    endpoint = re.sub(r"_\d+$", "", endpoint)
    initial_season = path.stem[:4] if path.stem[:4].isdigit() else ""

    def walk(value: Any, location: str, season: str, round_number: str) -> Iterator[dict[str, Any]]:
        if isinstance(value, dict):
            season = str(value.get("season") or season)
            round_number = str(value.get("round") or round_number)
            is_list_record = location.rsplit(".", 1)[-1].isdigit()
            text = "\n".join(_flatten(value)).strip()
            if is_list_record and text and len(text) >= 20:
                if endpoint in {"driverStandings", "constructorStandings"} and season:
                    source = f"Jolpica:{season}/{endpoint}"
                elif endpoint == "circuits" and value.get("circuitId"):
                    source = f"Jolpica:circuits/{value['circuitId']}"
                elif season and round_number:
                    source = f"Jolpica:{season}/{round_number}/{endpoint}"
                else:
                    source_parts = ["Jolpica"]
                    if season:
                        source_parts.append(season)
                    source_parts.append(endpoint)
                    source = ":".join(source_parts)
                yield {
                    "id": f"jolpica:{path.stem}:{location}",
                    "text": text,
                    "source": source,
                    "source_file": path.name,
                    "season": season,
                    "round": round_number,
                    "endpoint": endpoint,
                    "source_url": jolpica_source_url(source),
                }
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    yield from walk(child, f"{location}.{key}", season, round_number)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    yield from walk(child, f"{location}.{index}", season, round_number)

    yield from walk(root, "MRData", initial_season, "")


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_corpus_path(dataset: str, knowledge_base: Path = KNOWLEDGE_BASE) -> Path:
    if dataset not in {"wikipedia", "jolpica"}:
        raise ValueError(f"Unknown corpus: {dataset}")
    path = knowledge_base / "corpora" / f"{dataset}.jsonl"
    if path.is_file():
        return path
    compressed_path = path.with_suffix(path.suffix + ".gz")
    return compressed_path if compressed_path.is_file() else path


def build_knowledge_base(source_root: Path = SOURCE_DATA, output_root: Path = KNOWLEDGE_BASE) -> dict[str, int]:
    """Create self-contained source corpora and an evaluation set from the original dataset."""
    wiki_dir = source_root / "raw" / "wikipedia"
    jolpica_dir = source_root / "raw" / "jolpica"
    qa_path = source_root / "final" / "qa.jsonl"
    if not wiki_dir.is_dir() or not jolpica_dir.is_dir() or not qa_path.is_file():
        raise FileNotFoundError(
            "Expected source folders data/raw/wikipedia, data/raw/jolpica, and data/final/qa.jsonl."
        )

    wiki_files = sorted(wiki_dir.rglob("*.json"))
    jolpica_files = sorted(jolpica_dir.rglob("*.json"))
    if not wiki_files or not jolpica_files:
        raise FileNotFoundError("The Wikipedia or Jolpica source folder contains no JSON files.")

    def jolpica_documents() -> Iterator[dict[str, Any]]:
        for path in jolpica_files:
            yield from _jolpica_records(path)
        champions = source_root / "processed" / "champions.json"
        if champions.is_file():
            try:
                rows = json.loads(champions.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Could not read champions source {champions}: {error}") from error
            if isinstance(rows, list):
                for index, row in enumerate(rows):
                    if isinstance(row, dict):
                        text = "\n".join(_flatten(row)).strip()
                        if text:
                            original_source = str(row.get("source", ""))
                            source_url = (
                                jolpica_source_url(original_source)
                                if original_source.startswith("Jolpica:")
                                else JOLPICA_API_BASE + "/"
                            )
                            yield {
                                "id": f"champions:{index}",
                                "text": text,
                                "source": "synthetic:champions.json",
                                "source_file": champions.name,
                                "source_url": source_url,
                            }

    counts = {
        "wikipedia": _write_jsonl(
            output_root / "corpora" / "wikipedia.jsonl",
            (record for path in wiki_files for record in _wiki_records(path)),
        ),
        "jolpica": _write_jsonl(
            output_root / "corpora" / "jolpica.jsonl",
            jolpica_documents(),
        ),
        "evaluation": _write_jsonl(output_root / "evaluation" / "qa.jsonl", _read_jsonl(qa_path)),
    }
    if not counts["wikipedia"] or not counts["jolpica"] or not counts["evaluation"]:
        raise ValueError(f"Knowledge-base extraction produced an empty dataset: {counts}")

    manifest = {
        "source": "F1-Dataset data/raw and data/final/qa.jsonl",
        "documents": counts,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return counts


def iter_documents(dataset: str, knowledge_base: Path = KNOWLEDGE_BASE) -> Iterator[dict[str, Any]]:
    path = resolve_corpus_path(dataset, knowledge_base)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path} (or its .gz archive). Restore the original data sources before running "
            "`python -m f1_rag.ingest` to rebuild the knowledge base."
        )
    yield from _read_jsonl(path)


def load_documents(dataset: str, knowledge_base: Path = KNOWLEDGE_BASE) -> list[dict[str, Any]]:
    return list(iter_documents(dataset, knowledge_base))


if __name__ == "__main__":
    result = build_knowledge_base()
    print(f"Knowledge base created in {KNOWLEDGE_BASE}: {result}")
