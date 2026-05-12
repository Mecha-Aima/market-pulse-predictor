from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.sentiment.analyzer import SentimentAnalyzer


def process_unprocessed_files() -> list[Path]:
    analyzer = SentimentAnalyzer()
    processed = _load_processed_files()
    output_paths: list[Path] = []
    for raw_path in _raw_json_files():
        relative_path = raw_path.relative_to(Path.cwd()).as_posix()
        if relative_path in processed:
            continue
        records = json.loads(raw_path.read_text())
        enriched = analyzer.batch_analyze(records)
        output_path = _write_processed_parquet(raw_path, enriched)
        output_paths.append(output_path)
        processed.add(relative_path)
    _save_processed_files(processed)
    return output_paths


def _raw_json_files() -> list[Path]:
    raw_root = Path.cwd() / "data" / "raw"
    return sorted(path for path in raw_root.rglob("*.json") if path.name != "seen_ids.json")


def _write_processed_parquet(raw_path: Path, records: list[dict]) -> Path:
    output_dir = Path.cwd() / "data" / "processed" / raw_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{raw_path.stem}.parquet"
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return output_path


def _processed_files_path() -> Path:
    return Path.cwd() / "data" / "processed" / "processed_files.json"


def _load_processed_files() -> set[str]:
    path = _processed_files_path()
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def _save_processed_files(processed: set[str]) -> None:
    path = _processed_files_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(processed), indent=2))


def main() -> None:
    process_unprocessed_files()


if __name__ == "__main__":
    main()
