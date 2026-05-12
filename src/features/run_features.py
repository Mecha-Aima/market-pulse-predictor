from __future__ import annotations

from pathlib import Path

import yaml

from src.features.builder import TimeSeriesBuilder


def build_feature_artifacts() -> dict:
    params = yaml.safe_load((Path.cwd() / "params.yaml").read_text())
    builder = TimeSeriesBuilder(
        sequence_length=params["features"]["sequence_length"],
        train_ratio=params["features"]["train_ratio"],
        val_ratio=params["features"]["val_ratio"],
    )
    processed_paths = sorted((Path.cwd() / "data" / "processed").rglob("*.parquet"))
    return builder.build_and_save(processed_paths, Path.cwd() / "data" / "features")


def main() -> None:
    build_feature_artifacts()


if __name__ == "__main__":
    main()
