"""Load and validate the 160 main ConceptARC tasks.

This module builds a deterministic pair-level manifest. Each row represents one
input-output transformation and preserves concept, task, split, and pair identity.
The raw grids are stored as compact JSON strings so the manifest is self-contained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_CONCEPTS = 16
EXPECTED_TASKS_PER_CONCEPT = 10
EXPECTED_TASKS = EXPECTED_CONCEPTS * EXPECTED_TASKS_PER_CONCEPT
EXPECTED_PAIRS = 907
VALID_COLORS = set(range(10))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically and compactly."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def validate_grid(grid: Any, *, context: str) -> list[list[int]]:
    """Validate an ARC grid and return it with a precise list-of-lists type."""
    if not isinstance(grid, list) or not grid:
        raise ValueError(f"{context}: grid must be a non-empty list of rows")

    if not all(isinstance(row, list) and row for row in grid):
        raise ValueError(f"{context}: every row must be a non-empty list")

    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError(f"{context}: grid rows have inconsistent widths")

    typed_grid: list[list[int]] = []
    for row_index, row in enumerate(grid):
        typed_row: list[int] = []
        for col_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{context}: cell ({row_index}, {col_index}) is not an integer"
                )
            if value not in VALID_COLORS:
                raise ValueError(
                    f"{context}: cell ({row_index}, {col_index}) has invalid "
                    f"ARC color {value}; expected 0-9"
                )
            typed_row.append(value)
        typed_grid.append(typed_row)

    return typed_grid


def discover_task_files(corpus_dir: Path) -> list[Path]:
    """Find all JSON task files under the main ConceptARC corpus."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"ConceptARC corpus directory not found: {corpus_dir}\n"
            "Clone the official repository into data/ConceptARC first."
        )

    task_files = sorted(path for path in corpus_dir.rglob("*.json") if path.is_file())
    if not task_files:
        raise FileNotFoundError(f"No JSON tasks found under {corpus_dir}")
    return task_files


def infer_concept_and_task(task_path: Path, corpus_dir: Path) -> tuple[str, str]:
    """Infer stable concept and task identifiers from a task's relative path."""
    relative = task_path.relative_to(corpus_dir)
    if len(relative.parts) < 2:
        raise ValueError(
            f"Unexpected task location {relative}; expected corpus/<concept>/<task>.json"
        )

    concept = relative.parts[0]
    task_id = relative.with_suffix("").as_posix()
    return concept, task_id


def load_conceptarc(corpus_dir: Path) -> pd.DataFrame:
    """Load ConceptARC into a deterministic pair-level pandas DataFrame."""
    rows: list[dict[str, Any]] = []
    task_files = discover_task_files(corpus_dir)

    for task_path in task_files:
        concept, task_id = infer_concept_and_task(task_path, corpus_dir)
        relative_path = task_path.relative_to(corpus_dir).as_posix()
        task_hash = sha256_file(task_path)

        with task_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(f"{relative_path}: task JSON must contain an object")

        unexpected = set(payload) - {"train", "test"}
        if unexpected:
            raise ValueError(f"{relative_path}: unexpected top-level keys {unexpected}")

        for split in ("train", "test"):
            pairs = payload.get(split)
            if not isinstance(pairs, list) or not pairs:
                raise ValueError(f"{relative_path}: '{split}' must be a non-empty list")

            for pair_index, pair in enumerate(pairs):
                context = f"{relative_path}:{split}[{pair_index}]"
                if not isinstance(pair, dict):
                    raise ValueError(f"{context}: pair must be a JSON object")
                if "input" not in pair or "output" not in pair:
                    raise ValueError(f"{context}: pair must contain input and output grids")

                input_grid = validate_grid(pair["input"], context=f"{context}:input")
                output_grid = validate_grid(pair["output"], context=f"{context}:output")
                pair_id = f"{task_id}::{split}::{pair_index:02d}"

                rows.append(
                    {
                        "pair_id": pair_id,
                        "concept": concept,
                        "task_id": task_id,
                        "task_file": relative_path,
                        "task_sha256": task_hash,
                        "split": split,
                        "pair_index": pair_index,
                        "input_height": len(input_grid),
                        "input_width": len(input_grid[0]),
                        "output_height": len(output_grid),
                        "output_width": len(output_grid[0]),
                        "input_grid_json": compact_json(input_grid),
                        "output_grid_json": compact_json(output_grid),
                    }
                )

    manifest = pd.DataFrame(rows).sort_values(
        ["concept", "task_id", "split", "pair_index"],
        kind="stable",
        ignore_index=True,
    )

    if manifest["pair_id"].duplicated().any():
        duplicates = manifest.loc[manifest["pair_id"].duplicated(), "pair_id"].tolist()
        raise ValueError(f"Duplicate pair IDs found: {duplicates[:5]}")

    return manifest


def validate_corpus_counts(manifest: pd.DataFrame, *, strict: bool) -> None:
    """Validate the expected main-corpus structure and optionally fail on mismatch."""
    concept_count = manifest["concept"].nunique()
    task_count = manifest["task_id"].nunique()
    pair_count = len(manifest)
    tasks_per_concept = manifest.groupby("concept")["task_id"].nunique()

    problems: list[str] = []
    if concept_count != EXPECTED_CONCEPTS:
        problems.append(f"expected {EXPECTED_CONCEPTS} concepts, found {concept_count}")
    if task_count != EXPECTED_TASKS:
        problems.append(f"expected {EXPECTED_TASKS} tasks, found {task_count}")
    if pair_count != EXPECTED_PAIRS:
        problems.append(f"expected {EXPECTED_PAIRS} pairs, found {pair_count}")

    bad_concepts = tasks_per_concept[tasks_per_concept != EXPECTED_TASKS_PER_CONCEPT]
    if not bad_concepts.empty:
        problems.append(
            "expected 10 tasks per concept; mismatches: "
            + ", ".join(f"{name}={count}" for name, count in bad_concepts.items())
        )

    if problems:
        message = "ConceptARC corpus validation failed:\n- " + "\n- ".join(problems)
        if strict:
            raise ValueError(message)
        print(f"WARNING: {message}")



def git_commit(repo_dir: Path) -> str | None:
    """Return the Git commit of the cloned dataset repository when available."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def save_manifest(manifest: pd.DataFrame, output_path: Path, corpus_dir: Path) -> None:
    """Write the manifest as compressed Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False, compression="zstd")

    repo_dir = corpus_dir.parent
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "dataset_repository_commit": git_commit(repo_dir),
        "concept_count": int(manifest["concept"].nunique()),
        "task_count": int(manifest["task_id"].nunique()),
        "pair_count": len(manifest),
        "split_counts": {
            str(key): int(value)
            for key, value in manifest["split"].value_counts().sort_index().items()
        },
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/ConceptARC/corpus"),
        help="Path to the official ConceptARC corpus directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manifest.parquet"),
        help="Output path for the pair-level manifest.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Warn instead of failing if official corpus counts do not match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_conceptarc(args.corpus_dir)
    validate_corpus_counts(manifest, strict=not args.allow_count_mismatch)
    save_manifest(manifest, args.output, args.corpus_dir)

    print(f"Saved manifest: {args.output}")
    print(f"Saved manifest metadata: {args.output.with_suffix('.meta.json')}")
    print(f"Concepts: {manifest['concept'].nunique()}")
    print(f"Tasks:    {manifest['task_id'].nunique()}")
    print(f"Pairs:    {len(manifest)}")
    print("Split counts:")
    print(manifest["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
