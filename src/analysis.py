"""Run the three core ConceptARC transformation-direction analyses.

The script performs:
1. task-level organization with task-label permutation inference;
2. leave-one-task-out nearest-centroid concept decoding;
3. strict train-to-test decoding using only train pairs from other tasks.

The task is the independent unit wherever a learned template is constructed.
Undefined zero-difference directions are excluded and explicitly reported.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "pair_id",
    "concept",
    "task_id",
    "split",
    "pair_index",
    "model_name",
    "embedding_dim",
    "direction_embedding",
    "direction_valid",
}


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit vector and fail if the input has near-zero norm."""
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def stack_vectors(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Convert one list-valued DataFrame column into a 2D float array."""
    matrix = np.asarray(frame[column].tolist(), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(frame):
        raise ValueError(f"Column {column} does not contain equal-length vectors")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Column {column} contains non-finite values")
    return matrix


def validate_embeddings(frame: pd.DataFrame) -> tuple[str, int]:
    """Validate the cached embedding table and return model name and dimension."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Embedding file is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Embedding file is empty")
    if frame["pair_id"].duplicated().any():
        raise ValueError("Embedding file contains duplicate pair IDs")
    if set(frame["split"].unique()) - {"train", "test"}:
        raise ValueError("split must contain only 'train' and 'test'")

    model_names = frame["model_name"].unique().tolist()
    dimensions = frame["embedding_dim"].unique().tolist()
    if len(model_names) != 1 or len(dimensions) != 1:
        raise ValueError("One analysis input must contain exactly one model and dimension")

    vectors = stack_vectors(frame, "direction_embedding")
    if vectors.shape[1] != int(dimensions[0]):
        raise ValueError("Stored embedding_dim does not match direction vectors")

    valid = frame["direction_valid"].astype(bool).to_numpy()
    norms = np.linalg.norm(vectors, axis=1)
    if valid.any() and not np.allclose(norms[valid], 1.0, atol=2e-5, rtol=0):
        raise ValueError("Valid direction vectors are not unit-normalized")
    if (~valid).any() and not np.allclose(norms[~valid], 0.0, atol=2e-5, rtol=0):
        raise ValueError("Invalid direction vectors are not zero vectors")

    concept_per_task = frame.groupby("task_id")["concept"].nunique()
    if (concept_per_task != 1).any():
        raise ValueError("Each task_id must map to exactly one concept")

    return str(model_names[0]), int(dimensions[0])


def build_task_representations(
    frame: pd.DataFrame,
    *,
    split: str | None = None,
) -> pd.DataFrame:
    """Average valid pair directions within each task, then normalize the mean."""
    selected = frame.loc[frame["direction_valid"].astype(bool)].copy()
    if split is not None:
        selected = selected.loc[selected["split"] == split].copy()
    if selected.empty:
        raise ValueError("No valid directions remain for task aggregation")

    rows: list[dict[str, Any]] = []
    for task_id, group in selected.groupby("task_id", sort=True):
        concepts = group["concept"].unique().tolist()
        if len(concepts) != 1:
            raise ValueError(f"Task {task_id} maps to multiple concepts")

        directions = stack_vectors(group, "direction_embedding")
        mean_direction = directions.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_direction))
        if mean_norm <= 1e-12:
            raise ValueError(
                f"Valid pair directions cancel to a zero task mean for {task_id}"
            )

        rows.append(
            {
                "task_id": task_id,
                "concept": concepts[0],
                "split_used": split or "all",
                "valid_pair_count": len(group),
                "task_mean_pre_normalization_norm": mean_norm,
                "task_direction": (mean_direction / mean_norm).tolist(),
            }
        )

    task_frame = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)

    all_tasks = set(frame["task_id"].unique())
    represented_tasks = set(task_frame["task_id"])
    missing_tasks = sorted(all_tasks - represented_tasks)
    if split is None and missing_tasks:
        raise ValueError(
            "Some tasks have no valid directions: " + ", ".join(missing_tasks)
        )

    return task_frame


def organization_statistic(
    task_vectors: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, int, int]:
    """Compute within-concept mean, between-concept mean, and their difference."""
    cosine = task_vectors @ task_vectors.T
    upper = np.triu_indices(len(labels), k=1)
    pair_cosines = cosine[upper]
    same = labels[upper[0]] == labels[upper[1]]

    within = pair_cosines[same]
    between = pair_cosines[~same]
    if len(within) == 0 or len(between) == 0:
        raise ValueError("Need both same-concept and different-concept task pairs")

    within_mean = float(within.mean())
    between_mean = float(between.mean())
    return (
        within_mean,
        between_mean,
        within_mean - between_mean,
        int(len(within)),
        int(len(between)),
    )


def run_task_organization(
    task_frame: pd.DataFrame,
    *,
    permutations: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run the organization test with task-label permutations."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")

    vectors = stack_vectors(task_frame, "task_direction")
    labels = task_frame["concept"].to_numpy(dtype=str)
    observed = organization_statistic(vectors, labels)

    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = rng.permutation(labels)
        null[index] = organization_statistic(vectors, shuffled)[2]

    # One-sided test for stronger within-concept organization than expected.
    p_value = float((1 + np.count_nonzero(null >= observed[2])) / (permutations + 1))
    result = {
        "within_concept_mean_cosine": observed[0],
        "between_concept_mean_cosine": observed[1],
        "within_minus_between": observed[2],
        "within_task_pair_count": observed[3],
        "between_task_pair_count": observed[4],
        "permutations": permutations,
        "permutation_seed": seed,
        "permutation_p_value_one_sided": p_value,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_2_5_percentile": float(np.quantile(null, 0.025)),
        "null_97_5_percentile": float(np.quantile(null, 0.975)),
    }
    return result, null


def build_centroids(task_frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Construct normalized, task-balanced concept centroids."""
    concepts = sorted(task_frame["concept"].unique().tolist())
    centroids: list[np.ndarray] = []
    for concept in concepts:
        group = task_frame.loc[task_frame["concept"] == concept]
        if group.empty:
            raise ValueError(f"No task representation available for {concept}")
        vectors = stack_vectors(group, "task_direction")
        centroids.append(l2_normalize(vectors.mean(axis=0)))
    return concepts, np.vstack(centroids)


def nearest_centroid(
    vector: np.ndarray,
    concepts: list[str],
    centroids: np.ndarray,
) -> tuple[str, float, float]:
    """Predict a label by cosine similarity and return top-two margin."""
    similarities = centroids @ vector
    order = np.argsort(-similarities, kind="stable")
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    return (
        concepts[best],
        float(similarities[best]),
        float(similarities[best] - similarities[second]),
    )


def run_leave_one_task_out(task_frame: pd.DataFrame) -> pd.DataFrame:
    """Decode each held-out task using centroids from all other tasks."""
    rows: list[dict[str, Any]] = []
    for held_out in task_frame.itertuples(index=False):
        training = task_frame.loc[task_frame["task_id"] != held_out.task_id]
        concepts, centroids = build_centroids(training)
        vector = np.asarray(held_out.task_direction, dtype=np.float64)
        prediction, score, margin = nearest_centroid(vector, concepts, centroids)
        rows.append(
            {
                "task_id": held_out.task_id,
                "true_concept": held_out.concept,
                "predicted_concept": prediction,
                "correct": prediction == held_out.concept,
                "predicted_cosine": score,
                "top_two_margin": margin,
                "valid_pair_count": int(held_out.valid_pair_count),
                "training_task_count": len(training),
            }
        )
    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def run_strict_train_to_test(frame: pd.DataFrame) -> pd.DataFrame:
    """Decode held-out test pairs using only train pairs from other tasks.

    Within each fold, train-pair directions are first averaged within each
    training task. Concept centroids are then averaged across those task-level
    representations, preventing tasks with more demonstrations from receiving
    greater weight.
    """
    valid = frame.loc[frame["direction_valid"].astype(bool)].copy()
    valid_train = valid.loc[valid["split"] == "train"].copy()
    valid_test = valid.loc[valid["split"] == "test"].copy()
    if valid_train.empty or valid_test.empty:
        raise ValueError("Strict decoding requires valid train and test directions")

    rows: list[dict[str, Any]] = []
    all_task_ids = sorted(frame["task_id"].unique().tolist())
    for held_out_task in all_task_ids:
        training_pairs = valid_train.loc[valid_train["task_id"] != held_out_task]
        training_tasks = build_task_representations(training_pairs, split="train")
        concepts, centroids = build_centroids(training_tasks)

        held_out_all_test = frame.loc[
            (frame["task_id"] == held_out_task) & (frame["split"] == "test")
        ].sort_values("pair_index")
        if held_out_all_test.empty:
            raise ValueError(f"Held-out task has no test pairs: {held_out_task}")

        for pair in held_out_all_test.itertuples(index=False):
            base = {
                "task_id": held_out_task,
                "pair_id": pair.pair_id,
                "pair_index": int(pair.pair_index),
                "true_concept": pair.concept,
                "direction_valid": bool(pair.direction_valid),
                "training_task_count": len(training_tasks),
            }
            if not bool(pair.direction_valid):
                rows.append(
                    {
                        **base,
                        "predicted_concept": None,
                        "correct": pd.NA,
                        "predicted_cosine": np.nan,
                        "top_two_margin": np.nan,
                        "exclusion_reason": "undefined_zero_difference_direction",
                    }
                )
                continue

            vector = np.asarray(pair.direction_embedding, dtype=np.float64)
            prediction, score, margin = nearest_centroid(vector, concepts, centroids)
            rows.append(
                {
                    **base,
                    "predicted_concept": prediction,
                    "correct": prediction == pair.concept,
                    "predicted_cosine": score,
                    "top_two_margin": margin,
                    "exclusion_reason": None,
                }
            )

    result = pd.DataFrame(rows).sort_values(["task_id", "pair_index"]).reset_index(
        drop=True
    )
    result["correct"] = result["correct"].astype("boolean")
    return result


def bootstrap_task_accuracy(
    predictions: pd.DataFrame,
    *,
    bootstraps: int,
    seed: int,
    task_column: str = "task_id",
) -> dict[str, float]:
    """Compute a percentile CI by resampling held-out tasks with replacement."""
    if bootstraps <= 0:
        raise ValueError("bootstraps must be positive")

    usable = predictions.loc[predictions["correct"].notna()].copy()
    if usable.empty:
        raise ValueError("No usable predictions for accuracy")

    tasks = sorted(usable[task_column].unique().tolist())
    task_values = {
        task: usable.loc[usable[task_column] == task, "correct"]
        .astype(bool)
        .to_numpy(dtype=np.float64)
        for task in tasks
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstraps, dtype=np.float64)

    for index in range(bootstraps):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        values = np.concatenate([task_values[task] for task in sampled_tasks])
        samples[index] = values.mean()

    observed = float(usable["correct"].astype(bool).mean())
    return {
        "accuracy": observed,
        "ci_95_lower": float(np.quantile(samples, 0.025)),
        "ci_95_upper": float(np.quantile(samples, 0.975)),
        "bootstrap_replicates": bootstraps,
        "bootstrap_seed": seed,
        "evaluated_prediction_count": int(len(usable)),
        "held_out_task_count": int(len(tasks)),
    }


def per_concept_accuracy(predictions: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Summarize accuracy and sample count for each true concept."""
    usable = predictions.loc[predictions["correct"].notna()].copy()
    output: dict[str, dict[str, float | int]] = {}
    for concept, group in usable.groupby("true_concept", sort=True):
        output[str(concept)] = {
            "accuracy": float(group["correct"].astype(bool).mean()),
            "count": int(len(group)),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["clip", "dinov2"], required=True)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Defaults to outputs/<model>_embeddings.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase2"),
    )
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_path = args.embeddings or Path(
        f"outputs/{args.model}_embeddings.parquet"
    )
    frame = pd.read_parquet(embeddings_path)
    model_name, embedding_dim = validate_embeddings(frame)
    if model_name != args.model:
        raise ValueError(
            f"Requested model {args.model}, but file contains model {model_name}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.model

    task_frame = build_task_representations(frame)
    organization, null = run_task_organization(
        task_frame,
        permutations=args.permutations,
        seed=args.seed,
    )
    loto = run_leave_one_task_out(task_frame)
    strict = run_strict_train_to_test(frame)

    loto_accuracy = bootstrap_task_accuracy(
        loto,
        bootstraps=args.bootstraps,
        seed=args.seed + 1,
    )
    strict_accuracy = bootstrap_task_accuracy(
        strict,
        bootstraps=args.bootstraps,
        seed=args.seed + 2,
    )

    valid_count = int(frame["direction_valid"].sum())
    invalid_rows = frame.loc[~frame["direction_valid"].astype(bool)]
    strict_excluded = int(strict["correct"].isna().sum())

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "source_embeddings": str(embeddings_path),
        "seed": args.seed,
        "pair_count": int(len(frame)),
        "task_count": int(frame["task_id"].nunique()),
        "concept_count": int(frame["concept"].nunique()),
        "valid_direction_count": valid_count,
        "invalid_direction_count": int(len(invalid_rows)),
        "invalid_direction_pair_ids": invalid_rows["pair_id"].tolist(),
        "task_level_organization": organization,
        "leave_one_task_out": {
            **loto_accuracy,
            "chance_accuracy": 1.0 / frame["concept"].nunique(),
            "per_concept": per_concept_accuracy(loto),
        },
        "strict_train_to_test": {
            **strict_accuracy,
            "chance_accuracy": 1.0 / frame["concept"].nunique(),
            "all_test_pair_count": int((frame["split"] == "test").sum()),
            "excluded_undefined_test_direction_count": strict_excluded,
            "macro_task_accuracy": float(
                strict.loc[strict["correct"].notna()]
                .assign(correct_float=lambda x: x["correct"].astype(float))
                .groupby("task_id")["correct_float"]
                .mean()
                .mean()
            ),
            "per_concept": per_concept_accuracy(strict),
        },
        "method_notes": {
            "task_direction": (
                "L2-normalized mean of valid pair directions within each task, "
                "using train and test pairs for organization and LOTO."
            ),
            "organization_inference": (
                "One-sided permutation test shuffling concept labels across tasks "
                "while preserving label counts."
            ),
            "loto_centroids": (
                "L2-normalized mean of other-task task directions within each concept."
            ),
            "strict_centroids": (
                "For every held-out task, train-pair directions from other tasks are "
                "first averaged within task; task representations are then averaged "
                "within concept. No pair from the held-out task enters a centroid."
            ),
            "undefined_directions": (
                "Zero endpoint differences have no unit direction; they are preserved "
                "in the embedding file, excluded from direction analyses, and counted."
            ),
            "uncertainty": (
                "Accuracy confidence intervals resample held-out tasks, not individual pairs."
            ),
        },
    }

    task_frame.to_parquet(
        prefix.with_name(f"{args.model}_task_directions.parquet"),
        index=False,
        compression="zstd",
    )
    loto.to_csv(prefix.with_name(f"{args.model}_loto_predictions.csv"), index=False)
    strict.to_csv(
        prefix.with_name(f"{args.model}_strict_predictions.csv"), index=False
    )
    np.save(
        prefix.with_name(f"{args.model}_organization_permutation_null.npy"), null
    )
    summary_path = prefix.with_name(f"{args.model}_phase2_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Model: {model_name}")
    print(f"Valid directions: {valid_count}/{len(frame)}")
    print(
        "Organization effect "
        f"(within - between): {organization['within_minus_between']:.6f}"
    )
    print(
        "Permutation p-value: "
        f"{organization['permutation_p_value_one_sided']:.6g}"
    )
    print(
        "Leave-one-task-out accuracy: "
        f"{loto_accuracy['accuracy']:.4f} "
        f"[{loto_accuracy['ci_95_lower']:.4f}, {loto_accuracy['ci_95_upper']:.4f}]"
    )
    print(
        "Strict train-to-test accuracy: "
        f"{strict_accuracy['accuracy']:.4f} "
        f"[{strict_accuracy['ci_95_lower']:.4f}, {strict_accuracy['ci_95_upper']:.4f}]"
    )
    print(f"Saved Phase 2 outputs under: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()