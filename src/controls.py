"""Run the interpretation-critical ConceptARC controls.

Controls included:
1. static input, output, and input/output-concatenation representations;
2. explicit visual-change feature decoding;
3. foldwise ridge residualization of directions on visual-change features;
4. repetition of the direction analyses after removing Count.

Rendering robustness is run by extracting named renderer conditions with
embeddings.py and applying analysis.py to each resulting file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import (
    bootstrap_task_accuracy,
    build_task_representations,
    per_concept_accuracy,
    run_leave_one_task_out,
    run_strict_train_to_test,
    run_task_organization,
    validate_embeddings,
)


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > eps
    output = np.zeros_like(matrix, dtype=np.float64)
    output[valid] = matrix[valid] / norms[valid, None]
    return output, valid


def stack(frame: pd.DataFrame, column: str) -> np.ndarray:
    matrix = np.asarray(frame[column].tolist(), dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid vector column: {column}")
    return matrix


def representation_frame(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Convert a cached endpoint representation into analysis.py's schema."""
    inp = stack(frame, "input_embedding")
    out = stack(frame, "output_embedding")
    if kind == "input":
        matrix = inp
    elif kind == "output":
        matrix = out
    elif kind == "concat":
        matrix = np.concatenate([inp, out], axis=1)
    elif kind == "direction":
        return frame.copy()
    else:
        raise ValueError(kind)
    matrix, valid = normalize_rows(matrix)
    if not valid.all():
        raise ValueError(f"Unexpected zero vector in static representation {kind}")
    result = frame.copy()
    result["direction_embedding"] = [x.astype(np.float32).tolist() for x in matrix]
    result["direction_valid"] = True
    result["embedding_dim"] = matrix.shape[1]
    result["model_name"] = f"{frame['model_name'].iloc[0]}_{kind}"
    return result


def summarize_analysis(
    frame: pd.DataFrame,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    task_frame = build_task_representations(frame)
    org, _ = run_task_organization(task_frame, permutations=permutations, seed=seed)
    loto = run_leave_one_task_out(task_frame)
    strict = run_strict_train_to_test(frame)
    loto_ci = bootstrap_task_accuracy(loto, bootstraps=bootstraps, seed=seed+1)
    strict_ci = bootstrap_task_accuracy(strict, bootstraps=bootstraps, seed=seed+2)
    return {
        "task_level_organization": org,
        "leave_one_task_out": {
            **loto_ci,
            "chance_accuracy": 1/frame["concept"].nunique(),
            "per_concept": per_concept_accuracy(loto),
        },
        "strict_train_to_test": {
            **strict_ci,
            "chance_accuracy": 1/frame["concept"].nunique(),
            "all_test_pair_count": int((frame["split"] == "test").sum()),
            "excluded_count": int(strict["correct"].isna().sum()),
            "per_concept": per_concept_accuracy(strict),
        },
    }


def task_average(frame: pd.DataFrame, vector_column: str, split: str | None = None) -> pd.DataFrame:
    selected = frame if split is None else frame.loc[frame["split"] == split]
    rows = []
    for task_id, group in selected.groupby("task_id", sort=True):
        matrix = stack(group, vector_column)
        rows.append({
            "task_id": task_id,
            "concept": group["concept"].iloc[0],
            "vector": matrix.mean(axis=0),
            "pair_count": len(group),
        })
    return pd.DataFrame(rows)


def fit_classifier(x: np.ndarray, y: np.ndarray) -> Any:
    """A fixed, foldwise standardized multinomial linear baseline."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            class_weight=None,
            random_state=0,
        ),
    )
    model.fit(x, y)
    return model


def visual_loto(features: pd.DataFrame) -> pd.DataFrame:
    tasks = task_average(features, "visual_features")
    rows = []
    for held in tasks.itertuples(index=False):
        train = tasks.loc[tasks["task_id"] != held.task_id]
        model = fit_classifier(stack(train, "vector"), train["concept"].to_numpy())
        pred = str(model.predict(np.asarray(held.vector)[None, :])[0])
        rows.append({"task_id": held.task_id, "true_concept": held.concept,
                     "predicted_concept": pred, "correct": pred == held.concept})
    return pd.DataFrame(rows)


def visual_strict(features: pd.DataFrame) -> pd.DataFrame:
    tasks = sorted(features["task_id"].unique())
    rows = []
    for held_task in tasks:
        train_pairs = features.loc[(features["task_id"] != held_task) & (features["split"] == "train")]
        train_tasks = task_average(train_pairs, "visual_features")
        model = fit_classifier(stack(train_tasks, "vector"), train_tasks["concept"].to_numpy())
        test = features.loc[(features["task_id"] == held_task) & (features["split"] == "test")].sort_values("pair_index")
        preds = model.predict(stack(test, "visual_features"))
        for row, pred in zip(test.itertuples(index=False), preds):
            rows.append({"task_id": held_task, "pair_id": row.pair_id,
                         "true_concept": row.concept, "predicted_concept": str(pred),
                         "correct": str(pred) == row.concept})
    return pd.DataFrame(rows)


def visual_summary(features: pd.DataFrame, bootstraps: int, seed: int) -> dict[str, Any]:
    loto = visual_loto(features)
    strict = visual_strict(features)
    return {
        "model": "foldwise StandardScaler + multinomial logistic regression",
        "regularization": "C=1.0 fixed before evaluation",
        "task_balancing": "training pair features are averaged within task before fitting",
        "leave_one_task_out": {
            **bootstrap_task_accuracy(loto, bootstraps=bootstraps, seed=seed),
            "chance_accuracy": 1/features["concept"].nunique(),
            "per_concept": per_concept_accuracy(loto),
        },
        "strict_train_to_test": {
            **bootstrap_task_accuracy(strict, bootstraps=bootstraps, seed=seed+1),
            "chance_accuracy": 1/features["concept"].nunique(),
            "per_concept": per_concept_accuracy(strict),
        },
    }


def merge_features(emb: pd.DataFrame, visual: pd.DataFrame) -> pd.DataFrame:
    cols = ["pair_id", "visual_features", "visual_feature_dim"]
    merged = emb.merge(visual[cols], on="pair_id", how="left", validate="one_to_one")
    if merged["visual_features"].isna().any():
        raise ValueError("Missing visual features after pair_id join")
    return merged


def residualize_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit visual->direction on training data only and return unit residuals."""
    scaler = StandardScaler().fit(train_x)
    x_train = scaler.transform(train_x)
    x_eval = scaler.transform(eval_x)
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(x_train, train_y)
    train_res = train_y - ridge.predict(x_train)
    eval_res = eval_y - ridge.predict(x_eval)
    train_res, train_valid = normalize_rows(train_res)
    eval_res, eval_valid = normalize_rows(eval_res)
    if not train_valid.all() or not eval_valid.all():
        raise ValueError("Residualization produced a near-zero residual")
    return train_res, eval_res


def centroid_predict(train_vectors: np.ndarray, train_labels: np.ndarray, vector: np.ndarray) -> str:
    concepts = sorted(np.unique(train_labels).tolist())
    centroids = []
    for concept in concepts:
        centroid, valid = normalize_rows(train_vectors[train_labels == concept].mean(axis=0, keepdims=True))
        if not valid[0]:
            raise ValueError(f"Zero residual centroid for {concept}")
        centroids.append(centroid[0])
    scores = np.vstack(centroids) @ vector
    return concepts[int(np.argmax(scores))]


def residual_crossfit_tasks(merged: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Create one residual task vector using a model fit on all other tasks."""
    valid = merged.loc[merged["direction_valid"].astype(bool)]
    d_tasks = task_average(valid, "direction_embedding")
    v_tasks = task_average(valid, "visual_features")
    tasks = d_tasks.merge(
        v_tasks[["task_id", "vector"]], on="task_id",
        suffixes=("_direction", "_visual"), validate="one_to_one"
    )
    rows = []
    for held in tasks.itertuples(index=False):
        train = tasks.loc[tasks["task_id"] != held.task_id]
        _, held_res = residualize_fold(
            stack(train, "vector_visual"), stack(train, "vector_direction"),
            np.asarray(held.vector_visual)[None, :],
            np.asarray(held.vector_direction)[None, :], alpha,
        )
        rows.append({
            "task_id": held.task_id, "concept": held.concept,
            "task_direction": held_res[0].astype(np.float32).tolist(),
            "valid_pair_count": int(held.pair_count),
        })
    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def residual_loto(merged: pd.DataFrame, alpha: float) -> pd.DataFrame:
    valid = merged.loc[merged["direction_valid"].astype(bool)]
    direction_tasks = task_average(valid, "direction_embedding")
    visual_tasks = task_average(valid, "visual_features")
    tasks = direction_tasks.merge(visual_tasks[["task_id", "vector"]], on="task_id", suffixes=("_direction", "_visual"), validate="one_to_one")
    rows = []
    for held in tasks.itertuples(index=False):
        train = tasks.loc[tasks["task_id"] != held.task_id]
        train_res, held_res = residualize_fold(
            stack(train, "vector_visual"), stack(train, "vector_direction"),
            np.asarray(held.vector_visual)[None,:], np.asarray(held.vector_direction)[None,:], alpha,
        )
        pred = centroid_predict(train_res, train["concept"].to_numpy(), held_res[0])
        rows.append({"task_id": held.task_id, "true_concept": held.concept,
                     "predicted_concept": pred, "correct": pred == held.concept})
    return pd.DataFrame(rows)


def residual_strict(merged: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for held_task in sorted(merged["task_id"].unique()):
        train_pairs = merged.loc[(merged["task_id"] != held_task) & (merged["split"] == "train") & merged["direction_valid"].astype(bool)]
        d_tasks = task_average(train_pairs, "direction_embedding")
        v_tasks = task_average(train_pairs, "visual_features")
        train = d_tasks.merge(v_tasks[["task_id","vector"]], on="task_id", suffixes=("_direction","_visual"), validate="one_to_one")
        test = merged.loc[(merged["task_id"] == held_task) & (merged["split"] == "test")].sort_values("pair_index")
        usable = test.loc[test["direction_valid"].astype(bool)]
        if len(usable):
            train_res, test_res = residualize_fold(
                stack(train, "vector_visual"), stack(train, "vector_direction"),
                stack(usable, "visual_features"), stack(usable, "direction_embedding"), alpha,
            )
            labels = train["concept"].to_numpy()
            predictions = [centroid_predict(train_res, labels, v) for v in test_res]
            prediction_map = dict(zip(usable["pair_id"], predictions))
        else:
            prediction_map = {}
        for row in test.itertuples(index=False):
            pred = prediction_map.get(row.pair_id)
            rows.append({"task_id": held_task, "pair_id": row.pair_id,
                         "true_concept": row.concept, "predicted_concept": pred,
                         "correct": pd.NA if pred is None else pred == row.concept})
    result = pd.DataFrame(rows)
    result["correct"] = result["correct"].astype("boolean")
    return result


def residual_summary(merged: pd.DataFrame, alpha: float, bootstraps: int, seed: int, permutations: int) -> dict[str, Any]:
    crossfit_tasks = residual_crossfit_tasks(merged, alpha)
    organization, _ = run_task_organization(
        crossfit_tasks, permutations=permutations, seed=seed
    )
    loto = residual_loto(merged, alpha)
    strict = residual_strict(merged, alpha)
    return {
        "task_level_organization_cross_fitted": organization,
        "regression": "foldwise StandardScaler + multi-output Ridge",
        "ridge_alpha": alpha,
        "leakage_control": "scaler and ridge are fit using other tasks only in every held-out-task fold",
        "leave_one_task_out": {
            **bootstrap_task_accuracy(loto, bootstraps=bootstraps, seed=seed),
            "chance_accuracy": 1/merged["concept"].nunique(),
            "per_concept": per_concept_accuracy(loto),
        },
        "strict_train_to_test": {
            **bootstrap_task_accuracy(strict, bootstraps=bootstraps, seed=seed+1),
            "chance_accuracy": 1/merged["concept"].nunique(),
            "excluded_undefined_test_direction_count": int(strict["correct"].isna().sum()),
            "per_concept": per_concept_accuracy(strict),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["clip", "dinov2"], required=True)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--visual-features", type=Path, default=Path("outputs/visual_features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    emb_path = args.embeddings or Path(f"outputs/{args.model}_embeddings.parquet")
    emb = pd.read_parquet(emb_path)
    validate_embeddings(emb)
    visual = pd.read_parquet(args.visual_features)
    merged = merge_features(emb, visual)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static = {}
    for offset, kind in enumerate(["input", "output", "concat"]):
        static[kind] = summarize_analysis(
            representation_frame(emb, kind), permutations=args.permutations,
            bootstraps=args.bootstraps, seed=args.seed + 10*offset,
        )

    without_count = emb.loc[emb["concept"] != "Count"].copy().reset_index(drop=True)
    no_count = summarize_analysis(
        without_count, permutations=args.permutations,
        bootstraps=args.bootstraps, seed=args.seed+100,
    )
    rich = visual_summary(visual, args.bootstraps, args.seed+200)
    residual = residual_summary(merged, args.ridge_alpha, args.bootstraps, args.seed+300, args.permutations)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source_embeddings": str(emb_path),
        "source_visual_features": str(args.visual_features),
        "static_endpoints": static,
        "rich_visual_change_baseline": rich,
        "foldwise_residualization": residual,
        "direction_without_count": no_count,
        "notes": {
            "static_fairness": "Static representations use the same task aggregation, centroid decoder, permutation test, and task bootstrap as directions.",
            "visual_baseline": "Explicit visual features are standardized inside each fold; training pairs are averaged within task before fitting.",
            "residualization": "No held-out-task row is used to fit either the feature scaler or visual-to-direction ridge model.",
        },
    }
    out = args.output_dir / f"{args.model}_phase3_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {out}")
    for kind, result in static.items():
        print(f"{kind}: LOTO={result['leave_one_task_out']['accuracy']:.4f}, strict={result['strict_train_to_test']['accuracy']:.4f}")
    print(f"visual: LOTO={rich['leave_one_task_out']['accuracy']:.4f}, strict={rich['strict_train_to_test']['accuracy']:.4f}")
    print(f"residual: LOTO={residual['leave_one_task_out']['accuracy']:.4f}, strict={residual['strict_train_to_test']['accuracy']:.4f}")
    print(f"without Count: LOTO={no_count['leave_one_task_out']['accuracy']:.4f}, strict={no_count['strict_train_to_test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()