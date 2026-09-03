"""Collect ConceptARC analysis summaries into tables and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def phase2_row(model: str, condition: str, summary: dict[str, Any]) -> dict[str, Any]:
    org = summary["task_level_organization"]
    loto = summary["leave_one_task_out"]
    strict = summary["strict_train_to_test"]
    return {
        "model": model,
        "condition": condition,
        "representation": "direction",
        "organization_effect": org["within_minus_between"],
        "organization_p": org["permutation_p_value_one_sided"],
        "loto_accuracy": loto["accuracy"],
        "loto_ci_low": loto["ci_95_lower"],
        "loto_ci_high": loto["ci_95_upper"],
        "strict_accuracy": strict["accuracy"],
        "strict_ci_low": strict["ci_95_lower"],
        "strict_ci_high": strict["ci_95_upper"],
    }


def control_rows(model: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = {
        "input": summary["static_endpoints"]["input"],
        "output": summary["static_endpoints"]["output"],
        "input_output_concat": summary["static_endpoints"]["concat"],
        "residual_direction": summary["foldwise_residualization"],
        "direction_without_count": summary["direction_without_count"],
    }
    for name, result in groups.items():
        org = result.get("task_level_organization") or result.get(
            "task_level_organization_cross_fitted"
        )
        loto = result["leave_one_task_out"]
        strict = result["strict_train_to_test"]
        rows.append({
            "model": model,
            "condition": "baseline",
            "representation": name,
            "organization_effect": org["within_minus_between"],
            "organization_p": org["permutation_p_value_one_sided"],
            "loto_accuracy": loto["accuracy"],
            "loto_ci_low": loto["ci_95_lower"],
            "loto_ci_high": loto["ci_95_upper"],
            "strict_accuracy": strict["accuracy"],
            "strict_ci_low": strict["ci_95_lower"],
            "strict_ci_high": strict["ci_95_upper"],
        })
    visual = summary["rich_visual_change_baseline"]
    rows.append({
        "model": model,
        "condition": "baseline",
        "representation": "rich_visual_features",
        "organization_effect": None,
        "organization_p": None,
        "loto_accuracy": visual["leave_one_task_out"]["accuracy"],
        "loto_ci_low": visual["leave_one_task_out"]["ci_95_lower"],
        "loto_ci_high": visual["leave_one_task_out"]["ci_95_upper"],
        "strict_accuracy": visual["strict_train_to_test"]["accuracy"],
        "strict_ci_low": visual["strict_train_to_test"]["ci_95_lower"],
        "strict_ci_high": visual["strict_train_to_test"]["ci_95_upper"],
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument("--phase3-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument(
        "--rendering-dir", type=Path, default=Path("outputs/rendering_analysis")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    concept_rows: list[dict[str, Any]] = []
    for model in ("clip", "dinov2"):
        base = load_json(args.phase2_dir / f"{model}_phase2_summary.json")
        rows.append(phase2_row(model, "baseline", base))
        for concept, values in base["leave_one_task_out"]["per_concept"].items():
            concept_rows.append({
                "model": model,
                "concept": concept,
                "loto_accuracy": values["accuracy"],
                "count": values["count"],
            })
        controls = load_json(args.phase3_dir / f"{model}_phase3_summary.json")
        rows.extend(control_rows(model, controls))

        for condition in ("no_grid", "white_canvas", "padding0", "colorperm42"):
            path = (
                args.rendering_dir
                / f"{model}_{condition}"
                / f"{model}_phase2_summary.json"
            )
            rows.append(phase2_row(model, condition, load_json(path)))

    result_table = pd.DataFrame(rows)
    result_table.to_csv(args.output_dir / "results_table.csv", index=False)
    concept_table = pd.DataFrame(concept_rows)
    concept_table.to_csv(args.output_dir / "concept_loto_accuracy.csv", index=False)

    pivot = concept_table.pivot(index="concept", columns="model", values="loto_accuracy")
    pivot = pivot.sort_values("dinov2", ascending=False)
    ax = pivot.plot(kind="bar", figsize=(12, 5))
    ax.set_ylabel("Leave-one-task-out accuracy")
    ax.set_xlabel("Concept family")
    ax.set_ylim(0, 1)
    ax.axhline(1 / 16, linestyle="--", linewidth=1)
    ax.legend(title="Encoder")
    plt.xticks(rotation=55, ha="right")
    plt.tight_layout()
    plt.savefig(args.output_dir / "concept_accuracy_by_encoder.png", dpi=300)
    plt.close()

    agreement = pivot.dropna().corr().iloc[0, 1]
    (args.output_dir / "concept_agreement.json").write_text(
        json.dumps({"pearson_correlation": float(agreement)}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {args.output_dir / 'results_table.csv'}")
    print(f"Saved: {args.output_dir / 'concept_loto_accuracy.csv'}")
    print(f"Saved: {args.output_dir / 'concept_accuracy_by_encoder.png'}")
    print(f"CLIP-DINOv2 concept accuracy correlation: {agreement:.4f}")


if __name__ == "__main__":
    main()