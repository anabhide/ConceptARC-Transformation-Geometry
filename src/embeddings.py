"""Extract and cache normalized CLIP or DINOv2 endpoint and direction vectors.

For each ConceptARC pair, this script renders the input and output grids,
extracts frozen image representations, L2-normalizes both endpoints, computes
output minus input, and L2-normalizes that transformation direction.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
import PIL
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

from renderer import BASELINE_CONFIG, parse_grid, render_grid

MODEL_SPECS = {
    "clip": {
        "checkpoint": "openai/clip-vit-base-patch32",
        "expected_dim": 512,
        "feature_type": "projected_image_embedding",
    },
    "dinov2": {
        "checkpoint": "facebook/dinov2-base",
        "expected_dim": 768,
        "feature_type": "cls_token",
    },
}


def set_reproducibility(seed: int) -> None:
    """Set inference seeds and deterministic backend preferences."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """Resolve 'auto', 'cpu', or an explicit CUDA device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Choose a safe inference dtype."""
    if name == "auto":
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use float32 for CPU inference")
    return dtype


def load_model(
    model_name: str, device: torch.device, dtype: torch.dtype
) -> tuple[Any, torch.nn.Module, Callable[[Any], torch.Tensor]]:
    """Load one frozen encoder and return its processor and feature function."""
    spec = MODEL_SPECS[model_name]
    checkpoint = spec["checkpoint"]
    processor = AutoImageProcessor.from_pretrained(checkpoint)

    if model_name == "clip":
        model = CLIPVisionModelWithProjection.from_pretrained(
            checkpoint, torch_dtype=dtype
        )

        def feature_fn(outputs: Any) -> torch.Tensor:
            return outputs.image_embeds

    elif model_name == "dinov2":
        model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype)

        def feature_fn(outputs: Any) -> torch.Tensor:
            return outputs.last_hidden_state[:, 0, :]

    else:  # Protected by argparse choices, retained for direct function calls.
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return processor, model, feature_fn


def batched(items: list[Any], batch_size: int):
    """Yield successive batches from a list."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_images(
    images: list[Image.Image],
    *,
    processor: Any,
    model: torch.nn.Module,
    feature_fn: Callable[[Any], torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode images and return normalized vectors plus pre-normalization norms."""
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device)

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)
        raw_features = feature_fn(outputs).float()
        raw_norms = torch.linalg.vector_norm(raw_features, dim=1)
        if torch.any(raw_norms <= 1e-12):
            raise RuntimeError("Encountered an image feature with near-zero norm")
        normalized = F.normalize(raw_features, p=2, dim=1)

    return (
        normalized.cpu().numpy().astype(np.float32, copy=False),
        raw_norms.cpu().numpy().astype(np.float32, copy=False),
    )


def extract_embeddings(
    manifest: pd.DataFrame,
    *,
    model_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    """Render all endpoints and compute normalized endpoints and directions."""
    required = {
        "pair_id",
        "concept",
        "task_id",
        "task_file",
        "task_sha256",
        "split",
        "pair_index",
        "input_height",
        "input_width",
        "output_height",
        "output_width",
        "input_grid_json",
        "output_grid_json",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    processor, model, feature_fn = load_model(model_name, device, dtype)
    records = manifest.to_dict(orient="records")
    output_rows: list[dict[str, Any]] = []

    progress = tqdm(total=len(records), desc=f"Extracting {model_name}")
    for batch_records in batched(records, batch_size):
        input_images = [
            render_grid(parse_grid(record["input_grid_json"]), BASELINE_CONFIG)
            for record in batch_records
        ]
        output_images = [
            render_grid(parse_grid(record["output_grid_json"]), BASELINE_CONFIG)
            for record in batch_records
        ]

        # Encode inputs and outputs separately so endpoint identity stays explicit.
        input_features, input_raw_norms = encode_images(
            input_images,
            processor=processor,
            model=model,
            feature_fn=feature_fn,
            device=device,
        )
        output_features, output_raw_norms = encode_images(
            output_images,
            processor=processor,
            model=model,
            feature_fn=feature_fn,
            device=device,
        )

        differences = output_features - input_features
        difference_norms = np.linalg.norm(differences, axis=1)
        if np.any(difference_norms <= 1e-12):
            bad = np.where(difference_norms <= 1e-12)[0].tolist()
            pair_ids = [batch_records[index]["pair_id"] for index in bad]
            raise RuntimeError(
                "Encountered near-zero output-minus-input vector for: "
                + ", ".join(pair_ids)
            )
        directions = differences / difference_norms[:, None]

        for index, record in enumerate(batch_records):
            output_rows.append(
                {
                    **record,
                    "model_name": model_name,
                    "model_checkpoint": MODEL_SPECS[model_name]["checkpoint"],
                    "feature_type": MODEL_SPECS[model_name]["feature_type"],
                    "embedding_dim": int(input_features.shape[1]),
                    "renderer_id": BASELINE_CONFIG.renderer_id,
                    "input_embedding": input_features[index].tolist(),
                    "output_embedding": output_features[index].tolist(),
                    "direction_embedding": directions[index]
                    .astype(np.float32, copy=False)
                    .tolist(),
                    "input_raw_norm": float(input_raw_norms[index]),
                    "output_raw_norm": float(output_raw_norms[index]),
                    "endpoint_difference_norm": float(difference_norms[index]),
                }
            )

        progress.update(len(batch_records))

    progress.close()
    result = pd.DataFrame(output_rows)

    expected_dim = MODEL_SPECS[model_name]["expected_dim"]
    observed_dims = set(result["embedding_dim"].unique())
    if observed_dims != {expected_dim}:
        raise RuntimeError(
            f"Expected {model_name} dimension {expected_dim}, observed {observed_dims}"
        )

    return result


def validate_saved_vectors(frame: pd.DataFrame, tolerance: float = 2e-5) -> None:
    """Check vector dimensions, finiteness, row identity, and unit norms."""
    if frame["pair_id"].duplicated().any():
        raise ValueError("Embedding table contains duplicate pair IDs")

    expected_rows = len(frame)
    for column in ("input_embedding", "output_embedding", "direction_embedding"):
        matrix = np.asarray(frame[column].tolist(), dtype=np.float32)
        if matrix.shape[0] != expected_rows:
            raise ValueError(f"Unexpected row count in {column}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite values found in {column}")
        norms = np.linalg.norm(matrix, axis=1)
        max_error = float(np.max(np.abs(norms - 1.0)))
        if max_error > tolerance:
            raise ValueError(
                f"{column} is not unit-normalized; maximum norm error={max_error:.3g}"
            )


def git_commit() -> str | None:
    """Return the current Git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def write_run_metadata(
    output_path: Path,
    *,
    model_name: str,
    frame: pd.DataFrame,
    source_manifest: Path,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> None:
    """Save the settings and software versions required for the paper write-up."""
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_manifest": str(source_manifest),
        "row_count": len(frame),
        "concept_count": int(frame["concept"].nunique()),
        "task_count": int(frame["task_id"].nunique()),
        "split_counts": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
        "model_name": model_name,
        "model_checkpoint": MODEL_SPECS[model_name]["checkpoint"],
        "feature_type": MODEL_SPECS[model_name]["feature_type"],
        "embedding_dim": int(frame["embedding_dim"].iloc[0]),
        "renderer": asdict(BASELINE_CONFIG),
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    sidecar = output_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/manifest.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to outputs/<model>_embeddings.parquet.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', or an explicit device such as cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="float32 is the recommended paper baseline.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    output_path = args.output or Path(f"outputs/{args.model}_embeddings.parquet")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_reproducibility(args.seed)

    manifest = pd.read_parquet(args.manifest)
    frame = extract_embeddings(
        manifest,
        model_name=args.model,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
    )
    validate_saved_vectors(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")
    write_run_metadata(
        output_path,
        model_name=args.model,
        frame=frame,
        source_manifest=args.manifest,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    print(f"Saved embeddings: {output_path}")
    print(f"Saved run metadata: {output_path.with_suffix('.meta.json')}")
    print(f"Rows: {len(frame)}")
    print(f"Embedding dimension: {frame['embedding_dim'].iloc[0]}")


if __name__ == "__main__":
    main()
