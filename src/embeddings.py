"""Extract and cache CLIP or DINOv2 endpoint and transformation vectors.

For every ConceptARC input-output pair, this script:
1. renders both grids with the baseline renderer;
2. extracts frozen image features;
3. L2-normalizes both endpoint features;
4. computes output minus input;
5. L2-normalizes the difference when it is non-zero.

An unchanged input-output pair has no mathematically defined unit direction.
Such rows are preserved with a zero direction vector and direction_valid=False.
Downstream direction analyses must exclude those rows and report their count.
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
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel

from renderer import get_render_config, parse_grid, render_grid

MODEL_SPECS: dict[str, dict[str, Any]] = {
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
    """Set random seeds and deterministic inference preferences."""
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
    """Resolve the requested inference precision."""
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
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Load one frozen encoder and return a tensor-valued image encoder.

    Calling CLIPModel.get_image_features is version-sensitive in recent
    Transformers releases.  We therefore explicitly run the vision tower and
    apply CLIP's learned visual projection, which is the documented projected
    image embedding and always returns a tensor.
    """
    spec = MODEL_SPECS[model_name]
    checkpoint = spec["checkpoint"]
    processor = AutoImageProcessor.from_pretrained(checkpoint)

    if model_name == "clip":
        model = CLIPModel.from_pretrained(checkpoint)

        def encode(pixel_values: torch.Tensor) -> torch.Tensor:
            vision_output = model.vision_model(
                pixel_values=pixel_values, return_dict=True
            )
            pooled = vision_output.pooler_output
            if pooled is None:
                raise RuntimeError("CLIP vision model returned no pooler_output")
            return model.visual_projection(pooled)

    elif model_name == "dinov2":
        model = AutoModel.from_pretrained(checkpoint)

        def encode(pixel_values: torch.Tensor) -> torch.Tensor:
            outputs = model(pixel_values=pixel_values, return_dict=True)
            return outputs.last_hidden_state[:, 0, :]

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return processor, model, encode


def batched(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield successive batches from a list."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_images(
    images: list[Image.Image],
    *,
    processor: Any,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode images and return unit vectors plus their original norms."""
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

    with torch.inference_mode():
        raw_features = encode_fn(pixel_values).float()
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
    direction_eps: float,
    render_config: Any,
) -> pd.DataFrame:
    """Render all pairs and compute normalized endpoints and directions."""
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
    if manifest["pair_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate pair IDs")
    if direction_eps <= 0:
        raise ValueError("direction_eps must be positive")

    processor, model, encode_fn = load_model(model_name, device, dtype)
    records = manifest.to_dict(orient="records")
    output_rows: list[dict[str, Any]] = []

    print(
        f"Loaded {model_name} checkpoint {MODEL_SPECS[model_name]['checkpoint']} "
        f"on {device} with {dtype}."
    )

    progress = tqdm(total=len(records), desc=f"Extracting {model_name}")
    for batch_records in batched(records, batch_size):
        input_grids = [parse_grid(record["input_grid_json"]) for record in batch_records]
        output_grids = [parse_grid(record["output_grid_json"]) for record in batch_records]

        input_images = [render_grid(grid, render_config) for grid in input_grids]
        output_images = [render_grid(grid, render_config) for grid in output_grids]

        input_features, input_raw_norms = encode_images(
            input_images,
            processor=processor,
            encode_fn=encode_fn,
            device=device,
            dtype=dtype,
        )
        output_features, output_raw_norms = encode_images(
            output_images,
            processor=processor,
            encode_fn=encode_fn,
            device=device,
            dtype=dtype,
        )

        differences = output_features - input_features
        difference_norms = np.linalg.norm(differences, axis=1)
        direction_valid = difference_norms > direction_eps

        directions = np.zeros_like(differences, dtype=np.float32)
        directions[direction_valid] = (
            differences[direction_valid] / difference_norms[direction_valid, None]
        ).astype(np.float32, copy=False)

        for index, record in enumerate(batch_records):
            grids_identical = input_grids[index] == output_grids[index]
            output_rows.append(
                {
                    **record,
                    "model_name": model_name,
                    "model_checkpoint": MODEL_SPECS[model_name]["checkpoint"],
                    "feature_type": MODEL_SPECS[model_name]["feature_type"],
                    "embedding_dim": int(input_features.shape[1]),
                    "renderer_id": render_config.renderer_id,
                    "input_embedding": input_features[index].tolist(),
                    "output_embedding": output_features[index].tolist(),
                    "direction_embedding": directions[index].tolist(),
                    "direction_valid": bool(direction_valid[index]),
                    "direction_status": (
                        "valid" if direction_valid[index] else "undefined_zero_difference"
                    ),
                    "grids_identical": bool(grids_identical),
                    "input_raw_norm": float(input_raw_norms[index]),
                    "output_raw_norm": float(output_raw_norms[index]),
                    "endpoint_cosine": float(
                        np.dot(input_features[index], output_features[index])
                    ),
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
    """Validate dimensions, finite values, identities, and vector norms."""
    if frame.empty:
        raise ValueError("Embedding table is empty")
    if frame["pair_id"].duplicated().any():
        raise ValueError("Embedding table contains duplicate pair IDs")

    expected_rows = len(frame)
    endpoint_columns = ("input_embedding", "output_embedding")
    for column in endpoint_columns:
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

    directions = np.asarray(frame["direction_embedding"].tolist(), dtype=np.float32)
    if not np.isfinite(directions).all():
        raise ValueError("Non-finite values found in direction_embedding")

    valid = frame["direction_valid"].to_numpy(dtype=bool)
    direction_norms = np.linalg.norm(directions, axis=1)
    if valid.any():
        max_valid_error = float(np.max(np.abs(direction_norms[valid] - 1.0)))
        if max_valid_error > tolerance:
            raise ValueError(
                "Valid directions are not unit-normalized; "
                f"maximum norm error={max_valid_error:.3g}"
            )
    if (~valid).any():
        max_invalid_norm = float(np.max(direction_norms[~valid]))
        if max_invalid_norm > tolerance:
            raise ValueError(
                "Invalid directions must be stored as zero vectors; "
                f"maximum norm={max_invalid_norm:.3g}"
            )

    inconsistent = frame.loc[
        frame["grids_identical"] & frame["direction_valid"], "pair_id"
    ].tolist()
    if inconsistent:
        raise ValueError(
            "Identical grids unexpectedly produced valid directions: "
            + ", ".join(inconsistent)
        )


def git_commit() -> str | None:
    """Return the current repository Git commit when available."""
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
    direction_eps: float,
    render_config: Any,
    renderer_condition: str,
) -> None:
    """Save provenance and software details for the paper write-up."""
    invalid_rows = frame.loc[~frame["direction_valid"]]
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
        "renderer_condition": renderer_condition,
        "renderer": asdict(render_config),
        "direction_definition": "normalize(normalize(output)-normalize(input))",
        "direction_eps": direction_eps,
        "valid_direction_count": int(frame["direction_valid"].sum()),
        "invalid_direction_count": int((~frame["direction_valid"]).sum()),
        "invalid_direction_pair_ids": invalid_rows["pair_id"].tolist(),
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
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
    parser.add_argument(
        "--direction-eps",
        type=float,
        default=1e-8,
        help="Differences at or below this norm are marked as undefined directions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--renderer-condition",
        choices=["baseline", "no_grid", "white_canvas", "padding0", "colorperm42"],
        default="baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: outputs/<model>_embeddings.parquet for baseline, or outputs/rendering/<model>_<condition>_embeddings.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    render_config = get_render_config(args.renderer_condition)
    if args.output is not None:
        output_path = args.output
    elif args.renderer_condition == "baseline":
        output_path = Path(f"outputs/{args.model}_embeddings.parquet")
    else:
        output_path = Path(
            f"outputs/rendering/{args.model}_{args.renderer_condition}_embeddings.parquet"
        )
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
        direction_eps=args.direction_eps,
        render_config=render_config,
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
        direction_eps=args.direction_eps,
        render_config=render_config,
        renderer_condition=args.renderer_condition,
    )

    invalid = frame.loc[~frame["direction_valid"], "pair_id"].tolist()
    print(f"Saved embeddings: {output_path}")
    print(f"Saved run metadata: {output_path.with_suffix('.meta.json')}")
    print(f"Rows: {len(frame)}")
    print(f"Embedding dimension: {frame['embedding_dim'].iloc[0]}")
    print(f"Valid directions: {int(frame['direction_valid'].sum())}")
    print(f"Undefined directions: {len(invalid)}")
    if invalid:
        print("Undefined direction pair IDs:")
        for pair_id in invalid:
            print(f"  - {pair_id}")


if __name__ == "__main__":
    main()