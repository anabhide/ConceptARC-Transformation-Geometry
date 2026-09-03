"""Extract explicit, fixed-length visual-change features from ConceptARC pairs."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MAX_GRID = 30


def parse_grid(value: str) -> np.ndarray:
    arr = np.asarray(json.loads(value), dtype=np.int16)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Grid must be a non-empty 2D array")
    if arr.shape[0] > MAX_GRID or arr.shape[1] > MAX_GRID:
        raise ValueError(f"ARC grid exceeds {MAX_GRID}x{MAX_GRID}: {arr.shape}")
    return arr


def components(grid: np.ndarray, include_zero: bool = False) -> list[dict[str, float]]:
    """Return 4-connected monochromatic components."""
    seen = np.zeros(grid.shape, dtype=bool)
    result: list[dict[str, float]] = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r, c])
            if seen[r, c] or (color == 0 and not include_zero):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr,nc] and int(grid[nr,nc]) == color:
                        seen[nr,nc] = True
                        q.append((nr,nc))
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            h, w = max(rs)-min(rs)+1, max(cs)-min(cs)+1
            result.append({
                "color": color,
                "size": float(len(cells)),
                "height": float(h),
                "width": float(w),
                "fill": float(len(cells)/(h*w)),
            })
    return result


def bbox_nonzero(grid: np.ndarray) -> tuple[float, float, float, float, float]:
    coords = np.argwhere(grid != 0)
    if len(coords) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h, w = int(r1-r0+1), int(c1-c0+1)
    return float(r0), float(c0), float(h), float(w), float(h*w)


def symmetry(grid: np.ndarray) -> tuple[float, float, float, float]:
    h = float(np.array_equal(grid, np.flipud(grid)))
    v = float(np.array_equal(grid, np.fliplr(grid)))
    d = float(grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T))
    a = float(grid.shape[0] == grid.shape[1] and np.array_equal(grid, np.fliplr(np.flipud(grid)).T))
    return h, v, d, a


def crop_relation(big: np.ndarray, small: np.ndarray) -> float:
    if small.shape[0] > big.shape[0] or small.shape[1] > big.shape[1]:
        return 0.0
    for r in range(big.shape[0]-small.shape[0]+1):
        for c in range(big.shape[1]-small.shape[1]+1):
            if np.array_equal(big[r:r+small.shape[0], c:c+small.shape[1]], small):
                return 1.0
    return 0.0


def tile_relation(source: np.ndarray, target: np.ndarray) -> float:
    if target.shape[0] % source.shape[0] or target.shape[1] % source.shape[1]:
        return 0.0
    tiled = np.tile(source, (target.shape[0]//source.shape[0], target.shape[1]//source.shape[1]))
    return float(np.array_equal(tiled, target))



def exact_translation(inp: np.ndarray, out: np.ndarray) -> tuple[float, float, float]:
    """Detect an exact zero-padded translation for equal-shaped grids.

    Returns (match, row_shift/MAX_GRID, col_shift/MAX_GRID). Positive row and
    column shifts move content down and right. The identity shift is excluded
    because it is already represented by the identity relation.
    """
    if inp.shape != out.shape:
        return 0.0, 0.0, 0.0
    rows, cols = inp.shape
    for dr in range(-(rows - 1), rows):
        for dc in range(-(cols - 1), cols):
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(inp)
            src_r0, src_r1 = max(0, -dr), min(rows, rows - dr)
            src_c0, src_c1 = max(0, -dc), min(cols, cols - dc)
            dst_r0, dst_r1 = src_r0 + dr, src_r1 + dr
            dst_c0, dst_c1 = src_c0 + dc, src_c1 + dc
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = inp[src_r0:src_r1, src_c0:src_c1]
            if np.array_equal(shifted, out):
                return 1.0, float(dr / MAX_GRID), float(dc / MAX_GRID)
    return 0.0, 0.0, 0.0

def exact_relations(inp: np.ndarray, out: np.ndarray) -> list[float]:
    candidates = [
        inp,
        np.rot90(inp, 1),
        np.rot90(inp, 2),
        np.rot90(inp, 3),
        np.fliplr(inp),
        np.flipud(inp),
        inp.T,
        np.fliplr(np.flipud(inp)).T,
    ]
    equal_transforms = [float(c.shape == out.shape and np.array_equal(c, out)) for c in candidates]
    return equal_transforms + [
        crop_relation(inp, out), crop_relation(out, inp),
        tile_relation(inp, out), tile_relation(out, inp),
    ]


def profile(grid: np.ndarray) -> list[float]:
    occupied = (grid != 0).astype(np.float64)
    rows = occupied.mean(axis=1).tolist() + [0.0] * (MAX_GRID-grid.shape[0])
    cols = occupied.mean(axis=0).tolist() + [0.0] * (MAX_GRID-grid.shape[1])
    return rows + cols


def one_grid_features(grid: np.ndarray) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    rows, cols = grid.shape
    names += ["rows", "cols", "area", "aspect", "occupied", "occupied_fraction", "unique_nonzero_colors"]
    values += [float(rows), float(cols), float(rows*cols), float(cols/rows), float(np.count_nonzero(grid)), float(np.count_nonzero(grid)/(rows*cols)), float(len(set(grid.ravel())-{0}))]
    counts = np.bincount(grid.ravel(), minlength=10).astype(float)
    for color in range(10):
        names += [f"color_{color}_count", f"color_{color}_fraction"]
        values += [counts[color], counts[color]/(rows*cols)]
    comps = components(grid)
    sizes = np.asarray([c["size"] for c in comps], dtype=float)
    fills = np.asarray([c["fill"] for c in comps], dtype=float)
    names += ["component_count", "component_mean_size", "component_max_size", "component_mean_fill"]
    values += [float(len(comps)), float(sizes.mean()) if len(sizes) else 0.0, float(sizes.max()) if len(sizes) else 0.0, float(fills.mean()) if len(fills) else 0.0]
    per_color = np.bincount([int(c["color"]) for c in comps], minlength=10)
    for color in range(1,10):
        names.append(f"color_{color}_components")
        values.append(float(per_color[color]))
    b = bbox_nonzero(grid)
    names += ["bbox_top", "bbox_left", "bbox_height", "bbox_width", "bbox_area"]
    values += list(b)
    names += ["sym_horizontal", "sym_vertical", "sym_diagonal", "sym_antidiagonal"]
    values += list(symmetry(grid))
    p = profile(grid)
    names += [f"row_occ_{i}" for i in range(MAX_GRID)] + [f"col_occ_{i}" for i in range(MAX_GRID)]
    values += p
    return names, values


def pair_features(inp: np.ndarray, out: np.ndarray) -> tuple[list[str], np.ndarray]:
    in_names, in_vals = one_grid_features(inp)
    out_names, out_vals = one_grid_features(out)
    names = [f"input_{x}" for x in in_names] + [f"output_{x}" for x in out_names]
    values = in_vals + out_vals
    # Differences for all aligned scalar grid features.
    names += [f"delta_{x}" for x in in_names]
    values += (np.asarray(out_vals)-np.asarray(in_vals)).tolist()

    same_shape = inp.shape == out.shape
    if same_shape:
        changed = inp != out
        changed_count = float(changed.sum())
        unchanged_count = float((~changed).sum())
        changed_fraction = changed_count / inp.size
        color_transition = np.zeros((10,10), dtype=float)
        for a,b in zip(inp.ravel(), out.ravel()):
            color_transition[int(a),int(b)] += 1
        color_transition /= inp.size
    else:
        changed_count = unchanged_count = changed_fraction = 0.0
        color_transition = np.zeros((10,10), dtype=float)

    names += ["same_shape", "changed_cell_count", "unchanged_cell_count", "changed_cell_fraction"]
    values += [float(same_shape), changed_count, unchanged_count, changed_fraction]
    names += [f"transition_{a}_{b}" for a in range(10) for b in range(10)]
    values += color_transition.ravel().tolist()
    relation_names = [
        "identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud",
        "transpose", "anti_transpose", "output_is_crop_of_input",
        "input_is_crop_of_output", "output_is_tile_of_input",
        "input_is_tile_of_output",
    ]
    names += relation_names
    values += exact_relations(inp, out)
    translation_match, translation_dr, translation_dc = exact_translation(inp, out)
    names += ["exact_translation", "translation_row_shift", "translation_col_shift"]
    values += [translation_match, translation_dr, translation_dc]
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Visual features contain non-finite values")
    return names, array


def build_features(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected_names: list[str] | None = None
    for record in manifest.itertuples(index=False):
        names, values = pair_features(parse_grid(record.input_grid_json), parse_grid(record.output_grid_json))
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError("Feature schema changed across rows")
        rows.append({
            "pair_id": record.pair_id,
            "concept": record.concept,
            "task_id": record.task_id,
            "split": record.split,
            "pair_index": int(record.pair_index),
            "visual_feature_names": expected_names,
            "visual_features": values.astype(np.float32).tolist(),
            "visual_feature_dim": len(values),
        })
    result = pd.DataFrame(rows)
    if result["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visual_features.parquet"))
    args = parser.parse_args()
    manifest = pd.read_parquet(args.manifest)
    result = build_features(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    print(f"Saved: {args.output}")
    print(f"Rows: {len(result)}")
    print(f"Feature dimension: {result['visual_feature_dim'].iloc[0]}")


if __name__ == "__main__":
    main()