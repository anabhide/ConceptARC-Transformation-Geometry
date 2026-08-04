"""Deterministically render ARC grids and create a manual inspection sheet.

The baseline renderer uses a 224x224 RGB canvas, 14-pixel outer padding, a
near-white canvas, standard ARC colors, and one-pixel grid lines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Common RGB rendering of the ARC palette.
ARC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # black
    1: (0, 116, 217),    # blue
    2: (255, 65, 54),    # red
    3: (46, 204, 64),    # green
    4: (255, 220, 0),    # yellow
    5: (170, 170, 170),  # gray
    6: (240, 18, 190),   # magenta
    7: (255, 133, 27),   # orange
    8: (127, 219, 255),  # cyan
    9: (135, 12, 37),    # maroon
}


@dataclass(frozen=True)
class RenderConfig:
    """Parameters that fully define one rendering condition."""

    renderer_id: str = "baseline_v1"
    image_size: int = 224
    outer_padding: int = 14
    grid_line_width: int = 1
    canvas_rgb: tuple[int, int, int] = (250, 250, 250)
    grid_line_rgb: tuple[int, int, int] = (205, 205, 205)

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.outer_padding < 0:
            raise ValueError("outer_padding cannot be negative")
        if self.grid_line_width < 0:
            raise ValueError("grid_line_width cannot be negative")
        if 2 * self.outer_padding >= self.image_size:
            raise ValueError("outer_padding leaves no drawable canvas area")


BASELINE_CONFIG = RenderConfig()


def parse_grid(grid_json: str) -> list[list[int]]:
    """Deserialize a grid stored in the manifest."""
    grid = json.loads(grid_json)
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        raise ValueError("Invalid grid JSON in manifest")
    return grid


def _cell_size(rows: int, cols: int, config: RenderConfig) -> int:
    """Choose the largest integer cell size that fits inside the padded canvas."""
    available = config.image_size - 2 * config.outer_padding
    width_for_cells = available - (cols + 1) * config.grid_line_width
    height_for_cells = available - (rows + 1) * config.grid_line_width
    cell_size = min(width_for_cells // cols, height_for_cells // rows)
    if cell_size < 1:
        raise ValueError(
            f"Grid {rows}x{cols} cannot fit renderer '{config.renderer_id}'"
        )
    return cell_size


def render_grid(
    grid: Sequence[Sequence[int]], config: RenderConfig = BASELINE_CONFIG
) -> Image.Image:
    """Render one rectangular ARC grid as an RGB PIL image."""
    config.validate()
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty")

    rows = len(grid)
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("Grid rows must all have equal length")

    cell_size = _cell_size(rows, cols, config)
    grid_width = cols * cell_size + (cols + 1) * config.grid_line_width
    grid_height = rows * cell_size + (rows + 1) * config.grid_line_width

    left = (config.image_size - grid_width) // 2
    top = (config.image_size - grid_height) // 2

    image = Image.new("RGB", (config.image_size, config.image_size), config.canvas_rgb)
    draw = ImageDraw.Draw(image)

    if config.grid_line_width > 0:
        draw.rectangle(
            [left, top, left + grid_width - 1, top + grid_height - 1],
            fill=config.grid_line_rgb,
        )

    for row_index, row in enumerate(grid):
        for col_index, color_index in enumerate(row):
            try:
                fill = ARC_PALETTE[int(color_index)]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ARC color: {color_index}") from exc

            x0 = left + config.grid_line_width + col_index * (
                cell_size + config.grid_line_width
            )
            y0 = top + config.grid_line_width + row_index * (
                cell_size + config.grid_line_width
            )
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=fill,
            )

    return image


def make_inspection_sheet(
    manifest: pd.DataFrame,
    *,
    output_path: Path,
    sample_size: int,
    seed: int,
    config: RenderConfig = BASELINE_CONFIG,
) -> None:
    """Create a deterministic contact sheet of sampled input-output pairs."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(manifest):
        raise ValueError("sample_size exceeds the number of manifest rows")

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(manifest)), sample_size))
    sampled = manifest.iloc[sampled_indices]

    thumb = config.image_size
    label_height = 42
    gap = 12
    pair_width = 2 * thumb + gap
    pair_height = thumb + label_height
    columns = min(3, sample_size)
    rows = math.ceil(sample_size / columns)

    sheet = Image.new(
        "RGB",
        (
            columns * pair_width + (columns + 1) * gap,
            rows * pair_height + (rows + 1) * gap,
        ),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for position, (_, record) in enumerate(sampled.iterrows()):
        row = position // columns
        col = position % columns
        x = gap + col * pair_width
        y = gap + row * pair_height

        input_image = render_grid(parse_grid(record["input_grid_json"]), config)
        output_image = render_grid(parse_grid(record["output_grid_json"]), config)
        sheet.paste(input_image, (x, y + label_height))
        sheet.paste(output_image, (x + thumb + gap, y + label_height))

        label = (
            f"{record['concept']} | {record['task_id'].split('/')[-1]} | "
            f"{record['split']}:{record['pair_index']}"
        )
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        draw.text((x, y + 16), "INPUT", fill=(0, 0, 0), font=font)
        draw.text(
            (x + thumb + gap, y + 16), "OUTPUT", fill=(0, 0, 0), font=font
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)

    config_path = output_path.with_suffix(".renderer.json")
    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/manifest.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/render_inspection.png"),
    )
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest)
    make_inspection_sheet(
        manifest,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Saved inspection sheet: {args.output}")
    print(f"Saved renderer config: {args.output.with_suffix('.renderer.json')}")


if __name__ == "__main__":
    main()
