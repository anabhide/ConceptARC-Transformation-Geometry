# ConceptARC Transformation Geometry

Phase 1 of a research pipeline testing whether ConceptARC input-to-output
transformations form reusable directions in frozen CLIP and DINOv2 image spaces.

## Phase 1 outputs

The pipeline creates:

- `outputs/manifest.parquet`: validated pair-level ConceptARC data and metadata.
- `outputs/render_inspection.png`: deterministic rendering sanity-check sheet.
- `outputs/render_inspection.renderer.json`: exact renderer settings.
- `outputs/clip_embeddings.parquet`: normalized CLIP input, output, and direction vectors.
- `outputs/clip_embeddings.meta.json`: checkpoint, environment, renderer, and run metadata.
- `outputs/dinov2_embeddings.parquet`: normalized DINOv2 input, output, and direction vectors.
- `outputs/dinov2_embeddings.meta.json`: checkpoint, environment, renderer, and run metadata.

Generated data and model artifacts are intentionally excluded from Git.

## Repository layout

```text
.
├── data/
│   └── ConceptARC/          # official repository, cloned locally and ignored by Git
├── outputs/                 # generated artifacts, ignored by Git
├── src/
│   ├── loader.py
│   ├── renderer.py
│   └── embeddings.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment

Create and activate a Python 3.11 environment:

```bash
conda create -n conceptarc-geometry python=3.11 -y
conda activate conceptarc-geometry
python -m pip install --upgrade pip
```

Install a CUDA-compatible PyTorch build using the method recommended for your
cluster, then verify that CUDA is visible:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Record the fully resolved environment after a successful run:

```bash
python -m pip freeze > environment.lock.txt
```

## Obtain ConceptARC

Clone the official dataset repository without adding it to this Git repository:

```bash
git clone https://github.com/victorvikram/ConceptARC.git data/ConceptARC
```

The scripts use only `data/ConceptARC/corpus`. They do not load the separate
`MinimalTasks` attention-check set.

For stronger provenance, record the exact dataset commit:

```bash
git -C data/ConceptARC rev-parse HEAD
```

## Run Phase 1

All commands assume they are run from the repository root.

### 1. Build and validate the pair-level manifest

```bash
python src/loader.py
```

The command fails if the data do not contain 16 concepts, 160 main tasks, ten
tasks per concept, and 907 total train/test transformations.

### 2. Inspect the baseline renderer

```bash
python src/renderer.py --sample-size 12 --seed 42
```

Open `outputs/render_inspection.png` and manually check that grids are centered,
colors are correct, cell boundaries are visible, and input/output identities are
not reversed.

### 3. Extract CLIP embeddings on a GPU

```bash
python src/embeddings.py \
  --model clip \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

### 4. Extract DINOv2 embeddings on a GPU

```bash
python src/embeddings.py \
  --model dinov2 \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

Reduce `--batch-size` if GPU memory is insufficient. Keep `float32` for the
baseline paper run; lower precision can be examined separately if needed.

## What each embedding row contains

Each row preserves:

- pair, concept, task, split, and within-split pair identity;
- source task path and SHA-256 checksum;
- input and output dimensions and exact grids;
- checkpoint, feature type, embedding dimension, and renderer identifier;
- normalized input embedding;
- normalized output embedding;
- normalized `output - input` direction;
- pre-normalization endpoint norms and the endpoint-difference norm.

This is sufficient for task-level averaging, static endpoint controls,
leave-one-task-out decoding, strict train-to-test decoding, and direction-based
analyses without rerunning either vision model.
