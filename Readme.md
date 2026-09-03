# ConceptARC Transformation Geometry

Code for reproducing the analyses reported in the paper:

> **Representational Structure of ConceptARC Transformations in Pretrained Vision Models**

The code evaluates whether human-defined ConceptARC transformation families are reflected in reusable directions between input and output representations of frozen pretrained vision models.

The experiments use the main ConceptARC corpus (160 tasks from 16 concept families; 907 input-output pairs) and evaluate CLIP and DINOv2 using:

* input-to-output transformation directions;
* task-level organization by ConceptARC family;
* leave-one-task-out (LOTO) concept decoding;
* strict train-to-test decoding;
* static input, output, and input-output concatenation controls;
* a 446-dimensional hand-engineered visual-change representation;
* Count-family ablation;
* rendering-robustness analyses.


## Repository structure

```text
.
├── data/
│   └── ConceptARC/          # official ConceptARC repository, cloned locally
├── outputs/                 # generated artifacts
├── src/
│   ├── loader.py            # build and validate the pair-level manifest
│   ├── renderer.py          # deterministic grid rendering
│   ├── embeddings.py        # CLIP/DINOv2 embedding extraction
│   ├── analysis.py          # organization and concept decoding analyses
│   ├── visual_features.py   # hand-engineered visual-change features
│   ├── controls.py          # endpoint, visual-feature, and Count controls
│   └── summarize_results.py # collect analysis results
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment

The experiments were run with Python 3.11.

Create an environment:

```bash
conda create -n conceptarc-geometry python=3.11 -y
conda activate conceptarc-geometry
python -m pip install --upgrade pip
```

Install a PyTorch build compatible with the available hardware. Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

For the exact paper environment, use the versions recorded from the environment used for the final reported runs.

## Dataset

The experiments use the main ConceptARC corpus from the official repository:

```bash
git clone https://github.com/victorvikram/ConceptARC.git data/ConceptARC
```

The scripts use only:

```text
data/ConceptARC/corpus
```

and do not use the separate `MinimalTasks` attention-check set.

For exact dataset provenance, record the commit of the downloaded ConceptARC repository:

```bash
git -C data/ConceptARC rev-parse HEAD
```

The paper experiments use the corpus containing 16 concepts, 160 tasks, and 907 input--output pairs. `loader.py` validates these counts before continuing.

## Reproducing the paper analyses

All commands below should be run from the repository root.

### 1. Build the ConceptARC manifest

```bash
python src/loader.py
```

This creates:

```text
outputs/manifest.parquet
outputs/manifest.meta.json
```

The manifest contains the validated grids, task/concept identifiers, train/test split, and dataset checksums.

### 2. Inspect the renderer

Run:

```bash
python src/renderer.py --sample-size 12 --seed 42
```

This creates a rendering inspection sheet under `outputs/`.

The baseline renderer uses:

* 224 $\times$ 224 RGB images;
* the ARC color palette;
* 14-pixel outer padding;
* a near-white canvas;
* one-pixel grid lines.

The same deterministic renderer is applied to input and output grids.

### 3. Extract baseline CLIP embeddings

```bash
python src/embeddings.py \
  --model clip \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

This creates:

```text
outputs/clip_embeddings.parquet
outputs/clip_embeddings.meta.json
```

### 4. Extract baseline DINOv2 embeddings

```bash
python src/embeddings.py \
  --model dinov2 \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

This creates:

```text
outputs/dinov2_embeddings.parquet
outputs/dinov2_embeddings.meta.json
```

The models are loaded from:

```text
openai/clip-vit-base-patch32
facebook/dinov2-base
```

and are used without fine-tuning on ConceptARC.

### 5. Run the main organization and decoding analyses

CLIP:

```bash
python src/analysis.py \
  --model clip \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

DINOv2:

```bash
python src/analysis.py \
  --model dinov2 \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

These analyses produce:

* task-level transformation directions;
* within- and between-concept cosine similarity;
* the organization effect;
* the one-sided permutation test;
* LOTO predictions and bootstrap confidence intervals;
* strict train-to-test predictions and bootstrap confidence intervals;
* per-concept LOTO accuracy.

### 6. Build the hand-engineered visual-change representation

```bash
python src/visual_features.py
```

This creates:

```text
outputs/visual_features.parquet
```

The representation contains 446 scalar features describing explicit visual properties and relations between the input and output grids.

### 7. Run endpoint, visual-feature, and Count controls

CLIP:

```bash
python src/controls.py \
  --model clip \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

DINOv2:

```bash
python src/controls.py \
  --model dinov2 \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

These analyses reproduce the static endpoint controls, the rich visual-change control, and the Count ablation reported in the paper.

### 8. Run rendering-robustness analyses

The renderer provides four robustness conditions:

```text
no_grid
white_canvas
padding0
colorperm42
```

For each encoder and condition, first extract the embeddings.

For CLIP:

```bash
python src/embeddings.py \
  --model clip \
  --renderer-condition no_grid \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

```bash
python src/embeddings.py \
  --model clip \
  --renderer-condition white_canvas \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

```bash
python src/embeddings.py \
  --model clip \
  --renderer-condition padding0 \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

```bash
python src/embeddings.py \
  --model clip \
  --renderer-condition colorperm42 \
  --device cuda \
  --dtype float32 \
  --batch-size 32 \
  --seed 42
```

Run the corresponding analyses:

```bash
python src/analysis.py \
  --model clip \
  --embeddings outputs/rendering/clip_no_grid_embeddings.parquet \
  --output-dir outputs/rendering_analysis/clip_no_grid \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

```bash
python src/analysis.py \
  --model clip \
  --embeddings outputs/rendering/clip_white_canvas_embeddings.parquet \
  --output-dir outputs/rendering_analysis/clip_white_canvas \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

```bash
python src/analysis.py \
  --model clip \
  --embeddings outputs/rendering/clip_padding0_embeddings.parquet \
  --output-dir outputs/rendering_analysis/clip_padding0 \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

```bash
python src/analysis.py \
  --model clip \
  --embeddings outputs/rendering/clip_colorperm42_embeddings.parquet \
  --output-dir outputs/rendering_analysis/clip_colorperm42 \
  --permutations 9999 \
  --bootstraps 10000 \
  --seed 42
```

Repeat the same four commands with `--model dinov2` and the corresponding DINOv2 embedding paths.

### 9. Collect the generated result tables

After the baseline, controls, and rendering analyses have been run:

```bash
python src/summarize_results.py
```

This collects the generated analysis summaries into CSV files under:

```text
outputs/final/
```

## Reproducibility settings

The main reported analyses use:

```text
Random seed:               42
Permutation replicates:    9,999
Bootstrap replicates:     10,000
CLIP dimension:              512
DINOv2 dimension:             768
Valid pair directions:      883
Total input-output pairs:   907
Concept families:             16
Tasks:                       160
```

Accuracy confidence intervals are obtained by bootstrap resampling at the task level rather than the individual-pair level.

The organization test uses a one-sided permutation test that shuffles concept labels across tasks while preserving the number of tasks assigned to each concept.

## Generated outputs

The main generated files include:

```text
outputs/manifest.parquet
outputs/clip_embeddings.parquet
outputs/dinov2_embeddings.parquet
outputs/visual_features.parquet
outputs/phase2/
outputs/phase3/
outputs/rendering/
outputs/rendering_analysis/
outputs/final/
```

Generated outputs are intentionally excluded from version control. They can be recreated using the commands above.

## Third-party assets

This repository does not redistribute the ConceptARC dataset or pretrained model weights.

ConceptARC should be obtained from its official repository:

```text
https://github.com/victorvikram/ConceptARC
```

The pretrained models are downloaded through their public model repositories:

```text
openai/clip-vit-base-patch32
facebook/dinov2-base
```

Users should follow the licenses and terms associated with these third-party assets.


## Reproducibility and hardware

The embedding extraction stage uses pretrained vision models and is intended to run on a GPU. CPU execution is supported by the scripts but may require substantially more time. No generated outputs, model weights, cluster-specific job scripts, or execution logs are included in the repository.
