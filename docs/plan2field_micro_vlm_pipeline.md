# Buili Plan2Field Micro-VLM Pipeline

## Production Role

Buili uses a lightweight domain VLM as the primary semantic verifier for the 2D drawing to 3D pipeline. The deterministic geometry path still owns wall topology, coordinate transforms, and final mesh construction; the Micro-VLM reads plan patches to verify and correct objects, doors, and windows before the scene is compiled.

## Model

- Model family: `Buili Plan2Field Micro-VLM`
- Architecture: CNN patch encoder + task token + 2-layer Transformer encoder + class/objectness/box heads
- Artifact: `data/artifacts/plan2field_micro_vlm/micro_vlm.pt`
- Size: about 2.4 MB
- Classes: door, window, bathtub, cabinet run, ceiling light, duplex outlet, fixture tag, sink, shower, smoke detector, switch, toilet, washer/dryer, water heater, background
- Runtime path: `services/api/buili/spatial/micro_vlm.py`

## GPU Policy

All training and GPU inference jobs must run with only GPU 7 visible:

```bash
CUDA_VISIBLE_DEVICES=7 conda run -n cjh_buili python ml/train_plan2field_micro_vlm.py
```

The training script also calls `force_gpu_7()` and `assert_gpu_7()` before CUDA work.

## Training Data

Dataset generation uses public floor-plan crops and existing Buili semantic scene labels:

```bash
conda run -n cjh_buili python ml/build_plan2field_micro_vlm_dataset.py \
  --out-dir data/processed/plan2field_micro_vlm \
  --patch-size 224 \
  --negatives-per-scene 12
```

Current dataset manifest:

- Rows: 1,302
- Train rows: 1,119
- Eval rows: 183
- Source scenes: 13
- Dataset hash: `b6e1ac824c1cdbd40f83666aa4db4e3d88d9b3966b3d0b23529a63dbeb30e873`

## Latest Training Run

```bash
CUDA_VISIBLE_DEVICES=7 conda run -n cjh_buili python ml/train_plan2field_micro_vlm.py \
  --dataset data/processed/plan2field_micro_vlm/dataset.jsonl \
  --out-dir data/artifacts/plan2field_micro_vlm \
  --epochs 14 \
  --batch-size 96 \
  --image-size 128 \
  --dim 128 \
  --depth 2 \
  --heads 4
```

Result on GPU 7 / NVIDIA RTX A6000:

- Training time: 26.37 s
- Eval class accuracy: 85.79%
- Eval positive class accuracy: 83.65%
- Eval positive bbox L1: 0.000906

## Runtime Pipeline

1. Render PDF page to image and crop the floor plan.
2. Extract wall segments with OpenCV morphology and snap walls to dark source evidence.
3. Read room/dimension text via embedded PDF text or EasyOCR fallback.
4. Generate initial geometry proposals from OCR, circles, connected components, door gaps, and window markers.
5. Run Micro-VLM on proposal-centered patches to verify or correct objects/openings.
6. Run a bounded low-resolution Micro-VLM grid pass for missing candidates.
7. Dedupe with Micro-VLM-verified candidates ranked first.
8. Compile `SemanticScene`, evaluate source alignment, and render lightweight isometric 3D.

## Benchmark

Command:

```bash
CUDA_VISIBLE_DEVICES=7 BUILI_PLAN2FIELD_MICRO_VLM=data/artifacts/plan2field_micro_vlm/micro_vlm.pt \
  conda run -n cjh_buili python ml/benchmark_plan2field3d_full.py \
  --source-pdf data/sources/plan2field3d_house/maricopa_sample_floor_plan.pdf \
  --output-dir docs/plan2field3d_house_conversion/micro_vlm_timing
```

Measured result:

- OCR-included PDF to 3D: 6.46 s
- Micro-VLM runtime: 0.33 s
- Micro-VLM raw predictions/proposals: 63
- VLM-verified objects: 39
- VLM-verified openings: 15
- VLM corrections: 7 objects
- Final scene: 48 walls, 15 openings, 17 objects, 16 labels, 5 dimensions
- Wall-to-source mean distance: 0.036 px
- Alignment QA: passed
