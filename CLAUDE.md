# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMDetection is an open-source object detection toolbox built on PyTorch (1.8+), part of the OpenMMLab ecosystem. It depends on **MMEngine** (training framework, `>=0.7.1,<1.0`) and **MMCV** (CV utilities, `>=2.0.0rc4,<2.2.0`). Supports object detection, instance/panoptic segmentation, semi-supervised detection, and MOT.

## Conda Editable Install (known pitfalls and working recipe)

The editable install has several compatibility issues with the OpenMMLab ecosystem.
Follow these steps in order to avoid them.

### 1. Activate conda environment

```bash
conda activate openmmlab
```

### 2. Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

If it fails with `undefined symbol: iJIT_NotifyEvent` or similar, reinstall:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall
```

### 3. Pin setuptools to a compatible version

Conda ships setuptools 60.x which is too old for PEP 660 editable installs.
Latest 82.x removes `pkg_resources` which `mim` depends on.
Install a version that supports both (64–68):

```bash
pip install 'setuptools>=64,<69'
```

### 4. Install build dependencies

```bash
pip install -r requirements/build.txt
```

### 5. Install mmdet with optional deps

`--no-build-isolation` is required because `setup.py` imports `torch` at
the top level, and torch is not available in pip's isolated build env.

```bash
pip install --no-build-isolation -e . -r requirements/tracking.txt
```

### 6. Verify

```bash
python -c "import mmdet; print(mmdet.__version__)"
```

### 7. Download model weights

`mim download` fails under editable installs (path resolution issue).
Use `wget` instead — look up the URL in `configs/<model>/metafile.yml`:

```bash
wget <Weights URL from metafile.yml>
```

### 8. Test inference

Always pass the **full config path** (short model names don't work
with editable installs):

```bash
python demo/image_demo.py demo/demo.jpg \
    configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py \
    --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
    --device cpu
```

### Known limitations of editable install

| Feature | Status | Workaround |
|---------|--------|------------|
| `import mmdet`, training, inference | Works | — |
| `mim download` / `mim search` | Broken | Use `wget` with URL from metafile.yml |
| Short model names (e.g. `rtmdet-s`) | Broken | Use full config path |
| `DetInferencer` with short names | Broken | Use `init_detector` + `inference_detector` or full config path |

### Why these issues happen

- `setup.py` imports `torch` at module level → requires `--no-build-isolation`
- PEP 660 editable installs skip `add_mim_extension()` in `setup.py`, so
  `model-index.yml` and `configs/` symlinks are never created inside the
  package directory. `mim` and `mmengine.BaseInferencer` look for these
  under `site-packages/mmdet/`, which doesn't exist in editable mode.

Dependencies are split across `requirements/`:
- `runtime.txt` — core deps (numpy, pycocotools, matplotlib)
- `tests.txt` — pytest, xdoctest, parameterized, flake8, isort, yapf
- `build.txt` — cython, numpy
- `optional.txt`, `mminstall.txt`, `tracking.txt`, `multimodal.txt`

## Common Commands

```bash
# Training
python tools/train.py <config> --work-dir <dir>
python tools/train.py <config> --amp           # automatic mixed precision
python tools/train.py <config> --auto-scale-lr  # scale LR by batch size
python tools/train.py <config> --resume         # auto-resume from latest checkpoint
bash tools/dist_train.sh <config> <num_gpus>    # multi-GPU

# Testing / inference
python tools/test.py <config> <checkpoint>
python tools/test.py <config> <checkpoint> --tta   # test-time augmentation
bash tools/dist_test.sh <config> <checkpoint> <num_gpus>

# Running all tests
pytest tests/

# Run a specific test file
pytest tests/test_models/test_detectors/test_faster_rcnn.py

# Run a single test function
pytest tests/test_models/test_detectors/test_faster_rcnn.py -k "test_forward"

# Linting (no single lint command; individual tools)
isort mmdet/ tests/ --check-only
yapf -r -d mmdet/ tests/
flake8 mmdet/ tests/
codespell mmdet/ tests/ tools/ configs/ projects/
```

## High-Level Architecture

### Config-driven design

Everything is driven by Python config files in `configs/`. A config composes `_base_` files for dataset, model, and schedule, then overrides specific keys. Configs are parsed by `mmengine.Config` and build the full model/dataset/runner tree.

```
configs/
  _base_/
    datasets/coco_detection.py   # train/val dataloader + evaluator
    models/faster-rcnn_r50_fpn.py     # model architecture + train/test cfg
    schedules/schedule_1x.py     # optimizer, param scheduler, loop settings
    default_runtime.py           # hooks, logging, visualization, env
  faster_rcnn/                   # concrete configs inheriting from _base_
```

Config dict keys correspond directly to registry components: `model.backbone` -> `MODELS.build()`, `train_dataloader.dataset` -> `DATASETS.build()`, etc.

### Registry system (`mmdet/registry.py`)

MMDetection creates 17+ child registries rooted in MMEngine's registry tree. Each maps a string type name (e.g., `'FasterRCNN'`) to a class, discovered automatically from the declared `locations`. Key registries:

| Registry | Purpose | Location |
|----------|---------|----------|
| `MODELS` | All nn.Module subclasses | `mmdet.models` |
| `DATASETS` | Dataset classes | `mmdet.datasets` |
| `TRANSFORMS` | Data augmentation pipeline stages | `mmdet.datasets.transforms` |
| `TASK_UTILS` | Anchor generators, bbox coders, assigners, samplers | `mmdet.models` |
| `METRICS` | Evaluation metrics | `mmdet.evaluation` |
| `HOOKS` | Training hooks | `mmdet.engine.hooks` |
| `RUNNERS` | Runner classes | `mmdet.engine.runner` |
| `VISUALIZERS` | Visualization | `mmdet.visualization` |

Components are registered with `@REGISTRY.register_module()` decorators.

### Model architecture (`mmdet/models/`)

Detector inheritance chain:
```
BaseModel (mmengine)
  └── BaseDetector (mmdet/models/detectors/base.py)
        ├── SingleStageDetector  — backbone + neck + bbox_head (RetinaNet, FCOS, YOLOX, RTMDet)
        ├── TwoStageDetector     — backbone + neck + rpn_head + roi_head (Faster R-CNN, Mask R-CNN, Cascade R-CNN)
        ├── DetectionTransformer — DETR-family (DETR, Deformable DETR, DINO, DDQ)
        ├── Mask2Former, MaskFormer — mask classification architectures
        └── SemiBaseDetector     — SoftTeacher for semi-supervised detection
```

Standard building blocks under `mmdet/models/`:
- **`backbones/`** — ResNet, ResNeXt, Swin, ConvNeXt, HRNet, PVT, EfficientNet, etc.
- **`necks/`** — FPN, PAFPN, NAS-FPN, DyHead, ChannelMapper, etc.
- **`dense_heads/`** — Single-stage detection heads (RetinaHead, FCOSHead, RPNHead, GFLHead, ATSSHead, RTMDetHead, DETRHead, DINOHead, etc.)
- **`roi_heads/`** — Two-stage RoI heads with `bbox_heads/`, `mask_heads/`, `roi_extractors/`, `shared_heads/`
- **`losses/`** — Classification and regression losses (FocalLoss, IoULoss variants, L1Loss, etc.)
- **`task_modules/`** — `assigners/` (MaxIoU, Hungarian, ATSS), `samplers/`, `coders/` (DeltaXYWH, DistancePoint), `prior_generators/`
- **`data_preprocessors/`** — Input normalization and padding (handled before backbone)
- **`language_models/`** — Text encoders for open-vocabulary models (Grounding DINO, GLIP)

A model's `forward()` method dispatches on mode: `'loss'` (returns loss dict), `'predict'` (returns list of `DetDataSample`), `'tensor'` (raw tensor output).

### Data flow (`mmdet/datasets/`, `mmdet/structures/`)

```
Image + annotations
  → transforms pipeline (LoadImageFromFile → Resize → RandomFlip → PackDetInputs)
  → DetDataSample (wraps gt_instances as InstanceData, gt_panoptic_seg as PixelData)
  → collate_fn → batch of (tensor, DetDataSample list)
  → model
```

- **`DetDataSample`** (`mmdet/structures/det_data_sample.py`) — container for `gt_instances`, `pred_instances`, `proposals` (all `InstanceData`), `gt_panoptic_seg`/`pred_panoptic_seg` (all `PixelData`). Contains bboxes, labels, masks, scores.
- **`TrackDataSample`** — extends `DetDataSample` for video/MOT, adds `pred_track_instances`.
- **`ReIDDataSample`** — for re-identification tasks.

### Datasets (`mmdet/datasets/`)

`BaseDetDataset` extends `mmengine.BaseDataset`. Concrete datasets: `CocoDataset`, `LVISDataset`, `Objects365Dataset`, `CrowdHumanDataset`, `RefCocoDataset`, `MOTChallengeDataset`, etc. Video datasets extend `BaseVideoDataset`.

### Training/Testing entry points (`tools/train.py`, `tools/test.py`)

Both follow the same pattern:
1. Parse CLI args (config path, work-dir, --amp, --cfg-options overrides)
2. Load config with `Config.fromfile()`, merge in CLI overrides
3. Build `Runner` via `Runner.from_cfg(cfg)` (or `RUNNERS.build(cfg)` for custom runner types)
4. Call `runner.train()` or `runner.test()`

The Runner (from MMEngine) orchestrates the loop (epoch/iter-based), hooks, optimizer, logging, and checkpointing.

### Evaluation (`mmdet/evaluation/`)

- **`CocoMetric`** — COCO-style AP/AR evaluation
- **`LVISMetric`**, **`CityscapesMetric`**, **`OpenImagesMetric`**, **`RefSegMetric`**, **`YouTubeVISMetric`**
- **`functional/`** — lower-level eval operations (mean AP, recall, panoptic quality)

### APIs (`mmdet/apis/`)

High-level convenience functions:
- `init_detector(config, checkpoint)` — build and load a model
- `inference_detector(model, img)` — run inference on a single image
- `DetInferencer` — unified inference interface supporting file/dir/video/camera/webcam

### Test utilities (`mmdet/testing/`)

- `demo_mm_inputs()` — creates fake input tensors and data samples for unit tests
- `get_detector_cfg(cfg_file)` — loads a full config from config name
- `random_boxes()` — generates random bboxes for testing
- `FastStopTrainingHook` — hook to stop training early in tests

## Code Style

- Line length: 79 chars (isort, yapf configured in `setup.cfg`)
- Import order: standard library → third-party → local (isort enforces)
- Formatter: yapf (pep8-based style defined in `setup.cfg`)
- Docstring tests: xdoctest (configured in `pytest.ini`)
- Numpy-style docstrings
- Copyright header on every file: `# Copyright (c) OpenMMLab. All rights reserved.`

## Key Patterns

- **All model components** are registry-registered and built from config dicts via `MODELS.build(cfg_dict)`. The config dict must have a `type` key matching a registered module.
- **Config inheritance**: Use `_base_` lists in config files to compose base configs. Override specific keys in the child config.
- **`train_cfg`/`test_cfg`** dicts are passed through the model tree — used for assigner/sampler/NMS configuration at each stage.
- **`init_cfg`** pattern: models can specify weight initialization via config dict (e.g., `dict(type='Pretrained', checkpoint='torchvision://resnet50')`).
