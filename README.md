# Aerial Building Segmentation

Most segmentation projects stop at the pixel mask. This one goes further: clean vector building footprints in world coordinates, resolution-aware inference that handles SWISSIMAGE tiles at 0.1 m/px without manual intervention, and an evaluation pipeline that versions results by checkpoint hash so different model runs never overwrite each other.

Under the hood: U-Net with ResNet34 encoder and two-phase fine-tuning with differential learning rates, GIS post-processing via Douglas-Peucker simplification and area filtering, georeferenced GeoTIFF output with CRS preserved, async FastAPI service with SQLite job tracking and result caching. MLflow tracks training runs and evaluation results with full lineage between them.

---

## Results

Evaluated on Vienna (held-out validation city, not used during training). The Inria test set labels are not publicly available.

### Pixel-level

| City | IoU (raw) | IoU (clean) | ΔIoU | Dice (raw) | Dice (clean) |
|---|---|---|---|---|---|
| Vienna | 0.7592 | 0.5058 | -0.2534 | 0.8577 | 0.6322 |

> ΔIoU is negative as expected: Douglas-Peucker simplification moves boundary pixels to fit straight edges, which slightly reduces pixel-level overlap. Building-level metrics (below) tell the complete story.

### Building-level (object detection)

| | Precision | Recall | F1 | Mean matched IoU |
|---|---|---|---|---|
| Raw | 0.754 | 0.709 | 0.713 | 0.775 |
| Clean | 0.755 | 0.709 | 0.714 | 0.773 |

Building-level metrics are almost identical between raw and clean — postprocessing improves geometric quality without hurting detection.

**Size-stratified recall:**

| Size | GT buildings | Recall |
|---|---|---|
| Small (<50 m²) | 255 | 0.220 |
| Medium (50–500 m²) | 507 | 0.554 |
| Large (>500 m²) | 667 | 0.951 |

Small building recall is low (0.220) — a known limitation of models trained at 0.3 m/px on 512×512 patches. Large buildings are detected near-perfectly (0.951).

### Threshold analysis

Optimal threshold: **0.5** (F1=0.877, IoU=0.781) — the default is already optimal for this dataset, no tuning needed.

---

## Architecture

```
Raw GeoTIFF tiles (5000×5000 px, 0.3 m/px)
    ↓ patch extraction (512×512, stride 256)
    ↓ albumentations augmentations
    ↓ U-Net (ResNet34 encoder, ImageNet pretrained)
    ↓ two-phase fine-tuning + MLflow tracking
    ↓ GIS post-processing (polygonize → simplify → area filter)
    ↓ REST API (FastAPI + Docker)
    ↓ Azure-ready (Blob Storage + Queue Storage, swap via env vars)
```

### Model

| Component | Choice | Rationale |
|---|---|---|
| Backbone | ResNet34 (ImageNet) | Lightweight, strong spatial features, widely validated |
| Head | U-Net decoder (SMP) | Skip connections preserve fine boundary detail |
| Loss | Dice + BCE (0.5 / 0.5) | Handles class imbalance (buildings ≪ background) |
| Optimizer | AdamW | Weight decay without bias correction issues |
| Scheduler | CosineAnnealingLR (phase 2 only) | Smooth decay after warmup stabilises |

---

## Dataset

**Inria Aerial Image Labeling Dataset** — 180 aerial tiles at 5000×5000 px, 0.3 m/px resolution, binary building / background labels.

| Split | Cities | Tiles |
|---|---|---|
| Train | Austin, Chicago, Kitsap, Tyrol-W | 144 |
| Val | Vienna | 36 |
| Test | Bellingham, Bloomington, Innsbruck, SFO, Tyrol-E | 180 (labels not public) |

**Patch extraction** (`scripts/patch_dataset.py`): tiles are sliced into 512×512 crops with stride 256. Patches with fewer than 1% building pixels are discarded to reduce class imbalance. This yields ~33,000 training patches and ~10,000 validation patches.

---

## Fine-tuning Strategy

Most segmentation tutorials freeze the encoder for the entire training run or train everything from scratch. This project uses a two-phase strategy that's standard in NLP (ULMFiT) and increasingly common in vision:

**Phase 1 — Warmup** (10 epochs): encoder frozen, only decoder trains at `decoder_lr=1e-3`. The decoder is randomly initialised — training it alone first prevents large gradients from destroying the pretrained ImageNet encoder features (catastrophic forgetting).

**Phase 2 — Fine-tuning** (remaining epochs): encoder unfrozen with differential learning rates:
- Encoder: `encoder_lr=1e-4` (10× lower) — adapts slowly to aerial domain
- Decoder: `decoder_lr=1e-3` — continues learning the segmentation mapping

CosineAnnealingLR runs over phase 2 only — the cosine cycle is sized to the fine-tuning period, not the full training run.

MLflow logs encoder and decoder learning rates separately at every epoch, making the two-phase transition visible in the training curves.

---

## GIS Post-processing

Raw model output is a pixel mask — useful for computing metrics, but geometrically imprecise for GIS workflows. Building edges follow raster staircase patterns and small noise blobs appear in low-confidence regions.

The vectorization pipeline (`api/vectorize.py`) converts the mask to clean building polygons:

1. **Polygonization** — OpenCV contour extraction with two-level hierarchy (captures building courtyards as holes)
2. **Douglas-Peucker simplification** — fits straight line segments to building walls (default tolerance: 0.5 m)
3. **Area filtering** — removes detections below a minimum footprint (default: 10 m²)
4. **Optional rasterization** — polygons converted back to a pixel mask for evaluation

Output is a GeoJSON FeatureCollection with `area_px` and `area_m2` properties per building. When the input is a georeferenced GeoTIFF, coordinates are in world space (e.g. LV95 metres for SWISSIMAGE) and the GeoJSON includes a CRS member — directly loadable in QGIS, GeoPandas, or any GIS tool.

---

## Resolution-aware Inference

The model was trained at 0.3 m/px. The inference pipeline handles any input resolution:

| Input resolution | Behaviour |
|---|---|
| Finer than 0.3 m/px (e.g. SWISSIMAGE at 0.1 m/px) | Resampled to 0.3 m/px before inference, output upsampled back |
| Coarser than 0.3 m/px | Warning returned, no resampling |
| Unknown | No action, inference at native resolution |

For GeoTIFF inputs, resolution is auto-detected from the embedded CRS metadata. The probability map is upsampled **before** thresholding (bilinear interpolation) rather than after — this produces sharper building edges than upsampling a binary mask.

---

## API

Three input modes × two execution modes = six prediction endpoints:

| Endpoint | Description |
|---|---|
| `POST /predict/upload` | Multipart file upload → result directly |
| `POST /predict/from-path` | Local/mounted path → result directly (best for large tiles) |
| `POST /predict/from-url` | Azure Blob SAS URL or public URL → result directly |
| `POST /predict/async/upload` | Upload → `job_id` (202 Accepted) |
| `POST /predict/async/from-path` | Path → `job_id` (202 Accepted) |
| `POST /predict/async/from-url` | URL → `job_id` (202 Accepted) |
| `GET /jobs/{job_id}` | Poll status, get result URL when done |
| `GET /jobs/{job_id}/result` | Download result (stream or Azure SAS redirect) |
| `GET /health` | DB + storage + model reachability |

### Output parameters

| Parameter | Options | Default |
|---|---|---|
| `result_type` | `mask` / `overlay` / `vector` | `mask` |
| `processing` | `raw` / `clean` / `vectorized` | `raw` |
| `output_format` | `auto` / `png` / `tif` | `auto` |
| `resolution` | float (m/px) | auto-detected for GeoTIFF |
| `resample` | `true` / `false` | `true` |
| `simplify_tolerance` | float (metres) | `0.5` |
| `min_area` | float (m²) | `10.0` |

**Output format behaviour:** `auto` returns GeoTIFF for GeoTIFF inputs with CRS (preserving geotransform and CRS for GIS use), PNG otherwise. Explicit `tif` or `png` overrides.

### Quick start

```bash
# Start
docker compose up -d

# Async mask — recommended for large tiles
curl -X POST http://localhost:8000/predict/async/from-path \
     -H "Content-Type: application/json" \
     -d '{"path": "/data/swissimage_2494-1114.tif", "resolution": 0.1}' \
# → {"job_id": "abc-123", "status": "queued", "poll_url": "/jobs/abc-123"}

# Poll
curl http://localhost:8000/jobs/abc-123

# Download georeferenced GeoTIFF mask
curl http://localhost:8000/jobs/abc-123/result --output mask.tif

# GeoJSON in world coordinates (LV95), loadable in QGIS
curl -X POST http://localhost:8000/predict/async/from-path \
     -H "Content-Type: application/json" \
     -d '{"path": "/data/swissimage_2494-1114.tif", "resolution": 0.1, "result_type": "vector"}'
```

---

## Service Architecture

The same Docker image runs as API or worker — only the `CMD` differs. Switching from local to Azure requires only environment variable changes, no code changes.

```
Local (defaults):                    Azure production:
  STORAGE_BACKEND=local                STORAGE_BACKEND=azure
  DATABASE_URL=sqlite+aiosqlite://...  DATABASE_URL=postgresql+asyncpg://...
  WORKER_MODE=thread                   WORKER_MODE=queue

  [API container]                      [API container]  ←── thin, stateless
  [Worker: thread pool]                [Worker container] ←── Azure Queue Storage
  [SQLite + local disk]                [PostgreSQL + Azure Blob Storage]
```

**Job database** (SQLAlchemy, async): every prediction request is logged with status, input reference, params, result path, duration, and `user_id`. Enables job history, result caching, and audit trail.

**Result caching**: `SHA256(input + params)` is computed per request. Cache hit → result returned immediately with no inference. Cache miss → inference runs, result stored, cache entry written.

**Storage abstraction** (`storage/`): `AbstractStorage` interface implemented by `LocalStorage` and `AzureBlobStorage`. `get_url()` returns a local API path (local) or a time-limited SAS URL (Azure) — the API never touches result bytes again in Azure mode.

---

## Evaluation Pipeline

Modular CLI — run any combination of modules independently, each in its own versioned subdirectory:

```bash
# Run modules separately (CPU-friendly)
python evaluate.py --checkpoint checkpoints/best_model.pth \
    --mode inria --patches data/patches --cities vienna \
    --max-per-city 200 --eval pixel --out-dir outputs/evaluation

python evaluate.py --checkpoint checkpoints/best_model.pth \
    --mode inria --patches data/patches --cities vienna \
    --max-per-city 200 --eval building --out-dir outputs/evaluation

# Generate report from all completed modules
python evaluate.py --report-only \
    --checkpoint checkpoints/best_model.pth \
    --mode inria --out-dir outputs/evaluation
```

### Modules

| Module | What it measures |
|---|---|
| `pixel` | IoU, Dice, Precision, Recall — raw vs clean |
| `building` | Detection P/R/F1, mean matched IoU, size-stratified recall |
| `threshold` | PR curve, optimal threshold, F1/IoU vs threshold sweep |
| `postproc` | Building recall/F1 vs simplify tolerance and min area |
| `resolution` | Performance under native / resampled / simulated coarser inputs |

### Versioned output structure

```
outputs/evaluation/
└── c81e96af/                        ← SHA256[:8] of checkpoint
    ├── pixel_v1a2b3c4/              ← cities=vienna, max=200
    ├── building_d5e6f7a8/           ← min_area=10
    ├── threshold_dc167f6b/
    ├── qualitative/
    │   └── predictions_grid_vienna.png
    ├── report_selection.yaml        ← controls which run per module is shown
    └── report.md
```

Different model versions and different eval params never overwrite each other. `report_selection.yaml` is auto-generated and user-editable — change one line to switch the report to a different eval run.

### MLflow traceability

Evaluation results are logged to a **separate** MLflow run linked to the training run via tag:

```
Training run  [encoder=resnet34, lr=1e-4, ...]
  └── tag: training_run_id ──────────────────────┐
                                                  │
Evaluation run  [eval-20260909-082200]            │
  tag: training_run_id = abc-123  ←──────────────┘
  metrics: eval.pixel.overall.iou_raw = 0.7592
           eval.building.overall.f1_raw = 0.713
           eval.threshold.optimal_threshold = 0.5
```

---

## Repository Structure

```
inria-building-segmentation/
├── api/
│   ├── main.py          FastAPI app — 6 predict endpoints, job management, health
│   ├── inference.py     Sliding-window inference, resolution handling, TTA
│   ├── output.py        Format-aware serialisation (GeoTIFF / PNG / GeoJSON)
│   ├── vectorize.py     GIS pipeline: polygonize → simplify → area filter
│   └── schemas.py       Pydantic request/response models
├── configs/
│   └── config.yaml      All hyperparameters (single source of truth)
├── data/
│   ├── dataset.py       PyTorch Dataset, city-based splits
│   └── transforms.py    Albumentations train/val pipelines
├── db/
│   ├── models.py        SQLAlchemy Job + CachedResult models
│   ├── session.py       Async engine, session factory
│   └── crud.py          create_job, mark_done, cache lookup
├── evaluation/
│   ├── ground_truth.py  GT loading: Inria patches or swisstopo rasterization
│   ├── metrics_pixel.py Pixel IoU/Dice/Precision/Recall
│   ├── metrics_building.py Polygon matching, detection P/R/F1
│   ├── threshold_analysis.py PR curve, optimal threshold
│   ├── postproc_sensitivity.py Parameter sweep
│   ├── resolution_robustness.py Multi-resolution comparison
│   └── visualisation.py Prediction grids, report.md generation
├── models/
│   └── unet.py          SMP U-Net wrapper, freeze/unfreeze, TTA, classmethod load
├── scripts/
│   ├── download_inria.py Dataset download guide + validation
│   └── patch_dataset.py  5000×5000 → 512×512 patch extraction
├── storage/
│   ├── base.py          AbstractStorage interface
│   ├── local.py         Local filesystem backend
│   └── azure.py         Azure Blob Storage backend
├── tests/
│   └── unit/            Unit tests (metrics, inference utils, output, vectorize, crud)
├── worker/
│   ├── queue.py         LocalThreadQueue + AzureQueueStorage
│   └── job_runner.py    Async job runner
├── config.py            Pydantic-settings, all env vars with local defaults
├── train.py             Training loop, two-phase fine-tuning, MLflow logging
├── evaluate.py          Evaluation CLI orchestrator
├── docker-compose.yml   API + worker services, named volumes
└── Dockerfile           Multi-stage build, same image for API and worker
```

Directories not shown (gitignored, created locally):

```
data/                    Raw tiles, patches, swisstopo data (see Data Setup)
checkpoints/             Model weights
outputs/evaluation/      Versioned eval results: outputs/evaluation/{ckpt_hash}/{module}_{params_hash}/
outputs/inference/       API inference outputs
mlruns.db                MLflow tracking database
volumes/db/              Docker SQLite volume
volumes/results/         Docker results volume
```

---

## Setup and Usage

### A. Data setup

None of the data directories are committed to the repository — they are all gitignored. After cloning, you need to create the following structure manually.

**Expected `data/` layout:**

```
data/
├── raw/                          ← Inria raw tiles (download from Inria website)
│   └── train/
│       ├── images/               ← RGB GeoTIFFs: austin1.tif … tyrol-w36.tif
│       └── gt/                   ← Binary masks:  austin1.tif … tyrol-w36.tif
├── patches/                      ← Generated by scripts/patch_dataset.py
│   ├── austin/
│   │   ├── images/               ← 512×512 PNG crops
│   │   └── masks/                ← Corresponding binary mask PNGs
│   ├── chicago/
│   ├── kitsap/
│   ├── tyrol-w/
│   └── vienna/
└── swisstopo/                    ← Optional: custom evaluation data
    ├── SWISSIMAGE/               ← GeoTIFF tiles at 0.1 m/px (LV95, EPSG:2056)
    │   └── *.tif
    └── swissTLM3D/               ← Vector ground truth
        └── swissTLM3D_2026_LV95_LN02.gdb
```

**Accepted input formats:**

| Format | Extension | Notes |
|---|---|---|
| GeoTIFF | `.tif`, `.tiff` | Preferred — CRS and resolution auto-detected |
| JPEG | `.jpg`, `.jpeg` | No spatial metadata — supply `resolution=` manually |
| PNG | `.png` | No spatial metadata — supply `resolution=` manually |
| BMP | `.bmp` | Supported |
| WebP | `.webp` | Supported |

Bit depth: 8-bit and 16-bit images are both accepted. 16-bit is normalised to 8-bit via right-shift before inference. Multi-band images (e.g. RGBN): first 3 bands kept. Single-band (grayscale): replicated to RGB.

**Step 1 — Download Inria dataset:**

Register (free) at https://project.inria.fr/aerialimagelabeling/download/ then download via the official shell script:

```bash
mkdir -p data/raw && cd data/raw
curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash
cd ../..
```

**Step 2 — Extract patches:**

```bash
python scripts/patch_dataset.py     --src data/raw/train     --dst data/patches     --size 512     --stride 256
```

This yields ~33,400 training patches and ~10,600 validation patches across 5 cities.

**Step 3 (optional) — swisstopo data:**

- **SWISSIMAGE**: download 1 km × 1 km tiles from https://www.swisstopo.admin.ch/fr/orthophotos-swissimage-10-cm (GeoTIFF, 0.1 m/px, LV95)
- **swissTLM3D**: download the full geodatabase from https://www.swisstopo.admin.ch/fr/modele-du-territoire-swisstlm3d — place `swissTLM3D_YYYY_LV95_LN02.gdb` under `data/swisstopo/swissTLM3D/`

---

**What is NOT committed to the repository:**

| Path | Reason |
|---|---|
| `data/` | Large files (raw tiles ~15 GB, patches ~8 GB) |
| `checkpoints/` | Model weights (~100 MB) |
| `outputs/` | Evaluation results and inference outputs |
| `mlruns.db` | MLflow tracking database |
| `volumes/` | Docker named volumes (DB + results) |

---

### A. Training (Google Colab)

1. Set runtime to **GPU (T4)** before running any cell — changing runtime wipes `/content`
2. Open `notebooks/02_train_colab.ipynb`
3. Mount Google Drive (checkpoints and MLflow logs go there, ~100 MB total)
4. Download Inria dataset → `/content/inria_raw/` (VM local disk, not Drive)
5. Run patch extraction → `/content/inria_patches/`
6. Train — checkpoints auto-saved to Drive on IoU improvement
7. Resume after timeout: notebook detects `last_checkpoint.pth` on Drive automatically

```bash
# View training curves locally after downloading mlruns.db from Drive
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

### B. Evaluation (local CPU)

```bash
pip install -r requirements.txt -r requirements-dev.txt

# Sanity check (2-5 min)
python evaluate.py --checkpoint checkpoints/best_model.pth \
    --mode inria --patches data/patches --cities vienna \
    --max-per-city 10 --eval pixel --no-mlflow --out-dir outputs/evaluation

# Full evaluation (run modules separately)
python evaluate.py --checkpoint checkpoints/best_model.pth \
    --mode inria --patches data/patches --cities vienna \
    --max-per-city 200 --eval pixel --out-dir outputs/evaluation
# ... repeat for building, threshold, postproc

# Generate report
python evaluate.py --report-only \
    --checkpoint checkpoints/best_model.pth \
    --mode inria --out-dir outputs/evaluation
```

### C. Local API

```bash
# Set data directory (files mounted read-only at /data in container)
echo "DATA_DIR=/path/to/your/aerial/tiles" > .env

# Build and start
docker compose up -d --build

# Verify
curl http://localhost:8000/health
```

### D. Unit tests

```bash
pytest tests/unit/ -v
```

### E. Azure deployment

No code changes — set environment variables:

```bash
STORAGE_BACKEND=azure
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_STORAGE_CONTAINER=segmentation-results
DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname
WORKER_MODE=queue
AZURE_QUEUE_CONNECTION_STRING=...
AZURE_QUEUE_NAME=inference-jobs
```

```bash
# Start API + worker as separate containers
docker compose --profile queue up -d
```

---

## Limitations

- Evaluated on Vienna only — the Inria test set labels are not publicly available
- Small building recall is low (0.220) — a known limitation at 0.3 m/px training resolution
- Auth is an API key stub — production deployment would use Azure AD or JWT
- No CI/CD pipeline yet

---

## License

MIT
