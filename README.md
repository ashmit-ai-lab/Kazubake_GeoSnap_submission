# Kazubake — GeoSnap Land-Use Classification

Satellite land-use classification on the [EuroSAT](https://github.com/phelber/EuroSAT) dataset using GoogLeNet, covering both RGB and 13-band multispectral (Sentinel-2) inputs, plus explainability analysis with Grad-CAM, SHAP, and spectral indices.

## Tasks

| Notebook | Contents |
|---|---|
| `Kazubake_GeoSnap_full_trainin_and_inference_code.ipynb` | Task 1B — train GoogLeNet on 13-band `.tif` patches; export `ms_predictions.csv` |
| `Kazubake_GeoSnap_Explainability_notebook.ipynb` | Task 2 — Grad-CAM++, SHAP band importance, spectral signatures, t-SNE; loads RGB + MS checkpoints |
| `Kazubake_GeoSnap_Envinsights.ipynb` | Bonus — environmental interpretation via NDVI/NDWI/MNDWI on predicted classes |

## Results

| Model | Val accuracy |
|---|---|
| GoogLeNet RGB (3-band) | 98.30% |
| GoogLeNet MS (13-band) | **99.31%** |

## Setup

### Requirements

All notebooks run on **Google Colab** with a GPU runtime (tested on T4/A100). No local installation is needed beyond mounting Google Drive.

```
rasterio
shap
torch >= 2.0
torchvision
numpy, pandas, matplotlib, scikit-learn, tqdm
```

The first cell in each notebook installs `rasterio` and `shap` via `pip install -q`.

### Data

1. Download the EuroSAT dataset zip from the competition or the [EuroSAT GitHub](https://github.com/phelber/EuroSAT).
2. Upload it to Google Drive as `EuroSAT_Dataset.zip` (the zip must contain both `EuroSAT/` for RGB `.jpg` patches and `EuroSATallBands/` for 13-band `.tif` patches).
3. The notebooks unzip automatically on first run:
   ```python
   !unzip -nq "/content/drive/MyDrive/EuroSAT_Dataset.zip" -d data
   ```

Expected directory layout after unzip:

```
data/
├── EuroSAT/
│   ├── train/        # RGB .jpg, 10 classes
│   └── val/
├── EuroSATallBands/
│   ├── train/        # 13-band .tif, 10 classes
│   └── val/
├── EuroSAT_test_flat/    # unlabelled RGB test patches (4050 files)
├── EuroSATallBands_test_flat/  # unlabelled MS test patches (4050 .tif)
└── train.csv         # columns: filename, label, classname
```

### Checkpoints

Trained checkpoints are stored on Drive under `MyDrive/GeoSnap/`:

```
MyDrive/GeoSnap/
├── googlenet_rgb_best.pt
├── googlenet_ms_best.pt
└── label_map.json
```

The explainability notebook reads them directly from Drive (no copy needed).

## Reproducing Training (Task 1B — MS model)

1. Open `Kazubake_GeoSnap_full_trainin_and_inference_code.ipynb` in Colab.
2. Set **Runtime → Change runtime type → GPU**.
3. Run all cells in order. The notebook will:
   - Mount Drive and unzip the dataset.
   - Compute per-band mean/std over the training set and cache them to `data/ms_band_stats.json`.
   - Train GoogLeNet (13-channel stem, pretrained ImageNet weights) for 20 epochs.
   - Fine-tune for 10 more epochs at a lower LR.
   - Save the best checkpoint to `googlenet_ms_best.pt`.
   - Run inference on the test flat and write `ms_predictions.csv`.
4. Typical runtime: ~40 min on a T4.

Key hyperparameters (edit the Config cell to override):

| Parameter | Default |
|---|---|
| `EPOCHS` | 20 (+10 fine-tune) |
| `LR` | 3e-4 (1e-4 for fine-tune) |
| `BATCH_SIZE` | 64 |
| `IMG_SIZE` | 224 |
| `WEIGHT_DECAY` | 1e-4 |
| `SEED` | 42 |

## Reproducing Explainability (Task 2)

1. Open `Kazubake_GeoSnap_Explainability_notebook.ipynb` in Colab (GPU runtime).
2. Ensure both `.pt` checkpoints exist in `MyDrive/GeoSnap/` (see above).
3. Run all cells. The notebook produces:
   - `cm_rgb.png` / `cm_ms.png` — confusion matrices
   - `rgb_vs_ms_delta.png` — per-class accuracy gain from multispectral
   - `band_importance.png` — occlusion + SHAP band rankings
   - `spectral_signatures.png` — per-class reflectance profiles
   - `ndvi_ndwi_by_class.png` — vegetation/water index distributions
   - `tsne_rgb_vs_ms.png` — feature-space t-SNE comparison

## Reproducing Environmental Insights (Bonus)

1. Open `Kazubake_GeoSnap_Envinsights.ipynb` in Colab.
2. Run after the MS training notebook (requires the MS checkpoint and `ms_predictions.csv`).

## Output Files

| File | Description |
|---|---|
| `Kazubake_GeoSnap_ms_predictions.csv` | MS model predictions on 4050 test patches |
| `Kazubake_GeoSnap_rgb_predictions.csv` | RGB model predictions on 4050 test patches |

Both CSVs have columns `img_id, predicted_label` (class name, not integer).

## Classes

10 EuroSAT land-use categories:

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`

## Notes

- The 13-band stem sets `transform_input=False` on GoogLeNet — omitting this silently drops bands 3–12.
- Per-band normalization uses training-set statistics (not ImageNet stats); the cached `ms_band_stats.json` must be generated before training or copied from Drive.
- For the MS DataLoader, set `num_workers=0` when using `rasterio` (multiprocess fork conflicts with the library).

My contribution: GoogLeNet adaptation to 13-band multispectral input, training/eval pipeline, 11-page technical report.
