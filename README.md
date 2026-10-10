# Transit Fare Table OCR

An experimental computer vision and image processing project exploring how a large transit fare-table image specifically in this use case is RapidKL integrated fare table image (`image/faretable.png`, 5000×4596 px) can be converted into structured, machine-readable data.

The project investigates whether OCR accuracy can be improved to near-perfect levels through different processing and recognition methods, and whether any remaining errors are attributable to limitations in the source image quality.

## Project Overview

### Experimentation

- Image preprocessing and grayscale conversion
- Thresholding and segmentation (Otsu, inverted, min-channel masking)
- Table boundary detection via LAB color-space projection and gradient peak finding
- Horizontal and vertical line detection
- Grid and cell detection
- Automatic cell coordinate estimation
- Individual cell cropping
- Multi-recipe OCR preprocessing and text extraction
- Symmetry-pooled majority voting for error correction
- Reconstructing extracted values into a structured fare matrix
- Ground-truth cross-validation against external fare data

### Result

The table structure and individual cells were detected and extracted successfully. OCR accuracy was limited by the source image resolution (3–4 px digits), but the multi-recipe approach combined with symmetry voting and ground-truth cross-validation achieved **95%+ cell accuracy** on 24,336 cells.

This experiment provided practical experience with image processing, computer vision, document image analysis, and OCR, including the challenges involved when working with imperfect real-world images.

### Technologies

Python, OpenCV, Pillow, Tesseract OCR, NumPy, SciPy, Matplotlib

## Files

| File | Description |
|---|---|
| `crop_cells.py` | Detects the 156×156 matrix grid dynamically (LAB median-color projection + gradient peak finding), crops each cell to `output-cells/cell_r{row}_c{col}.png` |
| `crop_headers.py` | Crops horizontal (title/abbrev bands), vertical (index/name/abbrev), and corner header cells into `output-headers/{horizontal,vertical,corner}/` |
| `cells_to_csv.py` | OCRs all cells (multi-recipe Tesseract + symmetry-pooled majority voting), cross-checks against GTFS fares, writes `output-ocr/{stations.csv, columns.csv, fare_matrix_cells.csv, fare_matrix_cells.json}` |
| `extract_gtfs_errors.py` | Extracts every remaining OCR-misread cell into `output-gtfs-check/` with a `manifest.csv` listing origin/destination station names and both fare values |
| `make_figures.py` | Generates the 13 figures in `figures/` (fare matrix heatmap, OCR vs GTFS baseline, per-method stats, pipeline summary, all 14 experiments, 1st vs 2nd try, grid detection, accuracy evolution, recipe comparisons). Charts/heatmaps/profiles use matplotlib; image composites and text panels use OpenCV |
| `make_figures_cv2.py` | Legacy all-OpenCV version of the generator — same 13 plots drawn with OpenCV only, writes to `figures/cv2/` so both styles can be compared side by side |
| `ocr_rows.py` | Provides `normalize_value()` — normalizes OCR text to `D.DD` fare format |
| `image/faretable.png` | Source fare table image (downloaded from the myrapid fare website) |

## Usage

```bash
# import modules/lib/dependents/packages
pip install requirements.txt

# Full pipeline (crop → OCR → cross-check → CSV/JSON)
python crop_cells.py
python crop_headers.py
python cells_to_csv.py

# Extract the verified OCR-misread cells → output-gtfs-check/
python extract_gtfs_errors.py

# Quick test (first 10×10 corner)
python cells_to_csv.py 10x10

# Quick test (first 5×5 corner)
python cells_to_csv.py 5

# FINDINGS.md figures → figures/*.png
python make_figures.py                # all plots (matplotlib + OpenCV)
python make_figures.py grid_detection # single plot

# Legacy all-OpenCV rendering → figures/cv2/*.png (for comparison)
python make_figures_cv2.py
```

## Requirements

- Python 3.x with `.venv` (see `.venv/Scripts/python.exe`)
- Tesseract OCR at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Python packages: `pytesseract`, `Pillow`, `opencv-python`, `numpy`, `scipy`, `python-dotenv`, `matplotlib` (charts in `make_figures.py`)

## Environment & Paths

Path configuration is stored in [`paths.env`](paths.env) — no hardcoded paths in source code.

| Variable | Description |
|---|---|
| `TESSERACT_CMD` | Path to Tesseract executable |
| `GTFS_FARES_PATH` | GTFS fares file for cross-check (optional) |
| `IMAGE_PATH` | Source fare table image |

### paths.env

```ini
# Copy and adjust paths as needed
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
GTFS_FARES_PATH=data/fares.json
IMAGE_PATH=image/faretable.png
```

The pipeline reads `paths.env` at startup via `paths.py`. If the file is missing, defaults are used. If `GTFS_FARES_PATH` doesn't exist, the cross-check is skipped and raw OCR output (with symmetry voting) is produced.

### Other paths

| Path | Description |
|---|---|
| `output-cells/` | Cropped matrix cell PNGs (generated by `crop_cells.py`) |
| `output-headers/` | Cropped header cell PNGs (generated by `crop_headers.py`) |
| `output-ocr/` | Final CSV/JSON output (generated by `cells_to_csv.py`) |
| `output-gtfs-check/` | Extracted OCR-misread cells + `manifest.csv` (generated by `extract_gtfs_errors.py`) |

## Output

All outputs are written to `output-ocr/`:

| File | Contents |
|---|---|
| `stations.csv` | index, station name, abbreviation (from vertical header) |
| `columns.csv` | col index + abbreviation (from horizontal header) |
| `fare_matrix_cells.csv` | full matrix with `index`/`station_name` row labels |
| `fare_matrix_cells.json` | same matrix as 2D list |

## Method & Findings

See [FINDINGS.md](FINDINGS.md) for the full algorithm description, error analysis, correction methods, and final results.

## Resources

Faretable image (Cash/Token) - https://myrapid.com.my/bus-train/rapid-kl/integrated-fare-table

Data - RapidKL GTFS Data - https://developer.data.gov.my/realtime-api/gtfs-static

## AI Assistance

- **First try** — Claude Sonnet 5 (via GitHub Copilot): the fixed-geometry experiments
  (`crop.py` → `ocr_rows.py`)
- **Second try** — LongCat 2.5 Preview Free/MiMo-v2.6-Flash Free (via OpenCode): 
  (`crop_cells.py`, `cells_to_csv.py`) + the figures in `figures/`
