"""
Read cropped matrix cells and header cells, OCR them, and write CSV/JSON.

Inputs (produced by crop_cells.py and crop_headers.py):
    output-cells/cell_r{row:03d}_c{col:03d}.png
    output-headers/vertical/header_r{row:03d}_c{col:03d}.png   (index/name/abbrev)
    output-headers/horizontal/header_r{band:03d}_c{col:03d}.png (title/abbrev row)

Outputs (in output-ocr/):
    fare_matrix_cells.csv   -- matrix of fare values (row labels from stations)
    fare_matrix_cells.json  -- same as 2D list
    stations.csv            -- vertical header: index, station name, abbreviation
    columns.csv             -- horizontal header: column index + abbreviation

Usage:
    python cells_to_csv.py            # full matrix
    python cells_to_csv.py 10x10      # first 10x10 corner, quick test
    python cells_to_csv.py 5          # first 5x5 corner
"""

import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from ocr_rows import normalize_value
from paths import load_paths

_PATHS = load_paths()
pytesseract.pytesseract.tesseract_cmd = _PATHS["TESSERACT_CMD"]

CELLS_DIR = Path("output-cells")
HORIZONTAL_DIR = Path("output-headers/horizontal")
VERTICAL_DIR = Path("output-headers/vertical")
OUTPUT_DIR = Path("output-ocr")

CELL_RE = re.compile(r"cell_r(\d+)_c(\d+)\.png")
HCELL_RE = re.compile(r"header_r(\d+)_c(\d+)\.png")

FARE_CONFIGS = [
    "--psm 7 -c tessedit_char_whitelist=0123456789.",
    "--psm 8 -c tessedit_char_whitelist=0123456789.",
    "--psm 10 -c tessedit_char_whitelist=0123456789.",
]
INDEX_CONFIGS = [
    "--psm 7 -c tessedit_char_whitelist=0123456789",
    "--psm 8 -c tessedit_char_whitelist=0123456789",
    "--psm 13",
]
TEXT_CONFIGS = [
    "--psm 7",
    "--psm 8",
    "--psm 10",
]
# VERTICAL_DIR column layout (validated in crop_headers.py)
COL_INDEX = 0       # station index, e.g. "101"
COL_NAME = 1        # station name, e.g. "Gombak"
COL_ABBREV = 2      # station abbreviation, e.g. "GBK"

# HORIZONTAL_DIR row layout
BAND_TITLE = 0      # "INTEGRATED FARE TABLE..." (mostly blank cells)
BAND_ABBREV = 1     # column abbreviations: GBK, TAM, ...

WORKERS = 8


def newest_run(paths):
    """
    Keep only files from the most recent crop run.

    Older runs leave stale files with different grid geometry (e.g. the
    180-column run from a previous day), which would otherwise skew grid
    detection and inject out-of-grid rows/columns into the CSV.
    """
    if not paths:
        return paths
    newest = max(p.stat().st_mtime for p in paths)
    # Files written by one run are minutes apart; stale runs are hours/days.
    return [p for p in paths if newest - p.stat().st_mtime < 3600]


def grid_max(files, pattern):
    """Max row and max column indices found in a list of filenames.

    Computed independently: the widest row is not necessarily the last
    row, so a tuple-wise max would undercount columns.
    """
    max_row = max_col = -1
    for name in files:
        m = pattern.search(name)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            max_row = max(max_row, r)
            max_col = max(max_col, c)
    return max_row, max_col


UPSCALE = 6  # small cells need heavy upscaling for Tesseract


def load_variants(path, trim=0):
    """
    Read a cell as three binarization variants, upscaled for OCR.

    Plain grayscale, Otsu threshold, and inverted Otsu each succeed
    where the others fail (gray background vs. dark text vs. thin
    digits), so trying all three and voting/validating covers them.

    `trim` drops N pixels from every edge -- the crop includes half of
    the gridline on each side, and Tesseract reads that line as a
    leading '1' (e.g. '2.60' -> '12.60').
    """
    img = cv2.imread(str(path))
    if img is None:
        return {}
    if trim:
        img = img[trim:img.shape[0] - trim, trim:img.shape[1] - trim]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inv = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants = {}
    for name, arr in (("gray", gray), ("otsu", otsu), ("inv", inv)):
        pil = Image.fromarray(arr)
        variants[name] = pil.resize((pil.width * UPSCALE, pil.height * UPSCALE),
                                    Image.LANCZOS)
    return variants


def ocr_cell(path, configs, valid=None):
    """
    OCR one cell; returns stripped text ('' on failure).

    Tries each (binarization variant x psm config) combination in order
    and returns the first candidate accepted by `valid`. Candidates that
    are non-empty but rejected (OCR junk like 's(TM)' or a truncated
    '04') are kept only as a last-resort fallback -- without validation
    a wrong-but-non-empty result would short-circuit the remaining
    attempts.
    """
    try:
        variants = load_variants(path)
    except Exception:
        return ""
    fallback = ""
    for cfg in configs:
        for img in variants.values():
            try:
                text = pytesseract.image_to_string(img, config=cfg).strip()
            except Exception:
                continue
            if not text:
                continue
            if valid is None or valid(text):
                return text
            if not fallback:
                fallback = text
    return fallback


# --- fare-specific OCR (handles both cell styles) --------------------------

def is_blue_cell(path):
    """Blue diagonal cells have white text; other cells dark text."""
    img = cv2.imread(str(path))
    if img is None:
        return False
    # Sample the background (cell corners): blue = B high, R low.
    h, w = img.shape[:2]
    corners = np.concatenate([
        img[0:3, 0:3].reshape(-1, 3), img[0:3, w-3:w].reshape(-1, 3),
        img[h-3:h, 0:3].reshape(-1, 3), img[h-3:h, w-3:w].reshape(-1, 3),
    ]).mean(axis=0)  # BGR
    return corners[0] - corners[2] > 50


# Blue-diagonal recipes: (min-channel threshold, trim, scale, interp, psm, preup)
# Selected by greedy reduction from 350 candidates; majority vote over
# these five reads all sampled diagonals correctly.
BLUE_RECIPES = [
    (175, 1, 8, "l", 8, False),
    (200, 2, 1, "l", 7, True),
    (200, 2, 1, "l", 8, True),
    (200, 2, 8, "n", 7, False),
    (240, 2, 8, "n", 7, False),
]


def ocr_blue_readings(path):
    """
    Read a blue diagonal cell (white text on blue, gray border).

    Otsu groups the gray border with the white digits and misreads
    '0.80' as '9.80'. Masking on min(B,G,R) keeps only the white text
    and drops border/background entirely. No single recipe reads every
    diagonal, so return one candidate per recipe and let the caller
    majority-vote.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            return []
        readings = []
        for minv, trim, scale, interp, psm, preup in BLUE_RECIPES:
            im = img
            if preup:
                # Upscale first (smoother edges), then trim 2px-equivalent.
                im = cv2.resize(im, None, fx=4, fy=4,
                                interpolation=cv2.INTER_CUBIC)
                im = im[8:im.shape[0] - 8, 8:im.shape[1] - 8]
            elif trim:
                im = im[trim:im.shape[0] - trim, trim:im.shape[1] - trim]
            mn = im.min(axis=2)  # white text: all channels high
            mask = np.where(mn >= minv, 0, 255).astype(np.uint8)
            pil = Image.fromarray(mask)
            ip = Image.LANCZOS if interp == "l" else Image.NEAREST
            pil = pil.resize((pil.width * scale, pil.height * scale), ip)
            text = pytesseract.image_to_string(
                pil, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789."
            ).strip()
            value = ocr_fare(text)
            if value:
                readings.append(value)
        return readings
    except Exception:
        return []


def ocr_fare_readings(path):
    """
    Read one matrix cell -> list of candidate fare strings.

    Blue diagonal: dedicated white-text mask recipes.
    Normal (yellow/white) cell: gray, Otsu, and inverted Otsu at 1px
    trim (drops the gridline that OCR otherwise reads as a leading '1').
    """
    if is_blue_cell(path):
        readings = ocr_blue_readings(path)
        if readings:
            return readings
        # fall through to the normal path if every blue recipe failed

    try:
        variants = load_variants(path, trim=1)
    except Exception:
        return []
    readings = []
    for img in variants.values():
        try:
            text = pytesseract.image_to_string(
                img, config=FARE_CONFIGS[0]).strip()
        except Exception:
            continue
        value = ocr_fare(text)
        if value:
            readings.append(value)
    return readings


def majority(values):
    """Most common value; ties resolve toward the first occurrence."""
    if not values:
        return ""
    return max(dict.fromkeys(values), key=values.count)


# --- GTFS cross-check -------------------------------------------------------
#
# The image station list maps 1:1 onto GTFS stop codes by line order
# (verified against stops.csv): KJ1-37, AG18-1, SP12-22/24-29/31, MR1-11,
# BRT1-7, KG04-35, PY01-41.  The GTFS fare matrix is symmetric, so the
# transposed lookup always succeeds.

GTFS_FARES_PATH = Path(_PATHS["GTFS_FARES_PATH"])  # GTFS fares (optional, for cross-check)

GTFS_CODES = (
    [f"KJ{i}" for i in range(1, 38)] +               # rows 0-36
    [f"AG{i}" for i in range(18, 0, -1)] +             # rows 37-54
    [f"SP{i}" for i in list(range(12, 23)) + [24, 25, 26, 27, 28, 29, 31]] +
    [f"MR{i}" for i in range(1, 12)] +                # rows 73-83
    [f"BRT{i}" for i in range(1, 8)] +                # rows 84-90
    ["KG04", "KG05", "KG06", "KG07", "KG08", "KG09", "KG10", "KG12", "KG13",
     "KG14", "KG15", "KG16", "KG17", "KG18A", "KG20", "KG21", "KG22", "KG23",
     "KG24", "KG25", "KG26", "KG27", "KG28", "KG29", "KG30", "KG31", "KG33",
     "KG34", "KG35"] +                                # rows 91-119
    ["PY01", "PY03", "PY04", "PY05", "PY06", "PY07", "PY08", "PY09", "PY10",
     "PY11", "PY12", "PY13", "PY14", "PY15", "PY16", "PY17", "PY18", "PY19",
     "PY20", "PY21", "PY22", "PY23", "PY24", "PY27", "PY28", "PY29", "PY31",
     "PY32", "PY33", "PY34", "PY36", "PY37", "PY38", "PY39", "PY40", "PY41"]
)

# Diagonal cells where the image shows a non-standard value that GTFS
# does not have (GTFS says 0.80 for every diagonal).  Verified visually.
WHITE_DIAGONAL = {57, 65, 67, 85, 127, 151}

# Manual corrections for white diagonal cells (verified visually).
WHITE_DIAGONAL_FIX = {151: "1.50"}


def cross_check_with_gtfs(grid):
    """
    Correct OCR errors using the GTFS fare matrix as ground truth.

    Error model (established by visual verification of samples):
      - empty cell                     -> GTFS is correct
      - single-digit misread           -> GTFS is correct (visible text matches)
      - multi-digit / length mismatch  -> OCR is correct (GTFS disagrees with
                                           the image for cross-line MR/BRT fares)
    """
    if not GTFS_FARES_PATH.exists():
        print("GTFS fares not found, skipping cross-check")
        return grid
    if len(grid) > len(GTFS_CODES) or len(grid[0]) > len(GTFS_CODES):
        print("Grid larger than GTFS mapping, skipping cross-check")
        return grid

    with open(GTFS_FARES_PATH, encoding="utf-8") as f:
        fares = json.load(f)

    corrected = 0
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(r, cols):
            entry = fares.get(GTFS_CODES[r], {}).get(GTFS_CODES[c])
            if entry is None:
                entry = fares.get(GTFS_CODES[c], {}).get(GTFS_CODES[r])
            if entry is None:
                continue
            gtfs_v = entry["fares"]["fare"]

            # White diagonal cells: GTFS has 0.80 but the image shows a
            # different value -- keep OCR (with manual fixes applied).
            if r == c and r in WHITE_DIAGONAL:
                fix = WHITE_DIAGONAL_FIX.get(r)
                if fix and grid[r][c] != fix:
                    grid[r][c] = fix
                    corrected += 1
                continue

            # Blue diagonal cells: GTFS (0.80/0.90) is always correct; the
            # OCR sometimes transposes the digits ('0.80' -> '8.00').
            if r == c:
                if grid[r][c] != gtfs_v:
                    corrected += 1
                grid[r][c] = gtfs_v
                continue

            ocr_v = grid[r][c]
            if not ocr_v:
                grid[r][c] = grid[c][r] = gtfs_v
                corrected += 1
            elif ocr_v != gtfs_v and len(ocr_v) == len(gtfs_v):
                diff_pos = [i for i in range(len(ocr_v)) if ocr_v[i] != gtfs_v[i]]
                if len(diff_pos) == 1:
                    grid[r][c] = grid[c][r] = gtfs_v
                    corrected += 1

    print(f"GTFS cross-check: corrected {corrected} cells")
    return grid


def ocr_fare(text):
    """Normalize OCR output to a fare 'D.DD' string, or '' if unreadable."""
    value = normalize_value(text)
    if value is None:
        return ""
    # Enforce two decimals: OCR often returns '2.0' for '2.00'.
    if re.fullmatch(r"\d+\.\d", value):
        value += "0"
    # Reject anything that is not D.DD (e.g. '1.410', '2390863.30').
    if not re.fullmatch(r"\d{1,3}\.\d{2}", value):
        return ""
    return value


def clean_text(text):
    """Strip separator junk OCR picks up from cell borders (both ends)."""
    return re.sub(r"^[^0-9A-Za-z(]+|[^0-9A-Za-z)!?.]+$", "", text).strip()


# --- validation predicates (reject OCR junk, force trying more variants) ---

def is_fare(text):
    return normalize_value(text) is not None


def is_index(text):
    """Station indexes are 3-digit (101..256); rejects '04', '2', ..."""
    return re.fullmatch(r"\d{3}", text) is not None


def is_abbrev(text):
    """Column/station abbreviations are 2-6 ASCII letters (GBK, STW)."""
    return re.fullmatch(r"[A-Za-z]{2,6}", text) is not None


def is_name(text):
    """Station names contain at least 2 consecutive letters."""
    return re.search(r"[A-Za-z]{2,}", text) is not None


def collect_paths(directory, pattern):
    """Sorted list of (row, col, path) from the most recent crop run."""
    items = []
    for path in newest_run(list(directory.glob("*.png"))):
        m = pattern.search(path.name)
        if m:
            items.append((int(m.group(1)), int(m.group(2)), path))
    items.sort()
    return items


def read_stations(max_row, limit_rows):
    """OCR the vertical header -> list of (index, name, abbrev)."""
    paths = [p for r, c, p in collect_paths(VERTICAL_DIR, HCELL_RE)
             if c == COL_NAME and r <= max_row and r < limit_rows]
    print(f"OCR vertical header: {len(paths)} station name cells")

    def work(path):
        row = int(HCELL_RE.search(path.name).group(1))
        base = path.parent
        name = clean_text(ocr_cell(path, TEXT_CONFIGS, is_name))
        abbrev = clean_text(ocr_cell(
            base / f"header_r{row:03d}_c{COL_ABBREV:03d}.png",
            TEXT_CONFIGS, is_abbrev))
        index = clean_text(ocr_cell(
            base / f"header_r{row:03d}_c{COL_INDEX:03d}.png",
            INDEX_CONFIGS, is_index))
        return row, (index, name, abbrev)

    stations = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for row, values in ex.map(work, paths):
            stations[row] = values

    return [stations.get(r, ("", "", "")) for r in range(len(stations))]


def read_column_abbrevs(max_col, limit_cols):
    """OCR the horizontal header abbreviation band -> list of abbrevs."""
    paths = [p for r, c, p in collect_paths(HORIZONTAL_DIR, HCELL_RE)
             if r == BAND_ABBREV and c <= max_col and c < limit_cols]
    print(f"OCR horizontal header: {len(paths)} column cells")

    abbrevs = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = ex.map(
            lambda p: (int(HCELL_RE.search(p.name).group(2)),
                       clean_text(ocr_cell(p, TEXT_CONFIGS, is_abbrev))),
            paths,
        )
        for col, text in results:
            abbrevs[col] = text

    return [abbrevs.get(c, "") for c in range(len(abbrevs))]


def read_matrix(max_row, max_col, limit_rows, limit_cols):
    """
    OCR the matrix cells -> 2D list of fare strings.

    Each cell yields several candidate readings (multiple preprocessing
    recipes). The fare table is symmetric (A->B == B->A), so candidates
    from both cells of each (r,c)/(c,r) pair are pooled and majority-
    voted -- a stray '1' or a dropped digit on one side is outvoted by
    the partner cell's clean reads.
    """
    rows = min(max_row + 1, limit_rows)
    cols = min(max_col + 1, limit_cols)
    paths = [p for r, c, p in collect_paths(CELLS_DIR, CELL_RE)
             if r < rows and c < cols]
    print(f"OCR matrix: {len(paths)} cells ({rows} x {cols})")

    readings = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for (r, c), vals in zip(
            ((int(CELL_RE.search(p.name).group(1)),
              int(CELL_RE.search(p.name).group(2))) for p in paths),
            ex.map(ocr_fare_readings, paths),
        ):
            readings[(r, c)] = vals
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(paths)} cells")

    grid = [[""] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(r, cols):
            if r == c:
                pool = readings.get((r, c), [])
            else:
                pool = readings.get((r, c), []) + readings.get((c, r), [])
            value = majority(pool)
            grid[r][c] = value
            grid[c][r] = value

    return grid


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    limit_rows, limit_cols = sys.maxsize, sys.maxsize
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "x" in arg:
            r, c = arg.split("x")
            limit_rows, limit_cols = int(r), int(c)
        else:
            limit_rows = limit_cols = int(arg)

    cell_files = [p.name for p in newest_run(list(CELLS_DIR.glob("*.png")))
                  if CELL_RE.search(p.name)]
    if not cell_files:
        raise SystemExit(f"No cells found in {CELLS_DIR} -- run crop_cells.py first")
    max_row, max_col = grid_max(cell_files, CELL_RE)
    print(f"Matrix grid: {max_row + 1} rows x {max_col + 1} cols")

    # 1. Stations (vertical header)
    stations = read_stations(max_row, limit_rows)
    with open(OUTPUT_DIR / "stations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "station_name", "abbreviation"])
        writer.writerows(stations)
    print(f"Saved {OUTPUT_DIR / 'stations.csv'} ({len(stations)} stations)")

    # 2. Column abbreviations (horizontal header)
    col_abbrevs = read_column_abbrevs(max_col, limit_cols)
    with open(OUTPUT_DIR / "columns.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["col", "abbreviation"])
        writer.writerows(enumerate(col_abbrevs))
    print(f"Saved {OUTPUT_DIR / 'columns.csv'} ({len(col_abbrevs)} columns)")

    # 3. Fare matrix
    grid = read_matrix(max_row, max_col, limit_rows, limit_cols)
    grid = cross_check_with_gtfs(grid)

    # Prepend station name so each fare row is self-describing.
    with open(OUTPUT_DIR / "fare_matrix_cells.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["index", "station_name"] + [
            f"c{c:03d}" for c in range(len(grid[0]) if grid else 0)
        ]
        writer.writerow(header)
        for r, row in enumerate(grid):
            index, name, _ = stations[r] if r < len(stations) else ("", "", "")
            writer.writerow([index, name] + row)

    with open(OUTPUT_DIR / "fare_matrix_cells.json", "w",
              encoding="utf-8") as f:
        json.dump(grid, f, indent=2)

    print(f"Saved {OUTPUT_DIR / 'fare_matrix_cells.csv'}")
    print(f"Saved {OUTPUT_DIR / 'fare_matrix_cells.json'}")


if __name__ == "__main__":
    main()
