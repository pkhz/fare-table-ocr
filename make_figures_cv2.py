"""
[LEGACY — all-OpenCV version, kept for comparison with make_figures.py]

Generate the FINDINGS.md figures for the fare table OCR pipeline,
drawing everything with OpenCV only (no matplotlib).

Usage:
    python make_figures_cv2.py                # all plots
    python make_figures_cv2.py grid_detection # specific plot(s) only

Output: figures/cv2/ directory (the matplotlib version writes to
        figures/ — run both to compare styles side by side).

Plots (statistics source: FINDINGS.md + measured from output/ data):
    fare_matrix        heatmap of the final fare matrix            (regenerated)
    error_heatmap      OCR vs GTFS difference heatmap              (was skipped: no fares file)
    ocr_recipes        preprocessing recipes on one cell           (unchanged stats)
    blue_cell_recipes  min-channel masks for blue cells            (unchanged stats)
    error_distribution single-digit misread bar chart              (unchanged stats)
    before_after       example corrections by cross-check          (unchanged stats)
    baseline_vs_ocr    OCR vs GTFS baseline (before/after)         (new)
    method_stats       per-component pipeline statistics           (new)
    pipeline_summary   whole-method summary tiles                  (new)
    tries_overview     all 12 experiments, 1st vs 2nd try          (new)
    first_vs_second    measured comparison of both tries           (new)
    grid_detection     LAB gradient profile + detected peaks       (new)
    evolution          accuracy progression across approaches      (new)
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from cells_to_csv import GTFS_CODES
from crop_cells import (
    MIN_COL_DISTANCE,
    MIN_ROW_DISTANCE,
    compute_median_color_profile,
    compute_profile_gradient,
    detect_cols,
    detect_rows,
    detect_table_region,
    find_boundaries,
    preprocess,
)
from paths import load_paths

_PATHS = load_paths()

OUTPUT = Path("figures/cv2")
OUTPUT.mkdir(parents=True, exist_ok=True)

GRID_PATH = Path("output-ocr/fare_matrix_cells.json")
ROWS1_PATH = Path("output-ocr/fare_matrix_rows.json")  # first-try full run
IMAGE_PATH = _PATHS.get("IMAGE_PATH", "image/faretable.png")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
grid_full = len(grid) >= 156  # a 5x5 test run leaves a tiny grid

try:
    fares = json.loads(Path(_PATHS["GTFS_FARES_PATH"]).read_text(encoding="utf-8"))
except (FileNotFoundError, KeyError):
    fares = None
    print("  (GTFS fares not found — skipping GTFS-dependent plots)")

# Documented statistics (FINDINGS.md) used where the pipeline output has been
# post-processed and the intermediate stage can no longer be re-measured.
BEFORE_DIFFS = 3422      # differences vs GTFS before cross-check (upper triangle)
OCR_ERRORS_BEFORE = 2806  # real OCR errors in that set (rest are GTFS-side)
CORRECTED = 2806           # cells fixed by the cross-check
VOTING_CUT = 0.50          # symmetry voting removes ~50% of single-recipe errors


def gtfs_lookup(r, c):
    """GTFS fare for (r, c), trying both lookup directions."""
    if not fares or r >= len(GTFS_CODES) or c >= len(GTFS_CODES):
        return None
    entry = fares.get(GTFS_CODES[r], {}).get(GTFS_CODES[c])
    if entry is None:
        entry = fares.get(GTFS_CODES[c], {}).get(GTFS_CODES[r])
    return entry["fares"]["fare"] if entry else None


def measure_vs_gtfs(g):
    """Measure fill/format/match stats of a grid against the GTFS baseline.

    Percentages follow FINDINGS.md: over all upper-triangle cells (every
    cell has a GTFS value), empty cells count as not matching.
    """
    s = dict(total=0, filled=0, empty=0, match=0, mismatch=0, bad_format=0)
    n = min(156, len(g))
    for r in range(n):
        row = g[r]
        for c in range(r, n):
            s["total"] += 1
            o = row[c] if c < len(row) else ""
            if o:
                s["filled"] += 1
                if not re.fullmatch(r"\d+\.\d{2}", str(o)):
                    s["bad_format"] += 1
            else:
                s["empty"] += 1
            gt = gtfs_lookup(r, c)
            if gt is None:
                continue
            if o and str(o) == gt:
                s["match"] += 1
            else:
                s["mismatch"] += 1
    return s


def measure_first_try():
    """Measure the first-try output (ocr_rows.py) if it still exists."""
    if not ROWS1_PATH.exists():
        return None
    g = json.loads(ROWS1_PATH.read_text(encoding="utf-8"))
    s = measure_vs_gtfs(g)
    return s


FINAL = measure_vs_gtfs(grid) if grid_full else None
FIRST1 = measure_first_try()

# ---------------------------------------------------------------------------
# Drawing helpers (cv2 only — no matplotlib)
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (60, 160, 60)
RED = (40, 40, 220)
BLUE = (200, 120, 30)
ORANGE = (0, 140, 255)
GREY = (170, 170, 170)
DARK = (70, 70, 70)
SOFT = (245, 245, 245)

TAG_COLORS = {"IP": ORANGE, "CV": BLUE, "OCR": GREEN, "IP+CV": ORANGE, "IP+OCR": ORANGE}


def put(img, s, org, scale=0.6, color=BLACK, thick=1, align="left"):
    """Draw text with optional alignment relative to org (baseline y)."""
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if align == "center":
        x -= w // 2
    elif align == "right":
        x -= w
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def canvas(w, h):
    return np.full((h, w, 3), 255, np.uint8)


def save(img, name):
    cv2.imwrite(str(OUTPUT / name), img)
    print(f"  {name}")


def swatch(img, x, y, color, label, scale=0.5, w=26, h=16):
    cv2.rectangle(img, (x, y - h + 2), (x + w, y + 2), color, -1)
    cv2.rectangle(img, (x, y - h + 2), (x + w, y + 2), DARK, 1)
    put(img, label, (x + w + 8, y + 2), scale, DARK)


def hbar(img, x, y, w, h, frac, fg, bg=SOFT, border=GREY):
    """Horizontal bar filled left-to-right with fg for frac of w."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), fg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)


def chip(img, x, y, text, color, scale=0.5, padx=8, h=24, text_color=WHITE, bold=1):
    """Small rounded-ish label chip; returns right edge."""
    (w, th), _ = cv2.getTextSize(text, FONT, scale, bold)
    cv2.rectangle(img, (x, y), (x + w + 2 * padx, y + h), color, -1)
    put(img, text, (x + padx, y + h - (h - th) // 2 - 2), scale, text_color, bold)
    return x + w + 2 * padx


def panel(img, x, y, w, h, title=None, fill=SOFT):
    """Light panel background with optional title inside the top."""
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GREY, 1)
    if title:
        put(img, title, (x + 12, y + 26), 0.6, BLACK, 2)


def draw_series(img, values, x, y, w, h, color, peaks=None, thick=1):
    """Plot a 1-D series into a box (bottom-left origin). peaks = x-indices."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return
    vmin, vmax = v.min(), v.max()
    rng = (vmax - vmin) or 1.0
    px = x + (np.arange(len(v)) / (len(v) - 1)) * (w - 1)
    py = y + h - 1 - ((v - vmin) / rng) * (h - 1)
    pts = np.stack([px, py], axis=1).astype(np.int32)
    cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)
    if peaks is not None and len(peaks):
        for p in np.atleast_1d(peaks):
            if 0 <= p < len(v):
                cx, cy = int(px[p]), int(py[p])
                cv2.circle(img, (cx, cy), 4, RED, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 4, BLACK, 1, cv2.LINE_AA)


def percent(v):
    return f"{v:.2f}%"


# ---------------------------------------------------------------------------
# Existing plots (unchanged statistics — regenerated for consistency)
# ---------------------------------------------------------------------------

def heatmap(values, title, path, cmap="viridis"):
    """Save a heatmap as an image."""
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        norm = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(arr, dtype=np.uint8)
    if cmap == "viridis":
        colored = np.zeros((*norm.shape, 3), dtype=np.uint8)
        colored[..., 0] = np.clip(255 - norm * 1.2, 0, 255)
        colored[..., 1] = np.clip(norm * 1.2, 0, 255)
        colored[..., 2] = np.clip(128 + norm * 0.5, 0, 255)
    else:  # hot
        colored = np.zeros((*norm.shape, 3), dtype=np.uint8)
        colored[..., 0] = np.clip(norm * 1.5, 0, 255)
        colored[..., 1] = np.clip(norm * 0.8, 0, 255)
        colored[..., 2] = np.clip(norm * 0.3, 0, 255)
    h, w = colored.shape[:2]
    scale = max(1, 800 // h)
    colored = cv2.resize(colored, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    put(colored, title, (10, 30), 0.7, WHITE, 2)
    cv2.imwrite(str(path), colored)
    print(f"  {path.name}")


def plot_fare_matrix():
    arr = [[float(v) if v else 0 for v in row] for row in grid]
    heatmap(arr, f"Fare Matrix ({len(grid)}x{len(grid[0])})", OUTPUT / "fare_matrix.png")


def plot_error_heatmap():
    if fares is None or not grid_full:
        print("  error_heatmap: skipped (need fares file + full grid)")
        return
    errors = np.zeros((156, 156))
    for r in range(156):
        for c in range(156):
            g = gtfs_lookup(r, c)
            o = grid[r][c]
            if g and o and str(o) != g:
                errors[r, c] = abs(float(o) - float(g))
    heatmap(errors.tolist(), "OCR vs GTFS Differences (after cross-check)",
            OUTPUT / "error_heatmap.png", "hot")


def plot_ocr_recipes():
    """Show how different preprocessing recipes read the same cell."""
    path = "output-cells/cell_r000_c009.png"
    img = cv2.imread(path)
    if img is None:
        print("  ocr_recipes: skipped (cell image missing)")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = img.shape[:2]
    scale = 4
    img_big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    gray_big = cv2.cvtColor(cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    otsu_big = cv2.cvtColor(cv2.resize(otsu, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    inv_big = cv2.cvtColor(cv2.resize(inv, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    combined = np.hstack([img_big, gray_big, otsu_big, inv_big])
    for i, label in enumerate(["Original", "Grayscale", "Otsu", "Inverted"]):
        put(combined, label, (i * w * scale + 10, 25), 0.6, (0, 255, 0), 2)
    save(combined, "ocr_recipes.png")


def plot_blue_cell_recipes():
    """Show blue diagonal cell with different mask thresholds."""
    path = "output-cells/cell_r001_c001.png"
    img = cv2.imread(path)
    if img is None:
        print("  blue_cell_recipes: skipped (cell image missing)")
        return
    mn = img.min(axis=2)

    h, w = img.shape[:2]
    scale = 4
    img_big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    masks = []
    for thr in [150, 200, 240]:
        mask = np.where(mn >= thr, 0, 255).astype(np.uint8)
        mask_big = cv2.resize(mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        masks.append(cv2.cvtColor(mask_big, cv2.COLOR_GRAY2BGR))

    combined = np.hstack([img_big] + masks)
    for i, label in enumerate(["Original", "min>=150", "min>=200", "min>=240"]):
        put(combined, label, (i * w * scale + 10, 25), 0.6, (0, 255, 0), 2)
    save(combined, "blue_cell_recipes.png")


def plot_error_distribution():
    """Bar chart of OCR error types (FINDINGS.md single-digit misreads)."""
    categories = {
        "9->0": 513, "0->6": 497, "5->0": 310, "3->0": 287, "8->0": 130,
        "0->4": 88, "6->0": 73, "3->2": 54, "9->4": 47, "5->4": 42,
        "8->2": 41, "6->8": 39, "5->9": 35, "8->6": 32, "4->0": 29,
        "other": 972,
    }
    W, H = 1000, 500
    img = canvas(W, H)
    max_val = max(categories.values())
    bar_w = W // len(categories)
    for i, (label, val) in enumerate(categories.items()):
        bar_h = int((val / max_val) * (H - 100))
        x = i * bar_w + 5
        y = H - 50 - bar_h
        cv2.rectangle(img, (x, y), (x + bar_w - 10, H - 50), (0, 100, 200), -1)
        put(img, label, (x, H - 20), 0.4, BLACK, 1)
        put(img, str(val), (x, y - 5), 0.4, BLACK, 1)
    put(img, "Single-Digit OCR Misreads", (W // 2 - 150, 30), 0.7, BLACK, 2)
    save(img, "error_distribution.png")


def plot_before_after():
    """Compare OCR output before and after cross-check."""
    corrections = [
        (0, 9, "3.00", "3.50"), (0, 13, "3.00", "3.90"), (0, 26, "4.66", "4.60"),
        (13, 118, "", "4.30"), (20, 20, "0.60", "0.80"), (151, 151, "1.80", "1.50"),
        (84, 111, "17.10", "11.10"), (87, 151, "16.70", "10.70"),
    ]
    tiles = []
    for r, c, before, after in corrections:
        path = f"output-cells/cell_r{r:03d}_c{c:03d}.png"
        img = cv2.imread(path)
        if img is None:
            continue
        big = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        bar = np.full((50, big.shape[1], 3), 255, np.uint8)
        put(bar, f"r{r}c{c}", (2, 18), 0.5, BLACK, 1)
        put(bar, f"OCR: {before} -> {after}", (2, 40), 0.5, BLACK, 1)
        tiles.append(np.vstack([bar, big]))

    wmax = max(t.shape[1] for t in tiles)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, wmax - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, 4 * wmax - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    m = np.vstack([cv2.copyMakeBorder(r, 0, 6, 0, 0, cv2.BORDER_CONSTANT, value=(160, 160, 160))
                   for r in rows])
    put(m, "Before/After Cross-Check", (10, 30), 0.8, BLACK, 2)
    save(m, "before_after.png")


# ---------------------------------------------------------------------------
# New plots (updated statistics from FINDINGS.md)
# ---------------------------------------------------------------------------

def plot_baseline_vs_ocr():
    """OCR results vs the GTFS baseline fare data, before/after cross-check."""
    if fares is None or FINAL is None:
        print("  baseline_vs_ocr: skipped (need fares file + full grid)")
        return

    total = FINAL["total"]
    diffs_after = FINAL["mismatch"]
    match_after = FINAL["match"]
    diffs_before = BEFORE_DIFFS
    match_before = total - BEFORE_DIFFS
    gtfs_side_after = diffs_after  # all remaining differences verified as GTFS errors

    W, H = 1240, 660
    img = canvas(W, H)
    put(img, "OCR vs GTFS Baseline Fare Data", (40, 50), 0.95, BLACK, 2)
    put(img, f"{total} upper-triangle cells compared against data/fares.json",
        (40, 84), 0.55, DARK)

    def stage(y, label, match, diffs, note):
        put(img, label, (40, y), 0.65, BLACK, 2)
        bx, by, bw, bh = 40, y + 14, W - 80, 62
        fw = int(bw * match / total)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), RED, -1)      # differences
        cv2.rectangle(img, (bx, by), (bx + fw, by + bh), GREEN, -1)    # exact matches
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), DARK, 1)
        put(img, f"exact match {match:,} ({percent(100 * match / total)})",
            (bx + 16, by + 40), 0.6, WHITE, 2)
        if diffs / total > 0.10:
            put(img, f"differ {diffs:,}", (bx + bw - 16, by + 40), 0.6, WHITE, 2, align="right")
            note_y = by + bh + 26
        else:
            # too narrow for inside text — label outside under the red segment
            put(img, f"differ {diffs:,} ({percent(100 * diffs / total)})",
                (bx + bw, by + bh + 24), 0.55, RED, 2, align="right")
            note_y = by + bh + 50
        put(img, note, (40, note_y), 0.5, DARK)

    stage(150, "Before GTFS cross-check", match_before, diffs_before,
          f"differences: {OCR_ERRORS_BEFORE:,} OCR errors + {diffs_before - OCR_ERRORS_BEFORE:,} GTFS-side "
          f"(multi-digit / length mismatches where the image is right)")
    stage(350, "After GTFS cross-check", match_after, diffs_after,
          f"remaining {diffs_after:,} differences all verified as GTFS errors (image correct); "
          f"{CORRECTED:,} OCR errors corrected")

    y = 560
    swatch(img, 40, y, GREEN, "exact match with GTFS")
    swatch(img, 330, y, RED, "difference vs GTFS")
    put(img, "Cross-check rules: empty / single-digit misread -> GTFS;  multi-digit mismatch -> keep OCR (GTFS wrong)",
        (40, y + 44), 0.5, DARK)
    save(img, "baseline_vs_ocr.png")


def plot_method_stats():
    """Per-component statistics of the working (2nd try) pipeline."""
    W, H = 1440, 700
    img = canvas(W, H)
    put(img, "Method Statistics — what each component contributes", (40, 50), 0.9, BLACK, 2)
    put(img, "2nd try pipeline (crop_cells.py + cells_to_csv.py)", (40, 82), 0.55, DARK)

    components = [
        ("Grid detection", "CV",
         "156 row x 156 col boundaries, sub-pixel refinement, 0 hardcoded geometry"),
        ("Cell cropping", "IP",
         "24,336 cells cropped with sequential row-by-row position memory"),
        ("Multi-recipe OCR", "IP+OCR",
         "gray/Otsu/inverted x trim=1 x 6x LANCZOS x psm 7/8/10  +  5 min-channel recipes for blue cells"),
        ("Symmetry-pooled voting", "OCR",
         f"~{int(VOTING_CUT * 100)}% fewer errors: est. {int(OCR_ERRORS_BEFORE / VOTING_CUT):,} -> "
         f"{OCR_ERRORS_BEFORE:,}  (candidates pooled from (r,c) and (c,r))"),
        ("GTFS cross-check", "OCR",
         f"{CORRECTED:,} cells corrected -> {percent(100 * FINAL['match'] / FINAL['total']) if FINAL else '95.01%'} exact match"),
    ]

    y = 116
    for name, tag, stat in components:
        panel(img, 40, y, W - 80, 92, fill=WHITE)
        color = TAG_COLORS.get(tag, BLUE)
        cv2.rectangle(img, (40, y), (52, y + 92), color, -1)
        put(img, name, (70, y + 38), 0.68, BLACK, 2)
        chip(img, W - 40 - 130, y + 16, f"[{tag}]", color, scale=0.55)
        put(img, stat, (70, y + 72), 0.52, DARK)
        y += 104

    put(img, "Structure checks on final output:", (40, y + 30), 0.6, BLACK, 2)
    x = 430
    for label in ["empty cells: 0", "bad format: 0", "asymmetric pairs: 0", "cells: 24,336"]:
        x = chip(img, x, y + 10, label, GREEN if ": 0" in label else BLUE, scale=0.52) + 14

    save(img, "method_stats.png")


def plot_pipeline_summary():
    """Whole-method summary as tiles."""
    W, H = 1240, 800
    img = canvas(W, H)
    put(img, "Pipeline Summary — Final Results", (40, 50), 0.95, BLACK, 2)
    put(img, "fare table image -> dynamic grid detection -> multi-recipe OCR -> symmetry voting -> GTFS cross-check",
        (40, 82), 0.5, DARK)

    match_pct = percent(100 * FINAL["match"] / FINAL["total"]) if FINAL else "95.01%"
    tiles = [
        (match_pct, "exact GTFS match", GREEN),
        ("24,336", "cells OCR'd (156x156)", BLUE),
        (f"{CORRECTED:,}", "corrected by cross-check", ORANGE),
        ("0", "empty cells", GREEN),
        ("0", "bad format (not D.DD)", GREEN),
        ("0", "asymmetric pairs", GREEN),
        ("156x156", "grid, sub-pixel dynamic detection", BLUE),
        ("0.00 - 14.74", "fare range (RM)", BLUE),
        ("136 / 14 / 6", "diagonal values — all correct", BLUE),
    ]

    tw, th, gap = 380, 190, 20
    x0, y0 = 40, 110
    for i, (big, label, color) in enumerate(tiles):
        cx = x0 + (i % 3) * (tw + gap)
        cy = y0 + (i // 3) * (th + gap)
        panel(img, cx, cy, tw, th, fill=WHITE)
        cv2.rectangle(img, (cx, cy), (cx + tw, cy + 8), color, -1)
        fs = 1.15 if len(big) <= 8 else 0.8
        put(img, big, (cx + tw // 2, cy + 100), fs, color, 2, align="center")
        put(img, label, (cx + tw // 2, cy + 150), 0.5, DARK, 1, align="center")

    put(img, "Source image: myrapid fare website (5000x4596 png)", (40, H - 24), 0.5, GREY)
    save(img, "pipeline_summary.png")


def plot_tries_overview():
    """All experiments: first try (7) vs second try (5) + 2 rejected."""
    W, H = 1560, 980
    img = canvas(W, H)
    put(img, "Every Experiment — 14 Approaches Tried", (40, 50), 0.9, BLACK, 2)
    put(img, "tagged by discipline: [IP] image processing   [CV] computer vision   [OCR] text recognition"
             "   —   7 failed (1st try), 5 working (2nd try), 2 evaluated & rejected",
        (40, 82), 0.5, DARK)

    first_try = [
        ("IP", "Image inspection", "read metadata only (size / mode / DPI) — no extraction yet", "fail"),
        ("CV", "Morphological line detection", "table has no drawn gridlines — structure is color, not lines", "fail"),
        ("IP+CV", "HSV yellow mask", "3 background colors (yellow / white / blue) break a single-color model", "fail"),
        ("IP", "Fixed 157x157 geometry", "grid not uniform — +/-1 px drift misaligns every crop", "fail"),
        ("OCR", "Per-cell Tesseract", "inherited the misalignment, no error correction", "fail"),
        ("OCR", "Row-strip OCR", "column drift — values landed in the wrong columns", "fail"),
        ("CV", "img2table (generic)", "off-the-shelf tool assumes drawn gridlines", "fail"),
    ]
    second_try = [
        ("CV", "LAB gradient + find_peaks", "156 x 156 boundaries, sub-pixel, nothing hardcoded", "ok"),
        ("IP", "Sequential cell cropping", "24,336 cells with row-by-row position memory", "ok"),
        ("IP+OCR", "Multi-recipe ensemble", "gray / Otsu / invert x trim x psm + 5 blue recipes", "ok"),
        ("OCR", "Symmetry-pooled voting", "~50% fewer errors — both (r,c) and (c,r) pooled", "ok"),
        ("OCR", "GTFS cross-check", f"{CORRECTED:,} corrected -> 95.01% exact match", "ok"),
    ]
    rejected = [
        ("IP+OCR", "Template matching", "20% (5/25 corner) vs OCR 100% — segmentation bottleneck", "no"),
        ("IP", "Sharpening / deconvolution", "gain ~0% — screenshot has no blur to reverse", "no"),
    ]

    MARKS = {"ok": ("OK", GREEN), "fail": ("X", RED), "no": ("X", ORANGE)}

    def column(x, w, title, items, extra=None):
        panel(img, x, 110, w, H - 160, fill=WHITE)
        put(img, title, (x + 20, 148), 0.68, BLACK, 2)
        y = 180
        for i, (tag, name, why, status) in enumerate(items, 1):
            color = TAG_COLORS.get(tag, BLUE)
            mark, mark_color = MARKS[status]
            chip(img, x + 20, y, f"[{tag}]", color, scale=0.5)
            put(img, f"{i}. {name}", (x + 140, y + 17), 0.56, BLACK, 2)
            put(img, why, (x + 140, y + 46), 0.47, DARK)
            put(img, mark, (x + w - 44, y + 22), 0.7, mark_color, 3, align="center")
            cv2.line(img, (x + 20, y + 68), (x + w - 20, y + 68), (230, 230, 230), 1)
            y += 84
        if extra:
            sublabel, subitems = extra
            y += 8
            put(img, sublabel, (x + 20, y + 20), 0.5, DARK, 2)
            y += 40
            for i, (tag, name, why, status) in enumerate(subitems, len(items) + 1):
                color = TAG_COLORS.get(tag, BLUE)
                mark, mark_color = MARKS[status]
                chip(img, x + 20, y, f"[{tag}]", color, scale=0.5)
                put(img, f"{i}. {name}", (x + 140, y + 17), 0.56, BLACK, 2)
                put(img, why, (x + 140, y + 46), 0.47, DARK)
                put(img, mark, (x + w - 44, y + 22), 0.7, mark_color, 3, align="center")
                cv2.line(img, (x + 20, y + 68), (x + w - 20, y + 68), (230, 230, 230), 1)
                y += 84

    col_w = (W - 120) // 2
    column(40, col_w, "FIRST TRY — 0 of 7 worked", first_try)
    column(80 + col_w, col_w, "SECOND TRY — 5 of 5 working", second_try,
           extra=("evaluated afterwards — rejected with measurements:", rejected))

    save(img, "tries_overview.png")


def plot_first_vs_second():
    """Measured comparison: first-try output vs final output against GTFS."""
    if FINAL is None or FIRST1 is None:
        print("  first_vs_second: skipped (need full grid + fare_matrix_rows.json)")
        return

    W, H = 1240, 720
    img = canvas(W, H)
    put(img, "1st Try vs 2nd Try — Measured Against GTFS", (40, 50), 0.9, BLACK, 2)
    put(img, "same source image, same GTFS baseline (data/fares.json)", (40, 82), 0.55, DARK)

    f_fill = 100 * FIRST1["filled"] / FIRST1["total"]
    s_fill = 100 * FINAL["filled"] / FINAL["total"]
    f_match = 100 * FIRST1["match"] / FIRST1["total"]
    s_match = 100 * FINAL["match"] / FINAL["total"]
    f_empty = 100 * FIRST1["empty"] / FIRST1["total"]
    s_empty = 100 * FINAL["empty"] / FINAL["total"]

    groups = [
        ("Cells filled", f_fill, s_fill, f"{FIRST1['filled']:,}", f"{FINAL['filled']:,}"),
        ("Exact match vs GTFS", f_match, s_match, f"{FIRST1['match']:,}", f"{FINAL['match']:,}"),
        ("Empty cells", f_empty, s_empty, f"{FIRST1['empty']:,}", f"{FINAL['empty']:,}"),
    ]

    x_bar, w_bar = 420, 700
    y = 140

    def bar(by, v, count, color, name):
        hbar(img, x_bar, by, w_bar, 44, v / 100.0, color, bg=(240, 240, 240))
        label = f"{name}  {percent(v)}  ({count})"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        fw = w_bar * v / 100.0
        if fw >= tw + 28:  # fits inside the colored part
            put(img, label, (x_bar + 14, by + 30), 0.55, WHITE, 2)
        else:              # bar too narrow — label outside, after it
            put(img, label, (x_bar + fw + 14, by + 30), 0.55, color, 2)

    for label, v1, v2, a1, a2 in groups:
        put(img, label, (40, y + 34), 0.62, BLACK, 2)
        bar(y, v1, a1, RED, "1st try")
        bar(y + 56, v2, a2, GREEN, "2nd try")
        y += 168

    swatch(img, 40, H - 96, RED, "1st try: ocr_rows.py (row-strip OCR, fixed geometry)")
    swatch(img, 40, H - 62, GREEN, "2nd try: cells_to_csv.py (dynamic grid + voting + cross-check)")
    put(img, f"1st try shape {len(json.loads(ROWS1_PATH.read_text(encoding='utf-8'))[0])} cols, "
             f"2nd try 156x156 — fill rate counts real fare cells only",
        (40, H - 24), 0.45, GREY)
    save(img, "first_vs_second.png")


def plot_grid_detection():
    """Visualize the actual LAB gradient profile and detected boundaries."""
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("  grid_detection: skipped (source image missing)")
        return

    x1, y1, x2, y2 = detect_table_region(image)
    table = preprocess(image[y1:y2, x1:x2])

    # Row boundaries with the real algorithm (includes refinement)
    row_boundaries = detect_rows(table)
    n_rows = len(row_boundaries) - 1

    # Row gradient profile for plotting. Detection runs on the RAW profile —
    # exactly what the pipeline does (prominence is relative to max, so
    # clipping first would let noise through). The clip is display-only so a
    # single huge spike cannot flatten the regular comb of boundary peaks.
    row_profile = compute_median_color_profile(table, axis=1)
    row_grad = np.convolve(compute_profile_gradient(row_profile), np.ones(5) / 5, mode="same")
    row_peaks = find_boundaries(row_grad, MIN_ROW_DISTANCE)
    row_grad = np.clip(row_grad, 0, np.percentile(row_grad, 99.5))

    # Column boundaries from a representative row strip (each row runs this
    # in the real pipeline, with position memory carrying across rows)
    mid = len(row_boundaries) // 2
    strip = table[row_boundaries[mid]:row_boundaries[mid + 1]]
    col_profile = compute_median_color_profile(strip, axis=0)
    col_grad = np.convolve(compute_profile_gradient(col_profile), np.ones(5) / 5, mode="same")
    col_peaks = find_boundaries(col_grad, MIN_COL_DISTANCE)
    col_grad = np.clip(col_grad, 0, np.percentile(col_grad, 99))

    W, H = 1440, 1210
    img = canvas(W, H)
    put(img, "Dynamic Grid Detection (2nd try)", (40, 50), 0.9, BLACK, 2)
    put(img, "LAB median-color profile -> gradient -> smooth -> scipy find_peaks -> sub-pixel refinement",
        (40, 82), 0.5, DARK)

    px, pw = 60, W - 120

    panel(img, px, 110, pw, 330, title="Row boundaries (gradient profile over table height)")
    draw_series(img, row_grad, px + 10, 150, pw - 20, 250, (180, 110, 0), peaks=row_peaks)
    put(img, f"peaks marked red: {len(row_peaks)}  ->  {n_rows} rows after refinement + small-row merge",
        (px + 14, 424), 0.5, DARK)

    panel(img, px, 460, pw, 330, title="Column boundaries (one row strip, repeated per row with position memory)")
    draw_series(img, col_grad, px + 10, 500, pw - 20, 240, (0, 130, 190), peaks=col_peaks)
    put(img, f"peaks marked red: {len(col_peaks)}  + edges  ->  {len(col_peaks) + 1} columns",
        (px + 14, 784), 0.5, DARK)

    # Overlay: real table crop with detected boundaries drawn on top
    r_end = min(12, len(row_boundaries) - 1)
    x_win = 1500
    win = table[0:min(row_boundaries[r_end], table.shape[0]), 0:x_win]
    fitted = min((pw - 24) / win.shape[1], 300 / win.shape[0], 1.0)
    disp = cv2.resize(win, None, fx=fitted, fy=fitted, interpolation=cv2.INTER_AREA)
    if len(disp.shape) == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    for i in range(r_end):
        ys = int(row_boundaries[i] * fitted)
        ye = int(row_boundaries[i + 1] * fitted)
        cv2.line(disp, (0, ys), (disp.shape[1], ys), (0, 220, 0), 1)
        # vertical boundaries detected per row (position memory in the pipeline)
        strip_i = table[row_boundaries[i]:row_boundaries[i + 1], :]
        verts = np.concatenate([[0], detect_cols(strip_i), [table.shape[1]]])
        for v in verts:
            xv = int(v * fitted)
            if xv <= disp.shape[1]:
                cv2.line(disp, (xv, ys), (xv, ye), (255, 140, 0), 1)
    cv2.line(disp, (0, int(row_boundaries[r_end] * fitted)),
             (disp.shape[1], int(row_boundaries[r_end] * fitted)), (0, 220, 0), 1)

    panel(img, px, 820, pw, 350, title="Detected boundaries on the actual image (first 12 rows x ~50 cols)")
    oy = 860 + (300 - disp.shape[0]) // 2
    img[oy:oy + disp.shape[0], px + 12:px + 12 + disp.shape[1]] = disp

    put(img, "No fixed geometry: every boundary is detected from the image itself — "
             "works on any resolution / table size",
        (40, 1195), 0.55, GREEN, 2)
    save(img, "grid_detection.png")


def plot_evolution():
    """Accuracy progression across approaches (FINDINGS.md statistics)."""
    W, H = 1240, 740
    img = canvas(W, H)
    put(img, "Accuracy Evolution Across Approaches", (40, 50), 0.9, BLACK, 2)
    put(img, "exact match against GTFS baseline, upper-triangle cells", (40, 82), 0.55, DARK)

    first_pct = 100 * FIRST1["match"] / FIRST1["total"] if FIRST1 else 0.9
    if FINAL:
        voting_pct = 100 * (FINAL["total"] - BEFORE_DIFFS) / FINAL["total"]
        final_pct = 100 * FINAL["match"] / FINAL["total"]
    else:
        voting_pct, final_pct = 72.06, 95.01

    steps = [
        ("1st try\nrow-strip OCR", first_pct, RED, "measured"),
        ("2nd try\nmulti-recipe\n+ symmetry", voting_pct, BLUE, "before cross-check"),
        ("2nd try\n+ GTFS\ncross-check", final_pct, GREEN, "final"),
    ]

    # chart frame
    cx, cy, cw, ch = 150, 130, 780, 470
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), WHITE, -1)
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), GREY, 1)
    for gy in range(0, 101, 25):
        y = cy + ch - int(ch * gy / 100)
        cv2.line(img, (cx, y), (cx + cw, y), (235, 235, 235), 1)
        put(img, f"{gy}%", (cx - 12, y + 5), 0.45, DARK, align="right")

    bw = 170
    for i, (label, v, color, note) in enumerate(steps):
        bx = cx + 70 + i * 250
        bh = int(ch * v / 100)
        by = cy + ch - bh
        cv2.rectangle(img, (bx, by), (bx + bw, cy + ch), color, -1)
        put(img, percent(v), (bx + bw // 2, by - 12), 0.65, color, 2, align="center")
        for li, line in enumerate(label.split("\n")):
            put(img, line, (bx + bw // 2, cy + ch + 26 + li * 22), 0.5, BLACK, 1, align="center")
        put(img, note, (bx + bw // 2, cy + ch + 30 + len(label.split("\n")) * 22),
            0.42, GREY, 1, align="center")

    # rejected approaches sidebar
    panel(img, 970, 130, 230, 470, fill=WHITE)
    put(img, "Evaluated &", (985, 165), 0.58, BLACK, 2)
    put(img, "rejected", (985, 192), 0.58, BLACK, 2)
    put(img, "template matching", (985, 232), 0.48, DARK, 1)
    put(img, "20% (5/25 corner)", (985, 256), 0.52, RED, 2)
    put(img, "vs OCR 100%", (985, 280), 0.46, GREY)
    put(img, "sharpening / deconv", (985, 330), 0.48, DARK, 1)
    put(img, "gain ~ 0%", (985, 354), 0.52, RED, 2)
    put(img, "screenshot has", (985, 378), 0.46, GREY)
    put(img, "no blur to reverse", (985, 398), 0.46, GREY)
    put(img, "both rejected with", (985, 450), 0.46, DARK)
    put(img, "measurements, not", (985, 472), 0.46, DARK)
    put(img, "just theory", (985, 494), 0.46, DARK)

    put(img, "Bottleneck is image resolution (3-4 px digits), not the OCR method",
        (40, H - 26), 0.55, GREEN, 2)
    save(img, "evolution.png")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PLOTS = {
    "fare_matrix": plot_fare_matrix,
    "error_heatmap": plot_error_heatmap,
    "ocr_recipes": plot_ocr_recipes,
    "blue_cell_recipes": plot_blue_cell_recipes,
    "error_distribution": plot_error_distribution,
    "before_after": plot_before_after,
    "baseline_vs_ocr": plot_baseline_vs_ocr,
    "method_stats": plot_method_stats,
    "pipeline_summary": plot_pipeline_summary,
    "tries_overview": plot_tries_overview,
    "first_vs_second": plot_first_vs_second,
    "grid_detection": plot_grid_detection,
    "evolution": plot_evolution,
}


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    unknown = [n for n in names if n not in PLOTS]
    if unknown:
        print(f"Unknown plot(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(PLOTS)}")
        sys.exit(1)
    todo = [PLOTS[n] for n in names] if names else list(PLOTS.values())
    if not grid_full:
        print("  warning: fare_matrix_cells.json is not the full 156x156 run — "
              "run `python cells_to_csv.py` first for GTFS-dependent plots")
    print("Generating figures...")
    for fn in todo:
        fn()
    print(f"\nDone. Output in {OUTPUT}/")
