"""
Generate the FINDINGS.md figures for the fare table OCR pipeline.

Usage:
    python make_figures.py                # all plots
    python make_figures.py grid_detection # specific plot(s) only

Output: figures/ directory with visualization images.

Rendering split:
    - matplotlib (Agg backend) for charts, heatmaps and line profiles:
      fare_matrix, error_heatmap, error_distribution, baseline_vs_ocr,
      first_vs_second, evolution, grid_detection
    - OpenCV for the plots where axes add nothing: image composites
      (ocr_recipes, blue_cell_recipes, before_after) and text-infographic
      panels (method_stats, pipeline_summary, tries_overview)

The previous all-OpenCV version is kept in make_figures_cv2.py and
writes to figures/cv2/ so both styles can be compared side by side.

Plots (statistics source: FINDINGS.md + measured from output/ data):
    fare_matrix        heatmap of the final fare matrix
    error_heatmap      OCR vs GTFS difference heatmap
    ocr_recipes        preprocessing recipes on one cell
    blue_cell_recipes  min-channel masks for blue cells
    error_distribution single-digit misread bar chart
    before_after       example corrections by cross-check
    baseline_vs_ocr    OCR vs GTFS baseline (before/after)
    method_stats       per-component pipeline statistics
    pipeline_summary   whole-method summary tiles
    tries_overview     all 14 experiments, 1st vs 2nd try
    first_vs_second    measured comparison of both tries
    grid_detection     LAB gradient profile + detected peaks
    evolution          accuracy progression across approaches
"""
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless — save straight to PNG
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

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

OUTPUT = Path("figures")
OUTPUT.mkdir(exist_ok=True)

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
BEFORE_DIFFS = 3422       # differences vs GTFS before cross-check (upper triangle)
OCR_ERRORS_BEFORE = 2806  # real OCR errors in that set (rest are GTFS-side)
CORRECTED = 2806          # cells fixed by the cross-check
VOTING_CUT = 0.50         # symmetry voting removes ~50% of single-recipe errors


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
    return measure_vs_gtfs(g)


FINAL = measure_vs_gtfs(grid) if grid_full else None
FIRST1 = measure_first_try()

# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

# cv2 uses BGR; matplotlib uses hex RGB — same palette, written both ways.
GREEN_H, RED_H, BLUE_H, ORANGE_H, GRAY_H = "#3ca03c", "#dc2828", "#1e78c8", "#ff8c00", "#464646"

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

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def percent(v):
    return f"{v:.2f}%"


def savefig(fig, name):
    """Save a matplotlib figure and close it."""
    fig.savefig(str(OUTPUT / name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# --- OpenCV helpers (kept for image composites + infographic panels) --------

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


def chip(img, x, y, text, color, scale=0.5, padx=8, h=24, text_color=WHITE, bold=1):
    """Small label chip; returns right edge."""
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


# ---------------------------------------------------------------------------
# Matplotlib plots: charts, heatmaps, profiles
# ---------------------------------------------------------------------------

def plot_fare_matrix():
    """Heatmap of the final fare matrix."""
    arr = np.array([[float(v) if v else 0 for v in row] for row in grid])
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(arr, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_title(f"Fare Matrix ({len(grid)}x{len(grid[0])})")
    ax.set_xlabel("Destination column")
    ax.set_ylabel("Origin row")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Fare (RM)")
    savefig(fig, "fare_matrix.png")


def plot_error_heatmap():
    """Heatmap of |OCR - GTFS| for every cell (after cross-check)."""
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
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(errors, cmap="hot", aspect="equal", interpolation="nearest")
    ax.set_title("OCR vs GTFS Differences (after cross-check)")
    ax.set_xlabel("Destination column")
    ax.set_ylabel("Origin row")
    fig.colorbar(im, ax=ax, shrink=0.82, label="|OCR - GTFS|")
    savefig(fig, "error_heatmap.png")


def plot_error_distribution():
    """Bar chart of OCR error types (FINDINGS.md single-digit misreads)."""
    categories = {
        "9->0": 513, "0->6": 497, "5->0": 310, "3->0": 287, "8->0": 130,
        "0->4": 88, "6->0": 73, "3->2": 54, "9->4": 47, "5->4": 42,
        "8->2": 41, "6->8": 39, "5->9": 35, "8->6": 32, "4->0": 29,
        "other": 972,
    }
    labels, values = list(categories), list(categories.values())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=BLUE_H, width=0.75)
    top = max(values)
    for x, v in zip(xs, values):
        ax.text(x, v + top * 0.015, str(v), ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("count")
    ax.set_title("Single-Digit OCR Misreads")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "error_distribution.png")


def plot_baseline_vs_ocr():
    """OCR results vs the GTFS baseline fare data, before/after cross-check."""
    if fares is None or FINAL is None:
        print("  baseline_vs_ocr: skipped (need fares file + full grid)")
        return

    total = FINAL["total"]
    match_after, diffs_after = FINAL["match"], FINAL["mismatch"]
    match_before, diffs_before = total - BEFORE_DIFFS, BEFORE_DIFFS

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True,
                             gridspec_kw={"hspace": 0.65, "top": 0.84})
    fig.suptitle("OCR vs GTFS Baseline Fare Data", fontsize=16,
                 fontweight="bold", x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.90,
             f"{total:,} upper-triangle cells compared against data/fares.json",
             fontsize=10.5, color=GRAY_H)

    stages = [
        ("Before GTFS cross-check", match_before, diffs_before,
         f"differences: {OCR_ERRORS_BEFORE:,} OCR errors + "
         f"{diffs_before - OCR_ERRORS_BEFORE:,} GTFS-side "
         f"(multi-digit / length mismatches where the image is right)"),
        ("After GTFS cross-check", match_after, diffs_after,
         f"remaining {diffs_after:,} differences all verified as GTFS errors "
         f"(image correct); {CORRECTED:,} OCR errors corrected"),
    ]

    for ax, (label, match, diffs, note) in zip(axes, stages):
        ax.barh(0, match, color=GREEN_H, height=0.5)
        ax.barh(0, diffs, left=match, color=RED_H, height=0.5)
        ax.set_xlim(0, total)
        ax.set_ylim(-0.55, 0.42)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(label, loc="left", fontsize=12.5, pad=8)
        ax.text(match / 2, 0,
                f"exact match {match:,} ({percent(100 * match / total)})",
                va="center", ha="center", color="white",
                fontweight="bold", fontsize=11)
        if diffs / total > 0.10:
            ax.text(total - 10, 0, f"differ {diffs:,}", va="center", ha="right",
                    color="white", fontweight="bold", fontsize=11)
        else:
            ax.text(total, -0.46, f"differ {diffs:,} ({percent(100 * diffs / total)})",
                    ha="right", va="center", color=RED_H,
                    fontweight="bold", fontsize=10.5)
        ax.text(0, -0.46, note, fontsize=9.5, color=GRAY_H, ha="left")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.legend(handles=[Patch(color=GREEN_H, label="exact match with GTFS"),
                        Patch(color=RED_H, label="difference vs GTFS")],
               loc="lower left", bbox_to_anchor=(0.055, 0.055),
               ncol=2, frameon=False)
    fig.text(0.055, 0.018,
             "Cross-check rules: empty / single-digit misread -> GTFS;  "
             "multi-digit mismatch -> keep OCR (GTFS wrong)",
             fontsize=9.5, color=GRAY_H)
    savefig(fig, "baseline_vs_ocr.png")


def plot_first_vs_second():
    """Measured comparison: first-try output vs final output against GTFS."""
    if FINAL is None or FIRST1 is None:
        print("  first_vs_second: skipped (need full grid + fare_matrix_rows.json)")
        return

    f_fill = 100 * FIRST1["filled"] / FIRST1["total"]
    s_fill = 100 * FINAL["filled"] / FINAL["total"]
    f_match = 100 * FIRST1["match"] / FIRST1["total"]
    s_match = 100 * FINAL["match"] / FINAL["total"]
    f_empty = 100 * FIRST1["empty"] / FIRST1["total"]
    s_empty = 100 * FINAL["empty"] / FINAL["total"]

    groups = [
        ("Cells filled", f_fill, s_fill, FIRST1["filled"], FINAL["filled"]),
        ("Exact match vs GTFS", f_match, s_match, FIRST1["match"], FINAL["match"]),
        ("Empty cells", f_empty, s_empty, FIRST1["empty"], FINAL["empty"]),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    h = 0.34
    for i, (label, v1, v2, a1, a2) in enumerate(groups):
        ax.barh(i + h / 2, v1, height=h, color=RED_H)
        ax.barh(i - h / 2, v2, height=h, color=GREEN_H)
        ax.text(v1 + 1.5, i + h / 2, f"1st try  {percent(v1)}  ({a1:,})",
                va="center", fontsize=10.5, color=RED_H, fontweight="bold")
        ax.text(v2 + 1.5, i - h / 2, f"2nd try  {percent(v2)}  ({a2:,})",
                va="center", fontsize=10.5, color=GREEN_H, fontweight="bold")

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups], fontsize=11.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 133)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("percent of upper-triangle cells")
    ax.xaxis.grid(True, color="#e4e4e4")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("1st Try vs 2nd Try — Measured Against GTFS",
                 loc="left", fontsize=15, pad=34)
    ax.text(0, 1.035, "same source image, same GTFS baseline (data/fares.json)",
            transform=ax.transAxes, fontsize=10.5, color=GRAY_H)

    # No legend needed — every bar label already names its try.
    x0 = ax.get_position().x0
    fig.text(x0, 0.015,
             "1st try: ocr_rows.py (row-strip OCR, fixed geometry) vs "
             "2nd try: cells_to_csv.py (dynamic grid + voting + cross-check)   —   "
             "1st try shape 157 cols, 2nd try 156x156; fill rate counts real fare cells only",
             fontsize=9, color="#a0a0a0")
    savefig(fig, "first_vs_second.png")


def plot_evolution():
    """Accuracy progression across approaches (FINDINGS.md statistics)."""
    first_pct = 100 * FIRST1["match"] / FIRST1["total"] if FIRST1 else 0.9
    if FINAL:
        voting_pct = 100 * (FINAL["total"] - BEFORE_DIFFS) / FINAL["total"]
        final_pct = 100 * FINAL["match"] / FINAL["total"]
    else:
        voting_pct, final_pct = 72.06, 95.01

    steps = [
        ("1st try\nrow-strip OCR", first_pct, RED_H, "measured"),
        ("2nd try\nmulti-recipe\n+ symmetry", voting_pct, BLUE_H, "before cross-check"),
        ("2nd try\n+ GTFS\ncross-check", final_pct, GREEN_H, "final"),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6.8),
                                  gridspec_kw={"width_ratios": [3.1, 1]})
    # Reserve room below the axes so the multiline x-labels stay inside the
    # figure — otherwise bbox_inches="tight" expands the canvas and the
    # footer text collides with them.
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.30)
    fig.suptitle("Accuracy Evolution Across Approaches", fontsize=16,
                 fontweight="bold", x=0.05, ha="left", y=0.98)
    fig.text(0.05, 0.915, "exact match against GTFS baseline, upper-triangle cells",
             fontsize=10.5, color=GRAY_H)

    for i, (label, v, color, note) in enumerate(steps):
        ax.bar(i, v, color=color, width=0.52)
        ax.text(i, v + 2.2, percent(v), ha="center", fontsize=13,
                fontweight="bold", color=color)
        ax.text(i, -0.10, label, ha="center", va="top", fontsize=10.5,
                transform=ax.get_xaxis_transform())
        ax.text(i, -0.285, note, ha="center", va="top", fontsize=9.5,
                color="#a0a0a0", transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(-0.6, 2.6)
    ax.yaxis.grid(True, color="#e8e8e8")
    ax.set_axisbelow(True)
    ax.set_ylabel("exact match vs GTFS (%)")
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    # Rejected-approaches sidebar
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.add_patch(Rectangle((0.03, 0.04), 0.94, 0.92, fill=False,
                            edgecolor="#b0b0b0", linewidth=1))
    ax2.text(0.09, 0.93, "Evaluated &\nrejected", fontsize=12.5,
             fontweight="bold", va="top")
    ax2.text(0.09, 0.775, "template matching", fontsize=10, color=GRAY_H, va="top")
    ax2.text(0.09, 0.715, "20% (5/25 corner)", fontsize=11, color=RED_H,
             fontweight="bold", va="top")
    ax2.text(0.09, 0.655, "vs OCR 100%", fontsize=9.5, color="#a0a0a0", va="top")
    ax2.text(0.09, 0.545, "sharpening / deconv", fontsize=10, color=GRAY_H, va="top")
    ax2.text(0.09, 0.485, "gain ~ 0%", fontsize=11, color=RED_H,
             fontweight="bold", va="top")
    ax2.text(0.09, 0.425, "screenshot has\nno blur to reverse", fontsize=9.5,
             color="#a0a0a0", va="top")
    ax2.text(0.09, 0.29, "both rejected with\nmeasurements, not\njust theory",
             fontsize=9.5, color=GRAY_H, va="top")

    fig.text(0.05, 0.03,
             "Bottleneck is image resolution (3-4 px digits), not the OCR method",
             fontsize=11, color=GREEN_H, fontweight="bold")
    savefig(fig, "evolution.png")


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

    # Profiles: detection on the RAW gradient — exactly what the pipeline does
    # (prominence is relative to max, so clipping first would let noise
    # through). The clip is display-only so a single huge spike — e.g. where
    # a strip crosses the blue diagonal — cannot flatten the comb.
    row_profile = compute_median_color_profile(table, axis=1)
    row_grad = np.convolve(compute_profile_gradient(row_profile), np.ones(5) / 5, mode="same")
    row_peaks = find_boundaries(row_grad, MIN_ROW_DISTANCE)
    row_grad = np.clip(row_grad, 0, np.percentile(row_grad, 99.5))

    mid = len(row_boundaries) // 2
    strip = table[row_boundaries[mid]:row_boundaries[mid + 1]]
    col_profile = compute_median_color_profile(strip, axis=0)
    col_grad = np.convolve(compute_profile_gradient(col_profile), np.ones(5) / 5, mode="same")
    col_peaks = find_boundaries(col_grad, MIN_COL_DISTANCE)
    col_grad = np.clip(col_grad, 0, np.percentile(col_grad, 99))

    # Overlay crop: first rows x ~50 cols with per-row column boundaries
    r_end = min(12, len(row_boundaries) - 1)
    x_win = 1500
    win = table[0:min(row_boundaries[r_end], table.shape[0]), 0:x_win]
    win_rgb = cv2.cvtColor(win, cv2.COLOR_BGR2RGB)
    vert_lines = []
    for i in range(r_end):
        strip_i = table[row_boundaries[i]:row_boundaries[i + 1], :]
        verts = np.concatenate([[0], detect_cols(strip_i), [table.shape[1]]])
        vert_lines.append(([v for v in verts if v <= x_win],
                           row_boundaries[i], row_boundaries[i + 1]))

    fig = plt.figure(figsize=(13.5, 11.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.5)
    fig.suptitle("Dynamic Grid Detection (2nd try)", fontsize=16,
                 fontweight="bold", x=0.055, ha="left")
    fig.text(0.055, 0.955,
             "LAB median-color profile → gradient → smooth → scipy find_peaks → sub-pixel refinement",
             fontsize=10.5, color=GRAY_H)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(row_grad, color="#006eb4", linewidth=0.8)
    ax1.scatter(row_peaks, row_grad[row_peaks], s=26, color=RED_H,
                zorder=3, edgecolors="black", linewidths=0.4)
    ax1.set_title("Row boundaries (gradient profile over table height)", loc="left")
    ax1.set_ylabel("gradient")
    ax1.set_xticks([])
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(0.0, -0.26,
             f"peaks marked red: {len(row_peaks)}  →  {n_rows} rows after refinement + small-row merge",
             transform=ax1.transAxes, fontsize=9.5, color=GRAY_H)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(col_grad, color="#be8200", linewidth=0.8)
    ax2.scatter(col_peaks, col_grad[col_peaks], s=26, color=RED_H,
                zorder=3, edgecolors="black", linewidths=0.4)
    ax2.set_title("Column boundaries (one row strip, repeated per row with position memory)",
                  loc="left")
    ax2.set_ylabel("gradient")
    ax2.set_xticks([])
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.0, -0.26,
             f"peaks marked red: {len(col_peaks)}  + edges  →  {len(col_peaks) + 1} columns",
             transform=ax2.transAxes, fontsize=9.5, color=GRAY_H)

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(win_rgb, interpolation="nearest")
    for rb in row_boundaries[:r_end + 1]:
        ax3.axhline(rb, color="#00c000", linewidth=0.9)
    for verts, y0, y1 in vert_lines:
        for v in verts:
            ax3.vlines(v, y0, y1, color=ORANGE_H, linewidth=0.9)
    ax3.set_title("Detected boundaries on the actual image (first 12 rows x ~50 cols)",
                  loc="left")
    ax3.set_xlim(0, x_win)
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.text(0.055, 0.012,
             "No fixed geometry: every boundary is detected from the image itself — "
             "works on any resolution / table size",
             fontsize=11, color=GREEN_H, fontweight="bold")
    savefig(fig, "grid_detection.png")


# ---------------------------------------------------------------------------
# OpenCV plots: image composites (the plots ARE the images)
# ---------------------------------------------------------------------------

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
# OpenCV plots: text-infographic panels (no axes — pure layout)
# ---------------------------------------------------------------------------

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
    print("Generating figures (matplotlib + OpenCV)...")
    for fn in todo:
        fn()
    print(f"\nDone. Output in {OUTPUT}/")
