"""
Header cropper for the fare table image.

Crops the horizontal header (station names + abbreviations above the
matrix), the vertical header (station index + station names left of the
matrix), and the top-left corner block where the two headers intersect.

Reuses the detection helpers from crop_cells.py so the header geometry
stays consistent with the matrix cell crops:

- Horizontal header columns reuse the MATRIX column boundaries, so
  header_r000_c005.png sits exactly above cell_rNNN_c005.png.
- Vertical header rows reuse the MATRIX row boundaries, so
  header_r012_c000.png sits exactly left of cell_r012_cNNN.png.

Output layout (three subdirs to keep r/c numbering collision-free,
since horizontal rows and matrix rows share indices 0..1):

    output-headers/horizontal/header_r{hr:03d}_c{matrix_col:03d}.png
    output-headers/vertical/header_r{matrix_row:03d}_c{vc:03d}.png
    output-headers/corner/header_r{hr:03d}_c{vc:03d}.png

Debug: output-headers-debug/debug_headers.jpg (downscaled JPEG).
"""

import cv2
import numpy as np
from pathlib import Path

from crop_cells import (
    IMAGE_PATH,
    PROMINENCE_FACTOR,
    detect_table_region,
    preprocess,
    compute_median_color_profile,
    compute_profile_gradient,
    find_boundaries,
    refine_boundary,
    merge_small_rows,
    detect_rows,
    detect_cols,
)

OUTPUT_ROOT = Path("output-headers")
HORIZONTAL_DIR = OUTPUT_ROOT / "horizontal"
VERTICAL_DIR = OUTPUT_ROOT / "vertical"
CORNER_DIR = OUTPUT_ROOT / "corner"
DEBUG_DIR = Path("output-headers-debug")

# Header-specific tuning
MIN_BAND_DISTANCE = 6   # header text bands are closer together than matrix rows
MIN_BAND_HEIGHT = 8     # allow short band rows in the header strip
CONTENT_THRESHOLD = 200  # gray < this counts as content (vs. white background)


def find_content_bbox(image, threshold=CONTENT_THRESHOLD):
    """
    Bounding box of all non-white content in the image.

    Extends beyond the blue-diagonal matrix bbox to include the headers
    (station names/abbreviations above, index + names to the left).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < threshold)
    if len(xs) == 0:
        raise RuntimeError("No content found in image")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def detect_header_bands(strip):
    """
    Detect horizontal header text bands (rows) inside the header strip.

    Same median-color projection approach as detect_rows() but with a
    smaller peak distance -- header bands are much shorter than matrix
    rows and would otherwise be merged.
    """
    profile = compute_median_color_profile(strip, axis=1)
    gradient = compute_profile_gradient(profile)

    kernel = np.ones(5) / 5
    gradient_smooth = np.convolve(gradient, kernel, mode='same')

    peaks = find_boundaries(gradient_smooth, MIN_BAND_DISTANCE, PROMINENCE_FACTOR)
    refined = [refine_boundary(strip, p, axis=1) for p in peaks]

    boundaries = np.concatenate([[0], refined, [strip.shape[0]]])
    boundaries = merge_small_rows(boundaries, min_height=MIN_BAND_HEIGHT)
    return boundaries


def detect_strip_cols(strip):
    """Detect column boundaries inside a vertical strip (with edges)."""
    peaks = detect_cols(strip)
    return np.concatenate([[0], peaks, [strip.shape[1]]])


def clear_stale():
    """Remove header files from previous runs."""
    removed = 0
    for directory in (HORIZONTAL_DIR, VERTICAL_DIR, CORNER_DIR):
        for stale in directory.glob("header_*.png"):
            stale.unlink()
            removed += 1
    if removed:
        print(f"Removed {removed} stale header files from previous run")


def save_debug(image, hrows, mrows, mcols, vcols, mx1, my1, output_path):
    """
    Downscaled JPEG debug image of the full table with all boundaries:
    - red   = horizontal header band rows (full width)
    - orange = matrix row boundaries
    - green = matrix column boundaries (matrix area only)
    - blue  = vertical header column boundaries (left strip only)
    """
    x1, y1 = vcols[0], hrows[0]
    x2, y2 = mcols[-1], mrows[-1]
    crop = image[y1:y2, x1:x2].copy()

    def draw_h(y, color, thickness=1):
        cv2.line(crop, (0, y - y1), (crop.shape[1] - 1, y - y1), color, thickness)

    def draw_v(x, y_from, y_to, color, thickness=1):
        cv2.line(crop, (x - x1, y_from - y1), (x - x1, y_to - y1), color, thickness)

    for y in hrows:
        draw_h(y, (0, 0, 255), 2)          # red
    for y in mrows:
        draw_h(y, (0, 140, 255), 1)        # orange
    for x in mcols:
        draw_v(x, my1, mrows[-1], (0, 255, 0), 1)   # green
    for x in vcols:
        draw_v(x, hrows[0], mrows[-1], (255, 0, 0), 1)  # blue

    small = cv2.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), small, [cv2.IMWRITE_JPEG_QUALITY, 85])


def crop_headers():
    """Crop horizontal header, vertical header, and corner block."""
    for directory in (HORIZONTAL_DIR, VERTICAL_DIR, CORNER_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)
    clear_stale()

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise RuntimeError(f"Cannot load {IMAGE_PATH}")

    # 1. Matrix bbox (blue diagonal) and full content bbox (incl. headers)
    mx1, my1, mx2, my2 = detect_table_region(image)
    fx1, fy1, fx2, fy2 = find_content_bbox(image)
    hdr_left = min(fx1, mx1)
    hdr_top = min(fy1, my1)
    print(f"Matrix bbox : ({mx1}, {my1}) -> ({mx2}, {my2})")
    print(f"Content bbox: ({fx1}, {fy1}) -> ({fx2}, {fy2})")
    print(f"Header origin: ({hdr_left}, {hdr_top})")

    # 2. Matrix geometry (same detection as crop_cells.py)
    matrix = image[my1:my2, mx1:mx2]
    matrix_blur = preprocess(matrix)
    mrows_rel = detect_rows(matrix_blur)
    mrows = my1 + mrows_rel  # absolute y

    # Column boundaries from a representative (middle) matrix row
    mid = len(mrows_rel) // 2
    mid_strip = matrix_blur[mrows_rel[mid]:mrows_rel[mid + 1]]
    mcols_rel = detect_strip_cols(mid_strip)
    mcols = mx1 + mcols_rel  # absolute x
    print(f"Matrix rows: {len(mrows_rel) - 1}, cols: {len(mcols_rel) - 1}")

    # 3. Horizontal header strip (above matrix, full width)
    hstrip = image[hdr_top:my1, hdr_left:mx2]
    hstrip_blur = preprocess(hstrip)
    hrows_rel = detect_header_bands(hstrip_blur)
    hrows = hdr_top + hrows_rel  # absolute y
    print(f"Horizontal header bands: {len(hrows_rel) - 1}")

    # 4. Vertical header strip (left of matrix, from header top to matrix bottom)
    vstrip = image[hdr_top:my2, hdr_left:mx1]
    vstrip_blur = preprocess(vstrip)
    vcols_rel = detect_strip_cols(vstrip_blur)
    vcols = hdr_left + vcols_rel  # absolute x
    print(f"Vertical header columns: {len(vcols_rel) - 1}")

    total = 0

    # 5a. Horizontal header: header bands x matrix columns
    #     (aligned with cell_r*_c{col})
    for hr in range(len(hrows) - 1):
        for ci in range(len(mcols) - 1):
            cell = image[hrows[hr]:hrows[hr + 1], mcols[ci]:mcols[ci + 1]]
            name = f"header_r{hr:03d}_c{ci:03d}.png"
            cv2.imwrite(str(HORIZONTAL_DIR / name), cell)
            total += 1
        print(f"Horizontal band {hr}: {len(mcols) - 1} cells")

    # 5b. Vertical header: matrix rows x strip columns
    #     (aligned with cell_r{row}_c*)
    for ri in range(len(mrows) - 1):
        for vi in range(len(vcols) - 1):
            cell = image[mrows[ri]:mrows[ri + 1], vcols[vi]:vcols[vi + 1]]
            name = f"header_r{ri:03d}_c{vi:03d}.png"
            cv2.imwrite(str(VERTICAL_DIR / name), cell)
            total += 1

    # 5c. Corner block: header bands x strip columns (top-left intersection)
    for hr in range(len(hrows) - 1):
        for vi in range(len(vcols) - 1):
            cell = image[hrows[hr]:hrows[hr + 1], vcols[vi]:vcols[vi + 1]]
            name = f"header_r{hr:03d}_c{vi:03d}.png"
            cv2.imwrite(str(CORNER_DIR / name), cell)
            total += 1

    # 6. Debug visualization
    debug_path = DEBUG_DIR / "debug_headers.jpg"
    save_debug(image, hrows, mrows, mcols, vcols, mx1, my1, debug_path)

    print(f"\nTotal header cells: {total}")
    print(f"Saved to: {OUTPUT_ROOT}/{{horizontal,vertical,corner}}")
    print(f"Debug visualization: {debug_path}")


if __name__ == "__main__":
    crop_headers()
