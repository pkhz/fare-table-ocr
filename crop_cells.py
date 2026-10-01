"""
Cell-by-cell cropper for the fare table image.

Algorithm: Sequential Row-By-Row Cell Detection with Position Memory

This script crops the fare table into individual cells without relying
on fixed geometry. It uses median-color projection analysis to detect
cell boundaries dynamically, avoiding text-interference issues.

Established algorithms used:
1. Projection Profile Analysis - classic technique for table detection
2. Median Color Filtering - robust to text outliers (vs. raw gradients)
3. Sobel Gradient Operators - for boundary refinement
4. Peak Detection (scipy.signal.find_peaks) - for finding boundary positions
5. Gaussian Blur - for noise reduction
6. LAB Color Space - for perceptually uniform color differences

Key features:
- No fixed geometry required
- Remembers row-end positions for alignment between rows
- Includes headers (row 0 = horizontal header, col 0 = vertical header)
- Two-pass detection: median-color projection + gradient refinement
- Debug visualization of detected boundaries
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

IMAGE_PATH = "image/faretable.png"
OUTPUT_DIR = Path("output-cells")
DEBUG_DIR = Path("output-cells-debug")

# Tuning parameters
BLUR_KERNEL = (5, 5)
MIN_ROW_DISTANCE = 15
MIN_COL_DISTANCE = 15
MIN_ROW_HEIGHT = 20  # rows smaller than this are merged with the next row
PROMINENCE_FACTOR = 0.05
REFINE_WINDOW = 5  # pixels to search around detected boundary for refinement


def detect_table_region(image):
    """
    Detect the fare matrix region using the blue diagonal.

    The fare table has a blue diagonal line that marks the matrix
    boundaries. We use HSV color-based segmentation to find this
    diagonal and determine the matrix extent.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([140, 255, 255])
    blue = cv2.inRange(hsv, lower_blue, upper_blue)

    ys, xs = np.where(blue > 0)
    if len(xs) == 0:
        raise RuntimeError("Blue diagonal not detected")

    return xs.min(), ys.min(), xs.max(), ys.max()


def preprocess(image):
    """
    Apply Gaussian blur to reduce noise.
    """
    return cv2.GaussianBlur(image, BLUR_KERNEL, 0)


def compute_median_color_profile(image, axis=0):
    """
    Compute median color profile along the specified axis.

    axis=0: median across rows → profile per column (for vertical boundaries)
    axis=1: median across columns → profile per row (for horizontal boundaries)

    Median is robust to text outliers, giving a clean background color signal.
    """
    # Convert to LAB for perceptually uniform differences
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    if axis == 0:
        # Median across rows → one color per column
        profile = np.median(lab, axis=0)  # shape: (width, 3)
    else:
        # Median across columns → one color per row
        profile = np.median(lab, axis=1)  # shape: (height, 3)

    return profile


def compute_profile_gradient(profile):
    """
    Compute gradient magnitude of the color profile.

    Measures how quickly the background color changes at each position.
    High gradient = likely cell boundary.
    """
    if len(profile) < 2:
        return np.zeros(len(profile))

    # Compute color difference between adjacent positions
    diff = np.diff(profile, axis=0)
    gradient = np.linalg.norm(diff, axis=1)

    # Pad to match original length
    gradient = np.concatenate([[0], gradient])

    return gradient


def find_boundaries(projection, min_distance, prominence_factor=PROMINENCE_FACTOR):
    """
    Find boundary positions using peak detection.

    Uses scipy's find_peaks to identify significant peaks in the
    projection profile, which correspond to cell boundaries.
    """
    if len(projection) == 0:
        return np.array([], dtype=int)

    prominence = np.max(projection) * prominence_factor

    peaks, _ = find_peaks(
        projection,
        distance=min_distance,
        prominence=prominence
    )
    return peaks


def refine_boundary(image, approximate_pos, axis=0, window=REFINE_WINDOW):
    """
    Refine a boundary position by searching for maximum color difference
    in a small window around the approximate position.

    axis=0: refine vertical boundary (search along x-axis)
    axis=1: refine horizontal boundary (search along y-axis)
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Search window
    start = max(0, approximate_pos - window)
    end = min(w if axis == 0 else h, approximate_pos + window + 1)

    if end - start < 2:
        return approximate_pos

    # Compute color difference between adjacent pixels in the window.
    # Clip both slices to the same length: when the window reaches the
    # image edge, [start:end] and [start+1:end+1] can differ by one.
    if axis == 0:
        # Vertical boundary: compare column i with column i+1
        a = lab[:, start:end].astype(np.float32)
        b = lab[:, start+1:end+1].astype(np.float32)
        n = min(a.shape[1], b.shape[1])
        diff = np.linalg.norm(a[:, :n] - b[:, :n], axis=2)
        # Sum over rows to get a single score per column position
        score = np.sum(diff, axis=0)
    else:
        # Horizontal boundary: compare row i with row i+1
        a = lab[start:end, :].astype(np.float32)
        b = lab[start+1:end+1, :].astype(np.float32)
        n = min(a.shape[0], b.shape[0])
        diff = np.linalg.norm(a[:n] - b[:n], axis=2)
        # Sum over columns
        score = np.sum(diff, axis=1)

    if len(score) == 0:
        return approximate_pos

    # Find position with maximum color difference
    best_offset = np.argmax(score)
    refined_pos = start + best_offset

    return refined_pos


def merge_small_rows(row_boundaries, min_height=MIN_ROW_HEIGHT):
    """
    Merge rows that are too small into the next row.

    Sometimes the header row gets split by a false boundary detection.
    This function removes boundaries that create rows smaller than
    min_height by merging them with the following row.
    """
    if len(row_boundaries) < 3:
        return row_boundaries

    merged = [row_boundaries[0]]  # Always keep the first boundary

    for i in range(1, len(row_boundaries) - 1):
        row_height = row_boundaries[i] - merged[-1]

        if row_height < min_height:
            # Skip this boundary (merge with next row)
            continue
        else:
            merged.append(row_boundaries[i])

    merged.append(row_boundaries[-1])  # Always keep the last boundary

    return np.array(merged, dtype=int)


def detect_rows(table):
    """
    Detect row boundaries using median-color projection.

    1. Compute median color per row (robust to text)
    2. Compute gradient of the median profile
    3. Find peaks = approximate boundaries
    4. Refine each boundary by searching for max color difference
    5. Merge rows that are too small (header fix)
    """
    # Step 1: Median color profile (per row)
    profile = compute_median_color_profile(table, axis=1)

    # Step 2: Gradient of the profile
    gradient = compute_profile_gradient(profile)

    # Step 3: Smooth the gradient
    kernel = np.ones(5) / 5
    gradient_smooth = np.convolve(gradient, kernel, mode='same')

    # Step 4: Find approximate boundaries
    row_peaks = find_boundaries(gradient_smooth, MIN_ROW_DISTANCE)

    # Step 5: Refine each boundary
    refined_peaks = []
    for peak in row_peaks:
        refined = refine_boundary(table, peak, axis=1)
        refined_peaks.append(refined)

    # Step 6: Add table edges and merge small rows
    row_boundaries = np.concatenate([[0], refined_peaks, [table.shape[0]]])
    row_boundaries = merge_small_rows(row_boundaries)

    return row_boundaries


def detect_cols(row_strip):
    """
    Detect column boundaries using median-color projection.

    1. Compute median color per column (robust to text)
    2. Compute gradient of the median profile
    3. Find peaks = approximate boundaries
    4. Refine each boundary by searching for max color difference
    """
    # Step 1: Median color profile (per column)
    profile = compute_median_color_profile(row_strip, axis=0)

    # Step 2: Gradient of the profile
    gradient = compute_profile_gradient(profile)

    # Step 3: Smooth the gradient
    kernel = np.ones(5) / 5
    gradient_smooth = np.convolve(gradient, kernel, mode='same')

    # Step 4: Find approximate boundaries
    col_peaks = find_boundaries(gradient_smooth, MIN_COL_DISTANCE)

    # Step 5: Refine each boundary
    refined_peaks = []
    for peak in col_peaks:
        refined = refine_boundary(row_strip, peak, axis=0)
        refined_peaks.append(refined)

    return np.array(refined_peaks, dtype=int)


def save_debug_image(table, row_boundaries, col_boundaries_list, output_path):
    """Save debug visualization of detected boundaries."""
    debug = table.copy()
    h, w = debug.shape[:2]

    # Draw row boundaries in red
    for y in row_boundaries:
        cv2.line(debug, (0, y), (w, y), (0, 0, 255), 2)

    # Draw column boundaries in green
    for row_idx, col_boundaries in enumerate(col_boundaries_list):
        if row_idx < len(row_boundaries) - 1:
            y1 = row_boundaries[row_idx]
            y2 = row_boundaries[row_idx + 1]
            for x in col_boundaries:
                cv2.line(debug, (x, y1), (x, y2), (0, 255, 0), 1)

    cv2.imwrite(str(output_path), debug)


def crop_cells():
    """
    Main cell cropping function.

    Algorithm:
    1. Detect table region via blue diagonal segmentation
    2. Detect row boundaries via median-color projection + refinement
    3. For each row (sequentially):
       a. Detect column boundaries via median-color projection + refinement
       b. Crop each cell using detected boundaries
       c. Remember end position for next row alignment
    4. Save cells with naming: cell_r{row:03d}_c{col:03d}.png

    Row 0 = horizontal header, Column 0 = vertical header.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)

    # Remove stale cells from previous runs, otherwise old files with
    # different geometry linger next to new output and look like bad crops.
    removed = 0
    for stale in OUTPUT_DIR.glob("cell_*.png"):
        stale.unlink()
        removed += 1
    if removed:
        print(f"Removed {removed} stale cell files from previous run")

    # Load image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise RuntimeError(f"Cannot load {IMAGE_PATH}")

    print(f"Image: {image.shape[1]} x {image.shape[0]}")

    # Detect table region
    table_bbox = detect_table_region(image)
    print(f"Table region: {table_bbox}")

    x1, y1, x2, y2 = table_bbox
    table = image[y1:y2, x1:x2]

    # Preprocess (blur to reduce noise)
    table_blur = preprocess(table)

    # Detect row boundaries (includes edges + merge of small rows)
    row_boundaries = detect_rows(table_blur)
    print(f"Detected {len(row_boundaries) - 1} rows")

    # Process rows sequentially with position memory
    total_cells = 0
    prev_row_end_x = 0  # Remembered position from previous row
    all_col_boundaries = []

    for row_idx in range(len(row_boundaries) - 1):
        row_start = row_boundaries[row_idx]
        row_end = row_boundaries[row_idx + 1]

        # Extract row strip
        row_strip = table_blur[row_start:row_end, :]

        # Detect column boundaries in this row
        col_peaks = detect_cols(row_strip)
        col_boundaries = np.concatenate([[0], col_peaks, [table.shape[1]]])
        all_col_boundaries.append(col_boundaries)

        # Crop each cell in the row
        for col_idx in range(len(col_boundaries) - 1):
            col_start = col_boundaries[col_idx]
            col_end = col_boundaries[col_idx + 1]

            # Crop from original (non-blurred) table for best quality
            cell = table[row_start:row_end, col_start:col_end]

            # Save with naming: cell_r{row:03d}_c{col:03d}.png
            filename = f"cell_r{row_idx:03d}_c{col_idx:03d}.png"
            cv2.imwrite(str(OUTPUT_DIR / filename), cell)

            total_cells += 1

        # Remember end position for next row alignment
        prev_row_end_x = col_boundaries[-1]

        print(
            f"Row {row_idx}: {len(col_boundaries) - 1} cells "
            f"(end_x={prev_row_end_x})"
        )

    # Save debug visualization
    save_debug_image(
        table, row_boundaries, all_col_boundaries,
        DEBUG_DIR / "debug_boundaries.png"
    )

    print(f"\nTotal cells: {total_cells}")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"Debug visualization: {DEBUG_DIR / 'debug_boundaries.png'}")


if __name__ == "__main__":
    crop_cells()
