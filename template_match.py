"""
Template matching OCR for fare table cells.

Builds digit templates from high-confidence cells (5x5 corner),
then classifies cells by matching against templates.

Usage:
    python template_match.py              # test on sample cells
    python template_match.py --benchmark  # benchmark on all cells
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from cells_to_csv import is_blue_cell, load_variants, ocr_fare

CELLS_DIR = Path("output-cells")
TEMPLATE_SIZE = (20, 20)  # normalized digit size


def normalize_cell(img):
    """
    Preprocess a cell image for segmentation.
    Returns a binary image where text is white (255) on black (0).
    """
    # Trim border
    h, w = img.shape[:2]
    if h > 4 and w > 4:
        img = img[2:h-2, 2:w-2]
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Threshold: text is dark -> invert so text is white
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def normalize_blue_cell(img):
    """
    Preprocess a blue diagonal cell (white text on blue).
    Returns a binary image where text is white (255) on black (0).
    """
    h, w = img.shape[:2]
    if h > 4 and w > 4:
        img = img[2:h-2, 2:w-2]
    mn = img.min(axis=2)
    # White text: all channels high -> white text on black background
    mask = np.where(mn >= 150, 255, 0).astype(np.uint8)
    return mask


def build_templates():
    """
    Build digit templates from high-confidence cells.
    Uses the 5x5 corner plus the first row and diagonal to cover all digits.
    """
    # Known values: 5x5 corner + first row (r0, c0-20) + diagonal (r0-20)
    known = {}
    corner = [
        ["0.80", "1.30", "1.90", "2.00", "2.80"],
        ["1.30", "0.80", "1.40", "1.70", "2.40"],
        ["1.90", "1.40", "0.80", "1.70", "2.00"],
        ["2.00", "1.70", "1.70", "0.80", "1.60"],
        ["2.80", "2.40", "2.00", "1.60", "0.80"],
    ]
    for r in range(5):
        for c in range(5):
            known[(r, c)] = corner[r][c]

    # First row values (from verified OCR + GTFS)
    with open("output-ocr/fare_matrix_cells.json", encoding="utf-8") as f:
        grid = json.load(f)
    for c in range(min(30, len(grid[0]))):
        known[(0, c)] = grid[0][c]
        known[(c, 0)] = grid[c][0]

    templates = {}  # digit -> list of template images

    for (r, c), value in known.items():
        path = CELLS_DIR / f"cell_r{r:03d}_c{c:03d}.png"
        img = cv2.imread(str(path))
        if img is None:
            continue

        if is_blue_cell(path):
            cell = normalize_blue_cell(img)
        else:
            cell = normalize_cell(img)

        chars = segment_chars(cell)
        if chars is None:
            continue

        # Parse value into characters, matching to segmented groups
        # Value format: D.DD or DD.DD
        # Segmented groups: digit, dot+digit, digit (for D.DD)
        #                   digit, digit, dot+digit, digit (for DD.DD)
        # We need to map groups to value characters
        value_chars = list(value)  # e.g., ['1', '.', '4', '0']

        # Simple approach: if 3 groups, value is D.DD (dot merged with second digit)
        # If 4 groups, value is DD.DD (dot is separate)
        if len(chars) == 3 and len(value_chars) == 4:
            # D.DD: groups are [digit, dot+digit, digit]
            # Map: group0 -> value[0], group1 -> value[2], group2 -> value[3]
            mapping = [(0, 0), (1, 2), (2, 3)]
        elif len(chars) == 4 and len(value_chars) == 4:
            # DD.DD: groups are [digit, digit, dot, digit] or [digit, digit, dot+digit, digit]
            # Try: group0 -> value[0], group1 -> value[1], group2 -> value[3], group3 -> value[4]?
            # Actually for DD.DD: value_chars = ['1','1','.','1','0'] (5 chars)
            mapping = [(0, 0), (1, 1), (2, 3), (3, 4)]
        elif len(chars) == len(value_chars):
            mapping = list(enumerate(range(len(value_chars))))
        else:
            continue

        for group_idx, char_idx in mapping:
            if char_idx >= len(value_chars):
                continue
            char_val = value_chars[char_idx]
            if char_val == ".":
                continue
            if char_val not in templates:
                templates[char_val] = []
            templates[char_val].append(chars[group_idx])

    # Average templates for each digit
    avg_templates = {}
    for digit, imgs in templates.items():
        if imgs:
            resized = [cv2.resize(img, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA) for img in imgs]
            avg = np.mean(resized, axis=0).astype(np.uint8)
            avg_templates[digit] = avg

    return avg_templates


def segment_chars(cell_img):
    """
    Segment a binary cell image (text=white on black) into individual characters
    using vertical projection. Returns list of character images or None.
    """
    # Vertical projection: sum of white pixels per column
    proj = np.sum(cell_img, axis=0)

    # Find character boundaries (columns with text)
    threshold = np.max(proj) * 0.05 if np.max(proj) > 0 else 1
    in_char = False
    bounds = []
    for i, v in enumerate(proj):
        if v > threshold and not in_char:
            start = i
            in_char = True
        elif v <= threshold and in_char:
            bounds.append((start, i))
            in_char = False
    if in_char:
        bounds.append((start, len(proj)))

    if len(bounds) < 3:  # at least D.DD (3 groups: digit, dot+digit, digit)
        return None

    # Extract character images
    h = cell_img.shape[0]
    chars = []
    for x1, x2 in bounds:
        char_img = cell_img[0:h, x1:x2]
        chars.append(char_img)

    return chars


def match_char(char_img, templates):
    """Match a character image against templates. Returns best digit."""
    if not templates:
        return "?"

    # Resize char to template size
    char_resized = cv2.resize(char_img, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)

    best_digit = "?"
    best_score = -1

    for digit, template in templates.items():
        # Normalized cross-correlation
        score = cv2.matchTemplate(char_resized, template, cv2.TM_CCOEFF_NORMED)[0][0]
        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit


def ocr_cell_template(path, templates):
    """OCR a cell using template matching."""
    img = cv2.imread(str(path))
    if img is None:
        return ""

    if is_blue_cell(path):
        cell = normalize_blue_cell(img)
    else:
        cell = normalize_cell(img)

    chars = segment_chars(cell)
    if chars is None:
        return ""

    # Classify each character
    result = ""
    for char_img in chars:
        digit = match_char(char_img, templates)
        result += digit

    # Try to format as fare
    return ocr_fare(result)


def test_on_samples():
    """Test template matching on sample cells."""
    print("Building templates from 5x5 corner...")
    templates = build_templates()
    print(f"Built {len(templates)} templates: {sorted(templates.keys())}")

    # Test on the 5x5 corner (should be perfect)
    print("\nTesting on 5x5 corner:")
    corner_values = [
        ["0.80", "1.30", "1.90", "2.00", "2.80"],
        ["1.30", "0.80", "1.40", "1.70", "2.40"],
        ["1.90", "1.40", "0.80", "1.70", "2.00"],
        ["2.00", "1.70", "1.70", "0.80", "1.60"],
        ["2.80", "2.40", "2.00", "1.60", "0.80"],
    ]
    correct = 0
    total = 0
    for r in range(5):
        for c in range(5):
            path = CELLS_DIR / f"cell_r{r:03d}_c{c:03d}.png"
            result = ocr_cell_template(path, templates)
            expected = corner_values[r][c]
            ok = "OK" if result == expected else "FAIL"
            if result == expected:
                correct += 1
            total += 1
            print(f"  r{r}c{c}: {result} (expected {expected}) {ok}")
    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    # Test on some previously problematic cells
    print("\nTesting on previously problematic cells:")
    test_cells = [
        (0, 9, "3.50"), (0, 13, "3.90"), (0, 26, "4.60"),
        (13, 118, "4.30"), (20, 20, "0.80"), (151, 151, "1.50"),
        (84, 111, "11.10"), (87, 151, "10.70"),
    ]
    for r, c, expected in test_cells:
        path = CELLS_DIR / f"cell_r{r:03d}_c{c:03d}.png"
        result = ocr_cell_template(path, templates)
        ok = "OK" if result == expected else "FAIL"
        print(f"  r{r}c{c}: {result} (expected {expected}) {ok}")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        print("Benchmark mode not yet implemented")
    else:
        test_on_samples()
