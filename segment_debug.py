"""
Segment cells and save cropped character images to output-segmentation/
for manual inspection. Does NOT run template matching yet.

Usage:
    python segment_debug.py              # process 5x5 corner
    python segment_debug.py --all        # process all cells (slow)
    python segment_debug.py --rows 0 1 2 # process specific rows
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from template_match import normalize_cell, normalize_blue_cell, is_blue_cell, CELLS_DIR

OUT_DIR = Path("output-segmentation")


def crop_to_text(cell_img):
    """Crop binary image to the text bounding box."""
    coords = cv2.findNonZero(cell_img)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    # Add 1px padding
    y1 = max(0, y - 1)
    y2 = min(cell_img.shape[0], y + h + 1)
    x1 = max(0, x - 1)
    x2 = min(cell_img.shape[1], x + w + 1)
    return cell_img[y1:y2, x1:x2]


def segment_by_projection(cell_img):
    """Segment cropped cell into characters using vertical projection."""
    proj = np.sum(cell_img, axis=0)
    if np.max(proj) == 0:
        return []

    threshold = np.max(proj) * 0.05
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

    # Extract character images
    h = cell_img.shape[0]
    chars = []
    for x1, x2 in bounds:
        chars.append(cell_img[0:h, x1:x2])
    return chars


def process_cell(r, c):
    """Process one cell: crop, segment, save to output-segmentation/."""
    path = CELLS_DIR / f"cell_r{r:03d}_c{c:03d}.png"
    img = cv2.imread(str(path))
    if img is None:
        return None

    # Normalize to binary
    if is_blue_cell(path):
        cell = normalize_blue_cell(img)
    else:
        cell = normalize_cell(img)

    # Crop to text bounding box
    cropped = crop_to_text(cell)
    if cropped is None:
        return {"r": r, "c": c, "error": "no text found"}

    # Segment
    chars = segment_by_projection(cropped)
    if not chars:
        return {"r": r, "c": c, "error": "no chars segmented", "cropped_shape": cropped.shape}

    # Save results
    cell_dir = OUT_DIR / f"cell_r{r:03d}_c{c:03d}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Save cropped cell
    cv2.imwrite(str(cell_dir / "cropped.png"), cropped)

    # Save each character
    for i, char_img in enumerate(chars):
        cv2.imwrite(str(cell_dir / f"char_{i:02d}.png"), char_img)

    return {
        "r": r, "c": c,
        "cropped_shape": cropped.shape,
        "num_chars": len(chars),
        "char_shapes": [img.shape for img in chars],
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Determine which cells to process
    if "--all" in sys.argv:
        # Find all cell files
        cells = sorted(CELLS_DIR.glob("cell_r*_c*.png"))
        targets = []
        for p in cells:
            name = p.stem  # cell_r000_c001
            parts = name.split("_")
            r = int(parts[1][1:])
            c = int(parts[2][1:])
            targets.append((r, c))
    elif "--rows" in sys.argv:
        idx = sys.argv.index("--rows")
        rows = [int(x) for x in sys.argv[idx+1:]]
        targets = [(r, c) for r in rows for c in range(160)]
    else:
        # Default: 5x5 corner
        targets = [(r, c) for r in range(5) for c in range(5)]

    print(f"Processing {len(targets)} cells -> {OUT_DIR}/")
    results = []
    for r, c in targets:
        result = process_cell(r, c)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Results for {len(results)} cells:")
    print(f"{'='*60}")

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    print(f"OK: {len(ok)}, Errors: {len(errors)}")

    if errors:
        print(f"\nErrors:")
        for e in errors[:10]:
            print(f"  r{e['r']:03d}c{e['c']:03d}: {e['error']}")

    if ok:
        print(f"\nSample OK results:")
        for r in ok[:5]:
            print(f"  r{r['r']:03d}c{r['c']:03d}: {r['num_chars']} chars, "
                  f"cropped={r['cropped_shape']}, shapes={r['char_shapes']}")

    # Distribution of char counts
    from collections import Counter
    char_counts = Counter(r["num_chars"] for r in ok)
    print(f"\nCharacter count distribution:")
    for count, freq in sorted(char_counts.items()):
        print(f"  {count} chars: {freq} cells")


if __name__ == "__main__":
    main()
