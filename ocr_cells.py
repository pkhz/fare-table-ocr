"""
OCR the fare matrix cell-by-cell using the geometry already computed by
crop2.py/crop3.py, instead of feeding the whole colored table to a
generic table-structure OCR tool (img2table). Those tools rely on
detecting drawn gridlines; this table separates cells by background
color instead, which breaks their cell/column detection.
"""

import csv
import json
import sys
from pathlib import Path

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = "image/faretable.png"
OUTPUT_DIR = Path("output-ocr")

# Same matrix geometry as crop3.py
MATRIX_LEFT = 286
MATRIX_TOP = 113
MATRIX_RIGHT = 4930
MATRIX_BOTTOM = 4525

ROWS = 157
COLS = 157

# MATRIX_LEFT/TOP/RIGHT/BOTTOM are the grid's outer edges (same geometry
# validated in crop3.py) -- not diagonal-cell centers, so no half-cell shift.
CELL_WIDTH = (MATRIX_RIGHT - MATRIX_LEFT) / COLS
CELL_HEIGHT = (MATRIX_BOTTOM - MATRIX_TOP) / ROWS

# Fare values only contain digits and a decimal point.
TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789."


def cell_bbox(row, col):
    x1 = round(MATRIX_LEFT + col * CELL_WIDTH)
    y1 = round(MATRIX_TOP + row * CELL_HEIGHT)
    x2 = round(MATRIX_LEFT + (col + 1) * CELL_WIDTH)
    y2 = round(MATRIX_TOP + (row + 1) * CELL_HEIGHT)
    return x1, y1, x2, y2


def ocr_matrix(rows=ROWS, cols=COLS):
    OUTPUT_DIR.mkdir(exist_ok=True)

    img = Image.open(IMAGE_PATH).convert("L")

    print(f"Image: {img.size}")
    print(f"Matrix: {rows} x {cols} (of {ROWS} x {COLS})")
    print(f"Cell: {CELL_WIDTH:.2f} x {CELL_HEIGHT:.2f}px")

    values = [[None] * cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            x1, y1, x2, y2 = cell_bbox(row, col)

            # Inset slightly so neighboring cells' digits don't bleed in.
            inset = 1
            cell = img.crop((
                x1 + inset,
                y1 + inset,
                x2 - inset,
                y2 - inset,
            ))

            # Upscale small cells so Tesseract has enough pixels to work with.
            cell = cell.resize(
                (cell.width * 3, cell.height * 3),
                Image.LANCZOS,
            )

            text = pytesseract.image_to_string(
                cell,
                config=TESSERACT_CONFIG,
            ).strip()

            values[row][col] = text

        print(f"Row {row + 1}/{rows} done")

    return values


def save_csv(values, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(values)


def save_json(values, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)


if __name__ == "__main__":
    # Optional "N" or "ROWSxCOLS" arg to OCR only a corner for a quick test
    # before committing to the full 157x157 pass.
    limit_rows, limit_cols = ROWS, COLS

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "x" in arg:
            r, c = arg.split("x")
            limit_rows, limit_cols = int(r), int(c)
        else:
            limit_rows = limit_cols = int(arg)

    values = ocr_matrix(limit_rows, limit_cols)

    save_csv(values, OUTPUT_DIR / "fare_matrix.csv")
    save_json(values, OUTPUT_DIR / "fare_matrix.json")

    print(f"Saved {OUTPUT_DIR / 'fare_matrix.csv'}")
    print(f"Saved {OUTPUT_DIR / 'fare_matrix.json'}")
