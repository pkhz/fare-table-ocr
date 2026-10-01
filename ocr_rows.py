"""Hybrid OCR for the fare matrix.

The row-wide pass is fast and gets many values on the first sweep. A
second pass re-OCRs only the missing cells with a wider crop and extra
whitespace so the few noisy gaps are recovered without paying the full
per-cell cost of the original brute-force approach.

Row values are placed by their actual x-position (from Tesseract's word
boxes) rather than by counting tokens in order, so one misread/merged
value can no longer shift every later cell in the row out of alignment.
"""

import csv
import json
import re
import sys
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageOps

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = "image/faretable.png"
OUTPUT_DIR = Path("output-ocr")

MATRIX_LEFT = 286
MATRIX_TOP = 113
MATRIX_RIGHT = 4930
MATRIX_BOTTOM = 4525

ROWS = 157
COLS = 157

CELL_WIDTH = (MATRIX_RIGHT - MATRIX_LEFT) / COLS
CELL_HEIGHT = (MATRIX_BOTTOM - MATRIX_TOP) / ROWS

STRIP_SCALE = 3

TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789."
VALUE_PATTERN = re.compile(r"\d\.\d\d")
LOCAL_CONFIGS = [
    "--psm 7 -c tessedit_char_whitelist=0123456789.",
    "--psm 8 -c tessedit_char_whitelist=0123456789.",
    "--psm 10 -c tessedit_char_whitelist=0123456789.",
]



def normalize_value(text):
    cleaned = re.sub(r"[^0-9.]", "", (text or "").strip())
    if not cleaned:
        return None

    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    if cleaned.endswith("."):
        cleaned += "0"

    if re.fullmatch(r"\d+\.\d+", cleaned):
        return cleaned

    # Common Tesseract artifact: "130" means "1.30" when operating on a small
    # fare cell. Keep only 3-digit runs that map mechanically to D.DD.
    if re.fullmatch(r"\d{3}", cleaned):
        return f"{cleaned[0]}.{cleaned[1:]}"

    if re.fullmatch(r"\d{2}", cleaned):
        return f"{cleaned[0]}.{cleaned[1]}"

    return None


def ocr_row(img, row, cols=COLS):
    y1 = round(MATRIX_TOP + row * CELL_HEIGHT)
    y2 = round(MATRIX_TOP + (row + 1) * CELL_HEIGHT)
    x1 = MATRIX_LEFT
    x2 = round(MATRIX_LEFT + cols * CELL_WIDTH)

    strip = img.crop((round(x1), y1, x2, y2)).copy()

    diag_col = row if 0 <= row < cols else None
    if diag_col is not None:
        dx1 = round(diag_col * CELL_WIDTH)
        dx2 = round((diag_col + 1) * CELL_WIDTH)
        ImageDraw.Draw(strip).rectangle([dx1, 0, dx2, strip.height], fill=255)

    return strip.resize((strip.width * STRIP_SCALE, strip.height * STRIP_SCALE), Image.LANCZOS)


def enhance_cell(cell):
    """CLAHE + adaptive threshold fallback for cells too faint for plain
    autocontrast, since low-contrast digits sit on varying colored backgrounds."""
    arr = np.array(cell)
    arr = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(arr)
    arr = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(arr)


def ocr_cell(img, row, col, margin=14):
    x1 = MATRIX_LEFT + col * CELL_WIDTH - margin
    y1 = MATRIX_TOP + row * CELL_HEIGHT - margin
    x2 = MATRIX_LEFT + (col + 1) * CELL_WIDTH + margin
    y2 = MATRIX_TOP + (row + 1) * CELL_HEIGHT + margin

    cell = img.crop((
        max(0, int(x1)),
        max(0, int(y1)),
        min(img.width, int(x2)),
        min(img.height, int(y2)),
    )).convert("L")

    base = ImageOps.autocontrast(cell)
    base = ImageOps.expand(base, border=12, fill=255)
    base = base.resize((base.width * 5, base.height * 5), Image.LANCZOS)

    for candidate in (base, enhance_cell(base)):
        for cfg in LOCAL_CONFIGS:
            text = pytesseract.image_to_string(candidate, config=cfg).strip()
            value = normalize_value(text)
            if value is not None:
                return value

    return None


def ocr_span(img, row, start_col, end_col, margin=6):
    """OCR a short run of consecutive missing columns as one strip, so a
    cluster of gaps costs one Tesseract call instead of one per cell."""
    x1 = MATRIX_LEFT + start_col * CELL_WIDTH - margin
    y1 = MATRIX_TOP + row * CELL_HEIGHT - margin
    x2 = MATRIX_LEFT + (end_col + 1) * CELL_WIDTH + margin
    y2 = MATRIX_TOP + (row + 1) * CELL_HEIGHT + margin

    span = img.crop((
        max(0, int(x1)),
        max(0, int(y1)),
        min(img.width, int(x2)),
        min(img.height, int(y2)),
    )).convert("L")

    span = ImageOps.autocontrast(span)
    span = span.resize((span.width * STRIP_SCALE, span.height * STRIP_SCALE), Image.LANCZOS)

    def collect(strip_img):
        data = pytesseract.image_to_data(strip_img, config=TESSERACT_CONFIG, output_type=Output.DICT)
        found = {}
        for word, left, width in zip(data["text"], data["left"], data["width"]):
            word = word.strip()
            if not word:
                continue
            for match in VALUE_PATTERN.finditer(word):
                frac_center = (match.start() + match.end()) / 2 / len(word)
                x_orig = (left + frac_center * width) / STRIP_SCALE - margin
                col = start_col + int(x_orig / CELL_WIDTH)
                if start_col <= col <= end_col and col not in found:
                    found[col] = match.group()
        return found

    values = collect(span)

    # Retry with CLAHE + adaptive threshold only for columns still missing,
    # since the plain-autocontrast pass misses genuinely low-contrast digits.
    if len(values) < (end_col - start_col + 1):
        for col, value in collect(enhance_cell(span)).items():
            values.setdefault(col, value)

    return values


def fill_row_from_ocr(img, row, cols=COLS):
    values = [None] * cols
    strip = ocr_row(img, row, cols)
    data = pytesseract.image_to_data(strip, config=TESSERACT_CONFIG, output_type=Output.DICT)

    diag_col = row if row < cols else None

    for word, left, width in zip(data["text"], data["left"], data["width"]):
        word = word.strip()
        if not word:
            continue

        # Position each match by where it sits inside the word's bounding
        # box, so a merged "1.301.40" still lands as two separate cells.
        for match in VALUE_PATTERN.finditer(word):
            frac_center = (match.start() + match.end()) / 2 / len(word)
            x_orig = (left + frac_center * width) / STRIP_SCALE
            col = int(x_orig / CELL_WIDTH)

            if col == diag_col or not (0 <= col < cols):
                continue
            if values[col] is None:
                values[col] = match.group()

    return values


def ocr_matrix(rows=ROWS, cols=COLS):
    OUTPUT_DIR.mkdir(exist_ok=True)
    img = Image.open(IMAGE_PATH).convert("L")

    print(f"Image: {img.size}")
    print(f"Matrix: {rows} x {cols} (of {ROWS} x {COLS})")

    values = [[None] * cols for _ in range(rows)]

    for row in range(rows):
        row_values = fill_row_from_ocr(img, row, cols)

        # Only re-OCR gaps inside the row's populated span -- past the
        # last found value the table is legitimately blank (no fare), so
        # re-checking that whole empty tail cell-by-cell is wasted work.
        found_cols = [idx for idx, val in enumerate(row_values) if val is not None]
        last_col = max(found_cols) if found_cols else -1
        missing = [
            idx for idx, val in enumerate(row_values)
            if val is None and idx <= last_col
        ]
        if missing:
            print(f"Row {row + 1}/{rows}: rechecking {len(missing)} gap cells")

        # Group consecutive gap columns into spans and OCR each span with a
        # single Tesseract call instead of one call per cell.
        for _, group in groupby(enumerate(missing), lambda pair: pair[1] - pair[0]):
            span = [col for _, col in group]
            span_values = ocr_span(img, row, span[0], span[-1])
            for col in span:
                if col in span_values:
                    row_values[col] = span_values[col]

        # Last-resort single-cell OCR for any span that still came up empty.
        for col in missing:
            if row_values[col] is None:
                value = ocr_cell(img, row, col)
                if value is not None:
                    row_values[col] = value

        values[row] = row_values

    return values


def save_csv(values, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(values)


def save_json(values, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)


if __name__ == "__main__":
    limit_rows, limit_cols = ROWS, COLS

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "x" in arg:
            r, c = arg.split("x")
            limit_rows, limit_cols = int(r), int(c)
        else:
            limit_rows = limit_cols = int(arg)

    values = ocr_matrix(limit_rows, limit_cols)

    save_csv(values, OUTPUT_DIR / "fare_matrix_hybrid.csv")
    save_json(values, OUTPUT_DIR / "fare_matrix_hybrid.json")

    print(f"Saved {OUTPUT_DIR / 'fare_matrix_hybrid.csv'}")
    print(f"Saved {OUTPUT_DIR / 'fare_matrix_hybrid.json'}")
