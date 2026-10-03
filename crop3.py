from PIL import Image
from pathlib import Path
import math


IMAGE_PATH = "image/faretable.png"
OUTPUT_DIR = Path("output-crop")

# Detected matrix information
MATRIX_LEFT = 286
MATRIX_TOP = 113
MATRIX_RIGHT = 4930
MATRIX_BOTTOM = 4525

ROWS = 157
COLS = 157

# Derived from the bbox so blocks tile exactly to MATRIX_RIGHT/BOTTOM
# (hardcoding these separately causes drift that cuts off later blocks).
CELL_WIDTH = (MATRIX_RIGHT - MATRIX_LEFT) / COLS
CELL_HEIGHT = (MATRIX_BOTTOM - MATRIX_TOP) / ROWS

# Number of cells per crop
BLOCK_ROWS = 50
BLOCK_COLS = 50


def crop_blocks():
    OUTPUT_DIR.mkdir(exist_ok=True)

    img = Image.open(IMAGE_PATH)

    print(f"Image: {img.size}")
    print(f"Matrix: {ROWS} × {COLS}")
    print(f"Cell: {CELL_WIDTH:.2f} × {CELL_HEIGHT:.2f}px")

    block_count = 0

    for row_start in range(0, ROWS, BLOCK_ROWS):

        row_end = min(row_start + BLOCK_ROWS, ROWS)

        for col_start in range(0, COLS, BLOCK_COLS):

            col_end = min(col_start + BLOCK_COLS, COLS)

            # Convert cell coordinates → pixel coordinates
            x1 = round(
                MATRIX_LEFT +
                col_start * CELL_WIDTH
            )

            y1 = round(
                MATRIX_TOP +
                row_start * CELL_HEIGHT
            )

            x2 = round(
                MATRIX_LEFT +
                col_end * CELL_WIDTH
            )

            y2 = round(
                MATRIX_TOP +
                row_end * CELL_HEIGHT
            )

            crop = img.crop((x1, y1, x2, y2))

            filename = (
                f"block_"
                f"r{row_start:03d}-{row_end - 1:03d}_"
                f"c{col_start:03d}-{col_end - 1:03d}.png"
            )

            output_path = OUTPUT_DIR / filename

            crop.save(output_path)

            block_count += 1

            print(
                f"{filename} "
                f"{crop.size[0]}×{crop.size[1]} px"
            )

    img.close()

    print()
    print(f"Created {block_count} blocks.")


if __name__ == "__main__":
    crop_blocks()