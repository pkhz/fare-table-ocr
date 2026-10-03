from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_PATH = "image/faretable.png"
OUTPUT_DIR = Path("crop-output")


def inspect_image(path):
    """Read basic image information."""

    with Image.open(path) as img:
        print("=== IMAGE ===")
        print(f"Format      : {img.format}")
        print(f"Size        : {img.width} × {img.height}")
        print(f"Mode        : {img.mode}")
        print(f"DPI         : {img.info.get('dpi')}")

        print()


def load_grayscale(path):
    """
    Load image as grayscale.

    OpenCV loads the P-mode PNG and converts it to grayscale.
    """

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Cannot load image: {path}")

    return image


def find_table_bbox(gray):
    """
    Try to find the main table bounding box.

    This works best when the table contains visible
    horizontal/vertical lines.
    """

    # Binary image.
    # Dark pixels become white.
    _, binary = cv2.threshold(
        gray,
        220,
        255,
        cv2.THRESH_BINARY_INV
    )

    height, width = gray.shape

    # Detect horizontal lines.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, width // 20), 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Detect vertical lines.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, height // 20))
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # Combine detected table lines.
    table_lines = cv2.bitwise_or(horizontal, vertical)

    # Find connected components.
    contours, _ = cv2.findContours(
        table_lines,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        # Ignore tiny objects.
        if area < 1000:
            continue

        candidates.append((x, y, w, h, area))

    if not candidates:
        print("No obvious table detected.")
        return None

    # Bounding box containing all detected table-line objects.
    x1 = min(x for x, y, w, h, area in candidates)
    y1 = min(y for x, y, w, h, area in candidates)

    x2 = max(x + w for x, y, w, h, area in candidates)
    y2 = max(y + h for x, y, w, h, area in candidates)

    return x1, y1, x2, y2


def detect_grid_lines(gray, bbox):
    """
    Detect horizontal and vertical grid lines inside the table.
    """

    x1, y1, x2, y2 = bbox

    table = gray[y1:y2, x1:x2]

    # Binary.
    _, binary = cv2.threshold(
        table,
        220,
        255,
        cv2.THRESH_BINARY_INV
    )

    height, width = binary.shape

    # Horizontal lines.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, width // 30), 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Vertical lines.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, height // 30))
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # Calculate how much line exists at each row/column.
    horizontal_score = np.sum(
        horizontal > 0,
        axis=1
    )

    vertical_score = np.sum(
        vertical > 0,
        axis=0
    )

    # A row is probably a horizontal grid line if
    # a large percentage of the row contains line pixels.
    horizontal_threshold = width * 0.30
    vertical_threshold = height * 0.30

    horizontal_positions = np.where(
        horizontal_score > horizontal_threshold
    )[0]

    vertical_positions = np.where(
        vertical_score > vertical_threshold
    )[0]

    # Merge neighboring detected pixels into one line.
    horizontal_positions = merge_positions(
        horizontal_positions
    )

    vertical_positions = merge_positions(
        vertical_positions
    )

    cv2.imwrite(
        "output-debug/debug_horizontal.png",
        horizontal
    )

    cv2.imwrite(
        "output-debug/debug_vertical.png",
        vertical
    )

    cv2.imwrite(
        "output-debug/debug_table.png",
        table
    )

    return horizontal_positions, vertical_positions


def merge_positions(positions, max_gap=3):
    """
    Convert something like:

        [100,101,102,103,110,111]

    into:

        [101,110]

    representing the center of each detected line.
    """

    if len(positions) == 0:
        return np.array([], dtype=int)

    groups = []
    current = [positions[0]]

    for value in positions[1:]:

        if value - current[-1] <= max_gap:
            current.append(value)
        else:
            groups.append(current)
            current = [value]

    groups.append(current)

    return np.array([
        int(round(np.mean(group)))
        for group in groups
    ])


def calculate_grid_info(
    horizontal_lines,
    vertical_lines,
    bbox
):
    """
    Calculate number of rows/columns and average cell size.
    """

    x1, y1, x2, y2 = bbox

    table_width = x2 - x1
    table_height = y2 - y1

    print("=== TABLE ===")
    print(f"Bounding box : ({x1}, {y1}) → ({x2}, {y2})")
    print(f"Table size   : {table_width} × {table_height}")
    print()

    print("=== GRID ===")

    print(
        f"Horizontal lines detected : "
        f"{len(horizontal_lines)}"
    )

    print(
        f"Vertical lines detected   : "
        f"{len(vertical_lines)}"
    )

    # N lines create N-1 cells.
    rows = len(horizontal_lines) - 1
    columns = len(vertical_lines) - 1

    print(f"Estimated rows             : {rows}")
    print(f"Estimated columns          : {columns}")

    if rows > 0:
        cell_height = table_height / rows
    else:
        cell_height = 0

    if columns > 0:
        cell_width = table_width / columns
    else:
        cell_width = 0

    print(
        f"Average cell size         : "
        f"{cell_width:.2f} × {cell_height:.2f} px"
    )

    return rows, columns, cell_width, cell_height


def choose_block_size(cell_width, cell_height):
    """
    Choose how many cells to process in one OCR block.

    This is only a starting heuristic.
    """

    cell_size = min(cell_width, cell_height)

    if cell_size < 8:
        block = 50

    elif cell_size < 15:
        block = 100

    elif cell_size < 30:
        block = 200

    elif cell_size < 60:
        block = 300

    else:
        block = 500

    print()
    print("=== OCR BLOCK ===")
    print(f"Selected block : {block} × {block} cells")

    return block


def crop_blocks(
    image_path,
    bbox,
    rows,
    columns,
    block_rows,
    block_columns,
    output_dir
):
    """
    Crop the table into blocks.

    Example:

        1000 × 1000 cells
        block = 200 × 200

    gives:

        5 × 5 = 25 blocks
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with Image.open(image_path) as img:

        # Keep original image mode for now.
        x1, y1, x2, y2 = bbox

        table_width = x2 - x1
        table_height = y2 - y1

        cell_width = table_width / columns
        cell_height = table_height / rows

        block_count = 0

        for row_start in range(
            0,
            rows,
            block_rows
        ):

            row_end = min(
                row_start + block_rows,
                rows
            )

            for col_start in range(
                0,
                columns,
                block_columns
            ):

                col_end = min(
                    col_start + block_columns,
                    columns
                )

                # Convert cell coordinates → pixel coordinates.
                px1 = round(
                    x1 + col_start * cell_width
                )

                py1 = round(
                    y1 + row_start * cell_height
                )

                px2 = round(
                    x1 + col_end * cell_width
                )

                py2 = round(
                    y1 + row_end * cell_height
                )

                crop = img.crop(
                    (px1, py1, px2, py2)
                )

                filename = (
                    f"block_"
                    f"r{row_start}-{row_end}_"
                    f"c{col_start}-{col_end}.png"
                )

                crop.save(
                    output_dir / filename
                )

                block_count += 1

                print(
                    f"Created {filename} "
                    f"({crop.width} × {crop.height}px)"
                )

        print()
        print(f"Total blocks: {block_count}")


def main():

    # --------------------------------------------------
    # 1. Inspect image
    # --------------------------------------------------

    inspect_image(IMAGE_PATH)

    # --------------------------------------------------
    # 2. Load grayscale
    # --------------------------------------------------

    gray = load_grayscale(IMAGE_PATH)

    print(
        f"Loaded grayscale: "
        f"{gray.shape[1]} × {gray.shape[0]}"
    )

    # --------------------------------------------------
    # 3. Find table
    # --------------------------------------------------

    bbox = find_table_bbox(gray)

    if bbox is None:
        print()
        print(
            "Automatic table detection failed."
        )
        print(
            "Set bbox manually instead."
        )

        # Example:
        #
        # bbox = (
        #     200,   # x1
        #     300,   # y1
        #     4800,  # x2
        #     4400   # y2
        # )

        return

    # --------------------------------------------------
    # 4. Detect grid
    # --------------------------------------------------

    horizontal_lines, vertical_lines = (
        detect_grid_lines(
            gray,
            bbox
        )
    )

    # --------------------------------------------------
    # 5. Determine rows/columns
    # --------------------------------------------------

    rows, columns, cell_width, cell_height = (
        calculate_grid_info(
            horizontal_lines,
            vertical_lines,
            bbox
        )
    )

    if rows <= 0 or columns <= 0:
        print(
            "Could not determine table dimensions."
        )
        return

    # Debug: visualize detected grid lines
    debug = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    x1, y1, x2, y2 = bbox

    for y in horizontal_lines:
        cv2.line(
            debug,
            (x1, y1 + y),
            (x2, y1 + y),
            (0, 0, 255),
            2
        )

    for x in vertical_lines:
        cv2.line(
            debug,
            (x1 + x, y1),
            (x1 + x, y2),
            (0, 255, 0),
            2
        )

    cv2.imwrite(
        "output-debug/debug_detected_grid.png",
        debug
    )

    # --------------------------------------------------
    # 6. Choose OCR block size
    # --------------------------------------------------

    # block_size = choose_block_size(
    #     cell_width,
    #     cell_height
    # )

    # --------------------------------------------------
    # 7. Crop blocks
    # --------------------------------------------------

    # crop_blocks(
    #     IMAGE_PATH,
    #     bbox,
    #     rows,
    #     columns,
    #     block_size,
    #     block_size,
    #     OUTPUT_DIR
    # )


if __name__ == "__main__":
    main()