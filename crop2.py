import cv2
import numpy as np

IMAGE_PATH = "image/faretable.png"

image = cv2.imread("image/faretable.png")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Yellow background
lower = np.array([20, 100, 150])
upper = np.array([40, 255, 255])

yellow = cv2.inRange(
    hsv,
    lower,
    upper
)

# Count yellow pixels on every horizontal row
yellow_count = np.sum(yellow > 0, axis=1)

# A row containing lots of yellow pixels
is_yellow_row = yellow_count > image.shape[1] * 0.25

#group consecutive yellow rows into bands
def find_runs(mask):
    runs = []

    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(mask) - 1))

    return runs

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

def find_matrix_extent(
    image_path,
    rows,
    table_bbox
):
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Cannot load {image_path}"
        )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_blue = np.array([
        90, 80, 80
    ])

    upper_blue = np.array([
        140, 255, 255
    ])

    blue = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # Restrict detection to table.
    x1, y1, x2, y2 = table_bbox

    table_blue = blue[
        y1:y2,
        x1:x2
    ]

    ys, xs = np.where(
        table_blue > 0
    )

    if len(xs) == 0:
        raise RuntimeError(
            "Blue diagonal not detected"
        )

    # Convert table-relative coordinates
    # back to full-image coordinates.
    blue_x1 = x1 + xs.min()
    blue_x2 = x1 + xs.max()

    blue_y1 = y1 + ys.min()
    blue_y2 = y1 + ys.max()

    print("=== MATRIX ===")

    print(
        f"Blue extent : "
        f"({blue_x1}, {blue_y1}) → "
        f"({blue_x2}, {blue_y2})"
    )

    # Because this is a square fare matrix.
    columns = rows

    # Approximate cell size.
    matrix_width = blue_x2 - blue_x1

    cell_width = (
        matrix_width / (columns - 1)
        if columns > 1
        else 0
    )

    print(f"Rows        : {rows}")
    print(f"Columns     : {columns}")
    print(
        f"Approx cell width : "
        f"{cell_width:.2f}px"
    )

    return {
        "left": blue_x1,
        "right": blue_x2,
        "top": blue_y1,
        "bottom": blue_y2,
        "rows": rows,
        "columns": columns,
        "cell_width": cell_width,
    }

yellow_runs = find_runs(is_yellow_row)

print("Yellow bands:", len(yellow_runs))

for start, end in yellow_runs[:10]:
    print(start, end)

    centers = [
    (start + end) / 2
    for start, end in yellow_runs
]

spacing = np.diff(centers)

print("Average yellow-row spacing:",
      np.median(spacing))

gray = load_grayscale(IMAGE_PATH)

bbox = find_table_bbox(gray)

x1, y1, x2, y2 = bbox

table = gray[y1:y2, x1:x2]

table_height = y2 - y1

#number_of_rows = len(horizontal_lines) - 1
#number_of_rows = len(yellow_runs)
#number_of_rows = len(yellow_runs)

#print(f"Estimated rows             : {number_of_rows}")

yellow_height = end - start + 1

row_pitch = np.median(spacing) / 2
rows = round(table_height / row_pitch)

print("Row pitch:", row_pitch)
print("Rows:", rows)

for i, (start, end) in enumerate(yellow_runs[:10]):
    next_start = yellow_runs[:10][i + 1][0] if i < len(yellow_runs[:10]) - 1 else None
    print(
        f"Yellow {i}: "
        f"{start} → {end}, "
        f"height={end-start+1}, "
        f"spacing={next_start - start if i < len(yellow_runs[:10]) - 1 else 0}"
    )

matrix = find_matrix_extent("image/faretable.png", rows, bbox)

blue_x1 = matrix["left"]
blue_x2 = matrix["right"]
blue_y1 = matrix["top"]
blue_y2 = matrix["bottom"]
rows = matrix["rows"]
columns = matrix["columns"]
cell_width = matrix["cell_width"]

# if number_of_rows > 0:
#     cell_height = table_height / number_of_rows
# else:
#     cell_height = 0

# cell_height = table_height / number_of_rows

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



