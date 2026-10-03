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

# Calculate how much line exists at each row/column.
horizontal_score = np.sum(
    horizontal > 0,
    axis=1
)

# A row is probably a horizontal grid line if
# a large percentage of the row contains line pixels.
horizontal_threshold = width * 0.30

horizontal_positions = np.where(
    horizontal_score > horizontal_threshold
)[0]

# Merge neighboring detected pixels into one line.
horizontal_lines = merge_positions(
    horizontal_positions
)

#number_of_rows = len(horizontal_lines) - 1
number_of_rows = len(yellow_runs)

print(f"Estimated rows             : {number_of_rows}")

if number_of_rows > 0:
    cell_height = table_height / number_of_rows
else:
    cell_height = 0

cell_height = table_height / number_of_rows

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