from PIL import Image
import os

path = "image/faretable.png"

with Image.open(path) as img:
    width, height = img.size
    mode = img.mode
    format = img.format
    dpi = img.info.get("dpi")

file_size = os.path.getsize(path)

bytes_per_pixel = {
    "1": 1/8,
    "L": 1,
    "RGB": 3,
    "RGBA": 4,
}.get(mode)

raw_size = width * height * bytes_per_pixel if bytes_per_pixel else None

print(f"File: {path}")
print(f"Format: {format}")
print(f"Resolution: {width} × {height} px")
print(f"Mode: {mode}")
print(f"DPI: {dpi}")
print(f"File size: {file_size / 1024 / 1024:.2f} MB")

if raw_size:
    print(f"Raw image memory: {raw_size / 1024 / 1024:.2f} MB")