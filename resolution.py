from PIL import Image

path = "image/faretable.png"

with Image.open(path) as img:
    print("Size:", img.size)
    print("Mode:", img.mode)
    print("Format:", img.format)
    print("Palette:", img.getpalette()[:30] if img.mode == "P" else None)

    # Convert to RGB for OCR processing later
    rgb = img.convert("RGB")

    print("RGB size:", rgb.size)