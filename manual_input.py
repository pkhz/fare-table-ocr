"""
Manual input UI for fare table cell values.
Builds ground-truth mapping from visual inspection.

Usage:
    python manual_input.py              # 5x5 corner (25 cells)
    python manual_input.py --rows 0 1 2 # specific rows
    python manual_input.py --all        # all cells (not recommended)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from segment_debug import crop_to_text, segment_by_projection, OUT_DIR
from template_match import normalize_cell, normalize_blue_cell, is_blue_cell, CELLS_DIR

# Scale factor for displaying segments (original ~9px tall)
DISPLAY_SCALE = 50 // 9  # ~5x zoom
SEGMENT_DISPLAY_SIZE = (50, 50)

SAVE_FILE = Path("manual_values.json")


class ManualInputApp:
    def __init__(self, root, cell_list):
        self.root = root
        self.root.title("Fare Table Manual Input")

        self.cell_list = cell_list  # [(r, c), ...]
        self.current_idx = 0
        self.values = {}  # {(r,c): "value"} or {(r,c): None} for skipped

        # Load existing values if file exists
        if SAVE_FILE.exists():
            with open(SAVE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            self.values = {tuple(k.split("_")): v for k, v in data.items()}
            self.values = {(int(r), int(c)): v for (r, c), v in self.values.items()}
            print(f"Loaded {len(self.values)} existing values from {SAVE_FILE}")

        # UI elements
        self.setup_ui()
        self.load_cell()

    def setup_ui(self):
        # Header
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X)

        ttk.Label(header_frame, text="Cell:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.cell_label = ttk.Label(header_frame, text="", font=("Arial", 12))
        self.cell_label.pack(side=tk.LEFT, padx=(10, 20))

        ttk.Label(header_frame, text="Value:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.value_entry = ttk.Entry(header_frame, font=("Arial", 14), width=10)
        self.value_entry.pack(side=tk.LEFT, padx=(10, 0))
        self.value_entry.bind("<Return>", lambda e: self.save_and_next())

        # Progress
        self.progress_label = ttk.Label(header_frame, text="", font=("Arial", 10))
        self.progress_label.pack(side=tk.RIGHT)

        # Images frame
        img_frame = ttk.Frame(self.root, padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True)

        # Full cell image
        ttk.Label(img_frame, text="Full Cell:", font=("Arial", 10)).pack(anchor=tk.W)
        self.cell_canvas = tk.Canvas(img_frame, bg="white", width=200, height=100)
        self.cell_canvas.pack(pady=(0, 10))

        # Segments
        ttk.Label(img_frame, text="Segments (type value above):", font=("Arial", 10)).pack(anchor=tk.W)
        self.seg_frame = ttk.Frame(img_frame)
        self.seg_frame.pack(pady=(5, 10))

        # Navigation
        nav_frame = ttk.Frame(self.root, padding=10)
        nav_frame.pack(fill=tk.X)

        ttk.Button(nav_frame, text="← Prev (P)", command=self.prev_cell).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Next (Enter)", command=self.save_and_next).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Skip (S)", command=self.skip_cell).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Quit", command=self.save_and_quit).pack(side=tk.RIGHT, padx=5)

        # Keyboard shortcuts
        self.root.bind("<p>", lambda e: self.prev_cell())
        self.root.bind("<s>", lambda e: self.skip_cell())
        self.root.bind("<q>", lambda e: self.save_and_quit())

        # Focus entry for quick input
        self.value_entry.focus_set()

    def load_cell(self):
        """Load and display current cell."""
        if self.current_idx >= len(self.cell_list):
            self.save_all()  # value for last cell already saved by save_and_next
            messagebox.showinfo("Done", f"All {len(self.cell_list)} cells processed!")
            self.root.quit()
            return

        r, c = self.cell_list[self.current_idx]

        # Update labels
        self.cell_label.config(text=f"r{r:03d}c{c:03d}")
        self.progress_label.config(text=f"{self.current_idx + 1} / {len(self.cell_list)}")

        # Clear previous value
        existing = self.values.get((r, c), "")
        self.value_entry.delete(0, tk.END)
        if existing:
            self.value_entry.insert(0, existing)

        # Load and display cell image
        path = CELLS_DIR / f"cell_r{r:03d}_c{c:03d}.png"
        img = cv2.imread(str(path))
        if img is None:
            messagebox.showerror("Error", f"Cannot load {path}")
            return

        # Display full cell (scaled)
        cell_display = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        self.display_image(self.cell_canvas, cell_display)

        # Load segments (from segment_debug output)
        seg_dir = OUT_DIR / f"cell_r{r:03d}_c{c:03d}"
        if not seg_dir.exists():
            # Generate segments on-the-fly
            if is_blue_cell(path):
                binary = normalize_blue_cell(img)
            else:
                binary = normalize_cell(img)
            cropped = crop_to_text(binary)
            if cropped is not None:
                chars = segment_by_projection(cropped)
                seg_dir.mkdir(parents=True, exist_ok=True)
                for i, char_img in enumerate(chars):
                    cv2.imwrite(str(seg_dir / f"char_{i:02d}.png"), char_img)
            else:
                chars = []
        else:
            # Load existing segments
            char_files = sorted(seg_dir.glob("char_*.png"))
            chars = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in char_files]

        # Display segments
        for widget in self.seg_frame.winfo_children():
            widget.destroy()

        for i, char_img in enumerate(chars):
            # Scale to display size
            scaled = cv2.resize(char_img, SEGMENT_DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)

            # Create canvas
            canvas = tk.Canvas(self.seg_frame, width=SEGMENT_DISPLAY_SIZE[0], height=SEGMENT_DISPLAY_SIZE[1],
                             bg="white", highlightthickness=1, highlightbackground="gray")
            canvas.grid(row=0, column=i, padx=5)

            # Display image
            self.display_image(canvas, scaled, is_binary=True)

            # Label
            ttk.Label(self.seg_frame, text=f"Seg {i}").grid(row=1, column=i, padx=5)

    def display_image(self, canvas, img, is_binary=False):
        """Display OpenCV image on tkinter canvas."""
        if is_binary:
            # Binary image (grayscale)
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img
        else:
            # BGR image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PhotoImage
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_img)

        # Keep reference to prevent garbage collection
        canvas.image = photo
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def save_value(self):
        """Save current value."""
        if self.current_idx >= len(self.cell_list):
            return  # past the end — nothing to save
        r, c = self.cell_list[self.current_idx]
        value = self.value_entry.get().strip()
        if value:
            self.values[(r, c)] = value

    def save_and_next(self):
        """Save value and go to next cell."""
        self.save_value()
        self.current_idx += 1
        self.load_cell()
        self.value_entry.focus_set()

    def prev_cell(self):
        """Go to previous cell."""
        if self.current_idx > 0:
            self.save_value()
            self.current_idx -= 1
            self.load_cell()
            self.value_entry.focus_set()

    def skip_cell(self):
        """Skip current cell."""
        self.current_idx += 1
        self.load_cell()
        self.value_entry.focus_set()

    def save_all(self):
        """Write all values to JSON."""
        # Convert keys to strings for JSON
        data = {f"{r}_{c}": v for (r, c), v in self.values.items()}
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.values)} values to {SAVE_FILE}")

    def save_and_quit(self):
        """Save all values to JSON and quit."""
        self.save_value()
        self.save_all()
        self.root.quit()


def main():
    # Determine which cells to process
    if "--all" in sys.argv:
        cells = sorted(CELLS_DIR.glob("cell_r*_c*.png"))
        cell_list = []
        for p in cells:
            parts = p.stem.split("_")
            r = int(parts[1][1:])
            c = int(parts[2][1:])
            cell_list.append((r, c))
    elif "--rows" in sys.argv:
        idx = sys.argv.index("--rows")
        rows = [int(x) for x in sys.argv[idx+1:]]
        cell_list = [(r, c) for r in rows for c in range(160)]
    else:
        # Default: 5x5 corner
        cell_list = [(r, c) for r in range(5) for c in range(5)]

    print(f"Processing {len(cell_list)} cells")
    print(f"Save file: {SAVE_FILE}")
    print(f"Display: Enter=save&next, P=prev, S=skip, Q=quit")

    # Launch UI
    root = tk.Tk()
    root.geometry("600x500")
    app = ManualInputApp(root, cell_list)
    root.mainloop()


if __name__ == "__main__":
    main()
