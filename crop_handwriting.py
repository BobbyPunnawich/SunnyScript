"""
crop_handwriting.py
-------------------
Crop handwriting rows from Template A / Template B images
to prepare a TrOCR dataset.

Folder layout expected:
    inputs/          ← put source images here
    data/clean/      ← output for "clean" filenames
    data/messy/      ← output for "messy" filenames

Naming convention:
    - Filename contains 'A' (case-insensitive) → Template A  (sentences)
    - Filename contains 'B' (case-insensitive) → Template B  (formulas)
    - Filename contains 'messy'                → saved to data/messy/
    - Otherwise                                → saved to data/clean/
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

# ─────────────────────────── Configuration ────────────────────────────────
#
# Adjust these values if cropped rows are misaligned:
#   start_x / start_y  → shift the entire grid left/right or up/down
#   row_height          → increase if rows overlap; decrease if gaps appear
#   column_width        → extend or shrink horizontal crop width
#   padding             → pixels removed from each edge to hide grid lines

TEMPLATE_A = {
    "start_x":      150,   # left edge of the writing area (pixels)
    "start_y":      196,   # top edge of the first row   (pixels)
    "row_height":    90,   # height of one row            (pixels)
    "num_rows":      21,   # total rows to crop
    "column_width": 600,   # width of the writing area    (pixels)
    "padding":        5,   # inner margin to hide grid lines
}

TEMPLATE_B = {
    "start_x":      173,
    "start_y":      202,
    "row_height":    58,
    "num_rows":      21,
    "column_width": 600,
    "padding":        4,
}

INPUT_DIR    = Path("inputs")
OUTPUT_CLEAN = Path("data/clean")
OUTPUT_MESSY = Path("data/messy")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ──────────────────────────── Core Functions ──────────────────────────────

def crop_rows(image: np.ndarray, cfg: dict) -> list[tuple[int, np.ndarray]]:
    """
    Slice `image` into rows according to `cfg`.
    Returns [(row_number_1based, crop_ndarray), ...].
    """
    h, w = image.shape[:2]
    pad   = cfg["padding"]
    x1    = cfg["start_x"] + pad
    x2    = cfg["start_x"] + cfg["column_width"] - pad

    # Clamp X to image width
    x1, x2 = max(0, x1), min(w, x2)

    crops = []
    for i in range(cfg["num_rows"]):
        y1 = cfg["start_y"] + i * cfg["row_height"] + pad
        y2 = y1 + cfg["row_height"] - 2 * pad

        # Clamp Y to image height
        y1, y2 = max(0, y1), min(h, y2)

        if y2 <= y1 or x2 <= x1:          # sanity-check: skip empty slices
            continue

        crops.append((i + 1, image[y1:y2, x1:x2].copy()))

    return crops


def crop_template_a(image: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Template A – ประโยค (sentences)."""
    return crop_rows(image, TEMPLATE_A)


def crop_template_b(image: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Template B – สูตร (formulas)."""
    return crop_rows(image, TEMPLATE_B)


# ─────────────────────────── Debug / Preview ──────────────────────────────

def draw_grid(image: np.ndarray, cfg: dict, color=(0, 0, 255), thickness=2) -> np.ndarray:
    """
    Draw the crop grid on a copy of `image` so you can visually verify
    alignment before running the full batch.  Call with --debug flag.
    """
    vis   = image.copy()
    pad   = cfg["padding"]
    x1    = cfg["start_x"] + pad
    x2    = cfg["start_x"] + cfg["column_width"] - pad

    for i in range(cfg["num_rows"]):
        y1 = cfg["start_y"] + i * cfg["row_height"] + pad
        y2 = y1 + cfg["row_height"] - 2 * pad
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, str(i + 1), (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


# ─────────────────────────── File Processing ──────────────────────────────

def get_output_dir(stem: str) -> Path:
    """Return data/messy/ if 'messy' is in the filename, else data/clean/."""
    return OUTPUT_MESSY if "messy" in stem.lower() else OUTPUT_CLEAN


def process_file(filepath: Path, debug: bool = False) -> None:
    stem = filepath.stem
    ext  = filepath.suffix

    image = cv2.imread(str(filepath))
    if image is None:
        print(f"  [WARN] Cannot read image: {filepath.name} — skipped")
        return

    name_upper = stem.upper()

    # ── Choose template ────────────────────────────────────────────────────
    # Priority: 'A' checked before 'B' so 'AB' files go to A.
    # Flip the order below if you need a different priority.
    if "A" in name_upper:
        crops          = crop_template_a(image)
        template_label = "A"
        cfg            = TEMPLATE_A
    elif "B" in name_upper:
        crops          = crop_template_b(image)
        template_label = "B"
        cfg            = TEMPLATE_B
    else:
        print(f"  [WARN] No 'A' or 'B' in filename: {filepath.name} — skipped")
        return

    # ── Debug preview ──────────────────────────────────────────────────────
    if debug:
        vis = draw_grid(image, cfg)
        win = f"DEBUG – {filepath.name}  (press any key to continue)"
        cv2.imshow(win, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── Save crops ─────────────────────────────────────────────────────────
    out_dir = get_output_dir(stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for row_idx, crop in crops:
        out_path = out_dir / f"{stem}_row{row_idx:02d}{ext}"
        cv2.imwrite(str(out_path), crop)
        saved += 1

    print(f"  [OK] {filepath.name}  →  Template {template_label},"
          f"  {saved} rows  →  {out_dir}/")


# ──────────────────────────────── Main ────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop handwriting rows from Template A / B images.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show grid overlay on each image before saving (press any key).")
    parser.add_argument(
        "--input", default=str(INPUT_DIR),
        help=f"Input folder (default: {INPUT_DIR})")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"[ERROR] Input directory '{input_dir}' not found.")
        print("        Create it and place your template images inside.")
        return

    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )

    if not files:
        print(f"[INFO] No image files found in '{input_dir}'.")
        return

    print(f"Found {len(files)} image(s) in '{input_dir}'\n")

    for filepath in files:
        print(f"Processing: {filepath.name}")
        process_file(filepath, debug=args.debug)

    print("\nAll done.")


if __name__ == "__main__":
    main()
