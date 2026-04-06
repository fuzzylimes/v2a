"""OCR preprocessing and frame processing."""

import re
from pathlib import Path

from PIL import Image, ImageEnhance
import pytesseract


def preprocess(img: Image.Image) -> Image.Image:
    """
    Prepare a VobSub bitmap for Tesseract.

    Steps:
      1. Flatten transparent pixels to black (VobSub PNGs have an alpha channel).
      2. Convert to grayscale.
      3. Upscale 3× with LANCZOS — DVD subtitle strips (~720×60 px) are too small
         for reliable Tesseract recognition at native resolution.
      4. Boost contrast to sharpen the white text / black outline boundary.
    """
    bg   = Image.new("RGBA", img.size, (0, 0, 0, 255))
    flat = Image.alpha_composite(bg, img.convert("RGBA")).convert("L")
    w, h = flat.size
    flat = flat.resize((w * 3, h * 3), Image.LANCZOS)
    return ImageEnhance.Contrast(flat).enhance(1.8)


def ocr_frames(frame_paths: list[Path]) -> list[str]:
    """
    OCR each frame PNG and return a list of text strings (one per frame).

    Empty frames produce an empty string. Double-spaces and triple-newlines are
    collapsed to keep the output tidy.
    """
    results = []
    total   = len(frame_paths)
    for i, fp in enumerate(frame_paths, 1):
        print(f"\r    OCR: {i}/{total}  ({i * 100 // total}%)", end="", flush=True)
        img = preprocess(Image.open(fp))
        raw = pytesseract.image_to_string(img, config="--psm 6 --oem 3").strip()
        raw = re.sub(r"[ \t]{2,}", " ",  raw)
        raw = re.sub(r"\n{3,}",   "\n", raw)
        results.append(raw)
    print()
    return results
