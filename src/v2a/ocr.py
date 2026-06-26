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
      3. Sharpen edges before upscaling to give LANCZOS better boundaries to work with.
      4. Upscale 3x with LANCZOS — DVD subtitle strips (~720x60 px) are too small
         for reliable Tesseract recognition at native resolution.
      5. Boost contrast then binarize to pure black/white — Tesseract is trained on
         binary images and performs best without intermediate gray values.
    """
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    flat = Image.alpha_composite(bg, img.convert("RGBA")).convert("L")
    w, h = flat.size
    flat = ImageEnhance.Sharpness(flat).enhance(2.0)
    flat = flat.resize((w * 3, h * 3), Image.LANCZOS)
    enhanced = ImageEnhance.Contrast(flat).enhance(1.8)
    return enhanced.point(lambda x: 255 if x > 128 else 0)


# Tesseract config. `tessedit_char_blacklist=|` stops the engine from ever
# emitting a pipe — the tall vertical glyph shared by `I` and `l` is the most
# common VobSub misread — forcing it to choose a real letter by glyph shape.
_TESS_CONFIG = "--psm 6 --oem 3 -c tessedit_char_blacklist=|¦"

# Runs of one or more pipe characters that survived the blacklist (older
# Tesseract LSTM builds don't always honor char_blacklist).
_PIPE_RUN = re.compile(r"[|¦]+")


def _restore_il(text: str) -> str:
    """
    Replace any leftover pipe characters with `I` or `l` from context.

    The glyph is genuinely ambiguous, so we use the dominant English cases:
      * A standalone pipe (non-letters on both sides) is the pronoun ``I``.
      * A pipe before an apostrophe (``|'m``, ``|'ll``) is ``I``.
      * A pipe touching a lowercase letter is ``l`` (``wi|| -> will``).
      * Anything else (uppercase-only context) defaults to ``I``.
    """
    def repl(m: re.Match) -> str:
        s = m.string
        prev = s[m.start() - 1] if m.start() > 0 else ""
        nxt = s[m.end()] if m.end() < len(s) else ""
        n = len(m.group())
        if nxt == "'":
            return "I" * n
        prev_lower = prev.isalpha() and prev.islower()
        next_lower = nxt.isalpha() and nxt.islower()
        if prev_lower or next_lower:
            return "l" * n
        return "I" * n

    return _PIPE_RUN.sub(repl, text)


def ocr_frames(frame_paths: list[Path]) -> list[str]:
    """
    OCR each frame PNG and return a list of text strings (one per frame).

    Empty frames produce an empty string. Double-spaces and triple-newlines are
    collapsed to keep the output tidy.
    """
    results = []
    total = len(frame_paths)
    for i, fp in enumerate(frame_paths, 1):
        print(f"\r    OCR: {i}/{total}  ({i * 100 // total}%)",
              end="", flush=True)
        img = preprocess(Image.open(fp))
        raw = pytesseract.image_to_string(img, config=_TESS_CONFIG).strip()
        raw = _restore_il(raw)
        raw = re.sub(r"[ \t]{2,}", " ",  raw)
        raw = re.sub(r"\n{3,}",   "\n", raw)
        results.append(raw)
    print()
    return results
