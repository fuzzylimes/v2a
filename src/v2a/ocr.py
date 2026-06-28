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


# Runs of the digit '1' Tesseract emitted where a tall I/l was meant. Unlike the
# pipe, '1' is a real character we must keep, so this is repaired contextually
# rather than blacklisted.
_ONE_RUN = re.compile(r"1+")
# An ordinal suffix immediately after a lone '1' (1st / 1th) — a real number, so
# the '1' is left alone.
_ORDINAL = re.compile(r"(?:st|nd|rd|th)", re.IGNORECASE)


def _restore_one(text: str) -> str:
    """
    Repair ``1`` glyphs that should be ``I`` or ``l``.

    Tesseract frequently reads the tall subtitle ``I``/``l`` as a digit ``1``
    (``1t`` for ``It``, ``wi11`` for ``will``). The digit is a legitimate
    character, so — unlike the pipe — a run is only repaired when it is touching
    a letter (or an ``'`` contraction) and is not part of a larger number or an
    ordinal:

      * Following a lowercase letter   → ``l`` (``wi11`` → ``will``).
      * Otherwise (word start, all-caps, or ``1'm``) → ``I`` (``1t`` → ``It``).

    Standalone digits (``Take 1``), longer numbers (``1080``), and ordinals
    (``1st``) are left untouched.
    """
    def repl(m: re.Match) -> str:
        s = m.string
        prev = s[m.start() - 1] if m.start() > 0 else ""
        nxt = s[m.end()] if m.end() < len(s) else ""
        n = len(m.group())
        # Part of a longer number (e.g. 1080, 2016) — leave alone.
        if prev.isdigit() or nxt.isdigit():
            return m.group()
        # Ordinal such as 1st / 21st — leave alone.
        if n == 1 and _ORDINAL.match(s, m.end()):
            return m.group()
        # Only repair when a letter (or contraction apostrophe) is adjacent; a
        # truly standalone '1' is genuinely a number.
        if not (prev.isalpha() or nxt.isalpha() or nxt == "'"):
            return m.group()
        if prev.isalpha() and prev.islower():
            return "l" * n
        return "I" * n

    return _ONE_RUN.sub(repl, text)


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
        raw = _restore_one(raw)
        raw = re.sub(r"[ \t]{2,}", " ",  raw)
        raw = re.sub(r"\n{3,}",   "\n", raw)
        results.append(raw)
    print()
    return results
