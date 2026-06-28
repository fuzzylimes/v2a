"""OCR preprocessing and frame processing."""

import re
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
import pytesseract

# DVD-only preprocessing knobs. DVD subtitle strips are tiny, soft, and cropped
# tight to the glyphs, which is exactly what makes the tall I/l/1/[ shapes blur
# together. PGS (Blu-ray) strips are already sharp and high-resolution, so they
# skip this path (preprocess(dvd=False)).
_DVD_UPSCALE = 4        # vs 3x for PGS — more detail for the LSTM to read serifs
_QUIET_ZONE_PX = 20     # blank margin so no glyph touches the image edge


def _otsu_threshold(img: Image.Image) -> int:
    """
    Pick a binarization threshold from a grayscale image's histogram (Otsu's method).

    A fixed cutoff erodes thin strokes whenever a strip is dimmer or brighter than
    expected; Otsu finds the valley between the text and background peaks per-strip,
    which keeps the vertical stem of an ``I``/``l``/``1`` intact. Pure-Python over a
    256-bin histogram — negligible cost.
    """
    hist = img.histogram()[:256]
    total = sum(hist)
    if total == 0:
        return 128
    sum_all = sum(i * h for i, h in enumerate(hist))
    w_bg = 0
    sum_bg = 0.0
    max_var = -1.0
    threshold = 128
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        m_bg = sum_bg / w_bg
        m_fg = (sum_all - sum_bg) / w_fg
        between = w_bg * w_fg * (m_bg - m_fg) ** 2
        if between > max_var:
            max_var = between
            threshold = t
    return threshold


def preprocess(img: Image.Image, dvd: bool = False) -> Image.Image:
    """
    Prepare a subtitle bitmap for Tesseract.

    Steps shared by both sources:
      1. Flatten transparent pixels to black (the PNGs have an alpha channel).
      2. Convert to grayscale.
      3. Sharpen edges before upscaling to give LANCZOS better boundaries to work with.
      4. Upscale with LANCZOS — subtitle strips are too small for reliable
         recognition at native resolution.
      5. Boost contrast.

    With ``dvd=True`` (VobSub / DVD), three extra steps fight the soft, tightly
    cropped, low-resolution glyphs that make tall shapes (``I l 1 [ |``) blur
    together at the source:
      * a larger 4x upscale (vs 3x) for more serif detail;
      * an adaptive Otsu threshold instead of a fixed 128 cutoff, so dim or bright
        strips still binarize without eroding thin stems;
      * a blank quiet-zone border so no glyph sits flush against the image edge
        (Tesseract reads edge-touching characters poorly);
      * inversion to black text on a white background — Tesseract's models are
        trained on dark-on-light, so subtitles (bright text on black) read more
        reliably flipped.

    PGS (Blu-ray) is already sharp, so ``dvd=False`` keeps the original lighter
    path (white-on-black, no border).
    """
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    flat = Image.alpha_composite(bg, img.convert("RGBA")).convert("L")
    w, h = flat.size
    flat = ImageEnhance.Sharpness(flat).enhance(2.0)
    scale = _DVD_UPSCALE if dvd else 3
    flat = flat.resize((w * scale, h * scale), Image.LANCZOS)
    enhanced = ImageEnhance.Contrast(flat).enhance(1.8)

    if not dvd:
        return enhanced.point(lambda x: 255 if x > 128 else 0)

    thresh = _otsu_threshold(enhanced)
    # Invert: bright subtitle pixels → black text on a white background.
    binary = enhanced.point(lambda x: 0 if x > thresh else 255)
    # Border is filled with the (now white) background colour.
    return ImageOps.expand(binary, border=_QUIET_ZONE_PX, fill=255)


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


# A balanced bracket pair — a real sound cue such as "[ Honking ]" or "[Sighs]".
# These are masked out before bracket repair so their brackets are preserved.
_BRACKET_PAIR = re.compile(r"\[[^\[\]]*\]")
# Runs of stray square brackets left over after the pairs are masked — these are
# the ones Tesseract emitted in place of a tall I/l (``]'m``, ``]t``, ``wi]]``).
_BRACKET_RUN = re.compile(r"[\[\]]+")
_MASK = re.compile("\x00(\\d+)\x00")


def _restore_brackets(text: str) -> str:
    """
    Repair ``[`` / ``]`` glyphs that should be ``I`` or ``l``.

    Tesseract sometimes reads the tall subtitle ``I``/``l`` as a square bracket
    (``]'m`` for ``I'm``). Brackets are legitimate for sound cues (``[ Honking ]``),
    so a *balanced* ``[...]`` pair is masked out and left untouched; only the
    stray, unmatched brackets that remain are repaired, using the same casing
    rule as :func:`_restore_one`:

      * Following a lowercase letter → ``l`` (``wi]]`` → ``will``).
      * Otherwise (word start, ``]'m``, standalone) → ``I``.
    """
    protected: list[str] = []

    def stash(m: re.Match) -> str:
        protected.append(m.group())
        return f"\x00{len(protected) - 1}\x00"

    masked = _BRACKET_PAIR.sub(stash, text)

    def repl(m: re.Match) -> str:
        s = m.string
        prev = s[m.start() - 1] if m.start() > 0 else ""
        nxt = s[m.end()] if m.end() < len(s) else ""
        n = len(m.group())
        if nxt == "'":
            return "I" * n
        if prev.isalpha() and prev.islower():
            return "l" * n
        return "I" * n

    repaired = _BRACKET_RUN.sub(repl, masked)
    return _MASK.sub(lambda m: protected[int(m.group(1))], repaired)


def ocr_frames(frame_paths: list[Path], dvd: bool = False) -> list[str]:
    """
    OCR each frame PNG and return a list of text strings (one per frame).

    ``dvd=True`` enables the heavier DVD/VobSub preprocessing (see preprocess);
    PGS callers leave it False. Empty frames produce an empty string.
    Double-spaces and triple-newlines are collapsed to keep the output tidy.
    """
    results = []
    total = len(frame_paths)
    for i, fp in enumerate(frame_paths, 1):
        print(f"\r    OCR: {i}/{total}  ({i * 100 // total}%)",
              end="", flush=True)
        img = preprocess(Image.open(fp), dvd=dvd)
        raw = pytesseract.image_to_string(img, config=_TESS_CONFIG).strip()
        raw = _restore_il(raw)
        raw = _restore_one(raw)
        raw = _restore_brackets(raw)
        raw = re.sub(r"[ \t]{2,}", " ",  raw)
        raw = re.sub(r"\n{3,}",   "\n", raw)
        results.append(raw)
    print()
    return results
