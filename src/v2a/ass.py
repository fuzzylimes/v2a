"""ASS subtitle style definition and file assembly."""

from pathlib import Path

import pysubs2

from .spu import read_spu_at

# A subtitle running longer than this almost certainly has a bad or missing
# STP_DSP end-time. Cap it to prevent text freezing across scene breaks.
MAX_SUBTITLE_MS = 8_000
MIN_SUBTITLE_MS = 500
LAST_FALLBACK_MS = 3_000   # used only for the very last entry when STP_DSP is absent

# DVD frame dimensions used for zone-based alignment detection.
# PAL is 720x576; NTSC is 720x480. The PAL thresholds work acceptably for both.
DVD_WIDTH = 720
DVD_HEIGHT = 576

# If a SET_DAREA bounding box covers this much of the frame in both dimensions,
# the DVD author set it to the full canvas and it carries no position information.
# Pixel analysis of the rendered PNG is used instead.
_FULLFRAME_MIN_W = 650   # ~90% of 720
_FULLFRAME_MIN_H = 380   # ~79% of 480  (generous to catch NTSC full-frame subs)


def make_style() -> pysubs2.SSAStyle:
    """
    Return the default ASS style for converted subtitles.

    Edit this function to change the look of all output files.
    pysubs2.Color(r, g, b, a) — alpha: 0 = fully opaque, 255 = fully transparent.
    """
    s = pysubs2.SSAStyle()
    s.fontname = "Arial"
    s.fontsize = 36
    s.primarycolor = pysubs2.Color(255, 255, 255,   0)   # white text
    s.secondarycolor = pysubs2.Color(255, 255, 255,   0)
    s.outlinecolor = pysubs2.Color(0,   0,   0,   0)   # black border
    s.backcolor = pysubs2.Color(0,   0,   0, 160)   # soft shadow
    s.bold = False
    s.italic = False
    s.outline = 2.5
    s.shadow = 1.5
    s.alignment = pysubs2.Alignment.BOTTOM_CENTER   # overridden per-event for signs
    s.marginl = 40
    s.marginr = 40
    s.marginv = 30
    return s


def _bbox_is_fullframe(x1, y1, x2, y2) -> bool:
    """
    Return True when SET_DAREA covers essentially the full DVD frame.

    Some DVD authoring tools set the display area to the entire frame and
    position text using the pixel data, making the bbox useless for alignment.
    """
    if None in (x1, y1, x2, y2):
        return True
    return (x2 - x1 + 1) >= _FULLFRAME_MIN_W and (y2 - y1 + 1) >= _FULLFRAME_MIN_H


def _zone_alignment(cx: float, cy: float, width: float, height: float) -> int:
    """
    Map a centroid in a frame of (width, height) to an ASS \\an numpad value.

    The frame is divided into a 3x3 zone grid (proportional, so it works for any
    source resolution):
      Row thresholds:    top < h/4,  mid < 3h/5,  bot >= 3h/5.
      Column thresholds: left < w/3, center < 2w/3, right >= 2w/3.

    Numpad mapping:  7 8 9 / 4 5 6 / 1 2 3.
    """
    if cy < height / 4:
        vert = 'top'
    elif cy < height * 3 / 5:
        vert = 'mid'
    else:
        vert = 'bot'

    if cx < width / 3:
        horiz = 'left'
    elif cx < 2 * width / 3:
        horiz = 'center'
    else:
        horiz = 'right'

    return {
        ('top', 'left'): 7, ('top', 'center'): 8, ('top', 'right'): 9,
        ('mid', 'left'): 4, ('mid', 'center'): 5, ('mid', 'right'): 6,
        ('bot', 'left'): 1, ('bot', 'center'): 2, ('bot', 'right'): 3,
    }[(vert, horiz)]


def alignment_from_area(x1, y1, x2, y2) -> int:
    """
    Map a subtitle bounding box to an ASS \\an numpad value (1–9).

    The DVD frame is divided into a 3x3 zone grid:
      Row thresholds (576px PAL):  top < 144,  mid < 346,  bot >= 346.
      Column thresholds (720px):   left < 240, center < 480, right >= 480.

    Normal dialogue sits in the bottom zone and maps to \\an2 (bottom-center).
    Signs and forced subs in the top or middle zones get an explicit anchor tag.
    Returns 2 (bottom-center) for None coordinates.
    """
    if None in (x1, y1, x2, y2):
        return 2

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return _zone_alignment(cx, cy, DVD_WIDTH, DVD_HEIGHT)


def alignment_from_image(img_path: Path) -> int:
    """
    Derive subtitle alignment by analysing where visible pixels sit in the frame PNG.

    Used when SET_DAREA covers the full frame and carries no position information.
    Opens the RGBA PNG, thresholds the alpha channel to suppress noise, then finds
    the bounding box of visible content and maps its centre to an \\an numpad value.
    Returns 2 (bottom-center) on any error or if the image is fully transparent.
    """
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGBA")
        alpha = img.split()[3]
        # Threshold: ignore pixels whose alpha is likely compression noise.
        mask = alpha.point(lambda a: 255 if a > 32 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return 2

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w, h = img.size
        return _zone_alignment(cx, cy, w, h)
    except Exception:
        return 2


def build_ass(
    entries:     list[tuple[int, int]],
    texts:       list[str],
    sub_path:    Path,
    output_path: Path,
    frames:      list[Path] | None = None,
    verbose:     bool = False,
) -> int:
    """
    Assemble an SSAFile from timing entries and OCR texts, then write it to disk.

    When `frames` is supplied and a subtitle's SET_DAREA bbox covers the full frame
    (a common DVD authoring shortcut), alignment is derived from the rendered PNG's
    alpha channel instead of the bbox coordinates.

    End-time priority (per CLAUDE.md spec):
      1. SPU's own STP_DSP offset (most accurate)
      2. Next subtitle's start time minus 100 ms gap
      3. LAST_FALLBACK_MS fixed duration (last entry only)
      4. Hard cap: MAX_SUBTITLE_MS
      5. Hard floor: MIN_SUBTITLE_MS

    Returns the number of events written.
    """
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "720"
    subs.info["PlayResY"] = "576"
    subs.styles["Default"] = make_style()

    total = len(entries)

    for i, ((start_ms, filepos), text) in enumerate(zip(entries, texts)):
        if not text:
            continue

        spu = read_spu_at(sub_path, filepos)

        if spu["end_ms"] is not None:
            end_ms = start_ms + spu["end_ms"]
        elif i + 1 < total:
            end_ms = entries[i + 1][0] - 100
        else:
            end_ms = start_ms + LAST_FALLBACK_MS

        end_ms = min(end_ms, start_ms + MAX_SUBTITLE_MS)
        end_ms = max(end_ms, start_ms + MIN_SUBTITLE_MS)

        use_image = (
            frames is not None
            and i < len(frames)
            and _bbox_is_fullframe(spu["x1"], spu["y1"], spu["x2"], spu["y2"])
        )
        if use_image:
            an = alignment_from_image(frames[i])
            method = "img"
        else:
            an = alignment_from_area(spu["x1"], spu["y1"], spu["x2"], spu["y2"])
            method = "bbox"

        if verbose:
            print(f"    [{i:4d}] bbox=({spu['x1']},{spu['y1']})-({spu['x2']},{spu['y2']})  "
                  f"an={an} [{method}]  {text[:40]!r}")

        subs.events.append(pysubs2.SSAEvent(
            start=start_ms, end=end_ms, text=_event_text(text, an)))

    subs.save(str(output_path))
    return len(subs.events)


def _event_text(text: str, an: int) -> str:
    """Convert OCR text to an ASS event body, prepending an \\an tag for non-default zones."""
    body = text.replace("\n", "\\N")
    if an != 2:                       # \an2 is the style default — no override needed
        body = f"{{\\an{an}}}{body}"
    return body


def build_ass_pgs(
    records:     list[dict],
    texts:       list[str],
    output_path: Path,
    verbose:     bool = False,
) -> int:
    """
    Assemble an SSAFile from PGS (Blu-ray) subtitle records and OCR texts.

    Unlike VobSub, PGS carries reliable start *and* end times (from the
    presentation and clear composition segments), so timing is taken directly
    from each record — only the MIN_SUBTITLE_MS floor is applied, and the 8s cap
    / STP_DSP fallback chain used for DVD sources are not needed. A trailing
    subtitle the stream never clears (end_ms is None) falls back to LAST_FALLBACK_MS.

    Positioning is exact: PGS records carry each subtitle's real on-screen
    bounding box, so every event is anchored at its box center with
    ``{\\an5\\pos(cx,cy)}`` against the native canvas (declared as PlayRes). This
    reproduces the disc's placement for dialogue, raised dialogue, and signs
    alike — none of the 9-zone bucketing the DVD path needs. A record with no
    box (a blank/failed render) falls back to the style's bottom-center default.

    Styling matches the DVD look: same font family, colors, and outline/shadow,
    with the size-related metrics scaled proportionally to the taller canvas so
    subtitles appear the same on-screen size as DVD output.

    Returns the number of events written.
    """
    canvas_w = records[0]["canvas_w"] if records else 1920
    canvas_h = records[0]["canvas_h"] if records else 1080

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(canvas_w)
    subs.info["PlayResY"] = str(canvas_h)
    subs.styles["Default"] = _scaled_style(canvas_h)

    for i, (rec, text) in enumerate(zip(records, texts)):
        if not text:
            continue

        start_ms = rec["start_ms"]
        end_ms = rec["end_ms"]
        if end_ms is None:
            end_ms = start_ms + LAST_FALLBACK_MS
        end_ms = max(end_ms, start_ms + MIN_SUBTITLE_MS)

        body = text.replace("\n", "\\N")
        x1, y1, x2, y2 = rec["x1"], rec["y1"], rec["x2"], rec["y2"]
        if None not in (x1, y1, x2, y2):
            cx, cy = round((x1 + x2) / 2), round((y1 + y2) / 2)
            body = f"{{\\an5\\pos({cx},{cy})}}{body}"
            if verbose:
                print(f"    [{i:4d}] pos=({cx},{cy}) on {canvas_w}x{canvas_h}  {text[:40]!r}")
        elif verbose:
            print(f"    [{i:4d}] no box — bottom-center  {text[:40]!r}")

        subs.events.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=body))

    subs.save(str(output_path))
    return len(subs.events)


def _scaled_style(canvas_h: int) -> pysubs2.SSAStyle:
    """
    Return the Default style scaled for a canvas of height `canvas_h`.

    make_style() is tuned for the 576px-tall DVD frame. Size-related metrics
    (fontsize, outline, shadow) scale linearly with canvas height so 1080p
    Blu-ray output looks the same on screen; font, colors, and weight are kept.
    """
    style = make_style()
    scale = canvas_h / DVD_HEIGHT
    style.fontsize = round(style.fontsize * scale)
    style.outline = round(style.outline * scale, 1)
    style.shadow = round(style.shadow * scale, 1)
    return style
