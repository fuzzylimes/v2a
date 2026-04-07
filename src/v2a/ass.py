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


def make_style() -> pysubs2.SSAStyle:
    """
    Return the default ASS style for converted subtitles.

    Edit this function to change the look of all output files.
    pysubs2.Color(r, g, b, a) — alpha: 0 = fully opaque, 255 = fully transparent.
    """
    s = pysubs2.SSAStyle()
    s.fontname = "Arial"
    s.fontsize = 52
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


def alignment_from_area(x1, y1, x2, y2) -> int:
    """
    Map a subtitle bounding box to an ASS \\an numpad value (1–9).

    The DVD frame is divided into a 3x3 zone grid. Normal dialogue sits in the
    bottom zone and maps to \\an2 (bottom-center). Signs and forced subs in the
    top or middle zones get a more appropriate anchor point.
    """
    if None in (x1, y1, x2, y2):
        return 2

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cy < DVD_HEIGHT / 3:
        vert = 'top'
    elif cy < 2 * DVD_HEIGHT / 3:
        vert = 'mid'
    else:
        vert = 'bot'

    if cx < DVD_WIDTH / 3:
        horiz = 'left'
    elif cx < 2 * DVD_WIDTH / 3:
        horiz = 'center'
    else:
        horiz = 'right'

    return {
        ('top', 'left'): 7, ('top', 'center'): 8, ('top', 'right'): 9,
        ('mid', 'left'): 4, ('mid', 'center'): 5, ('mid', 'right'): 6,
        ('bot', 'left'): 1, ('bot', 'center'): 2, ('bot', 'right'): 3,
    }[(vert, horiz)]


def build_ass(
    entries:     list[tuple[int, int]],
    texts:       list[str],
    sub_path:    Path,
    output_path: Path,
) -> int:
    """
    Assemble an SSAFile from timing entries and OCR texts, then write it to disk.

    End-time priority (per CLAUDE.md spec):
      1. SPU's own STP_DSP offset (most accurate)
      2. Next subtitle's start time minus 100 ms gap
      3. LAST_FALLBACK_MS fixed duration (last entry only)
      4. Hard cap: MAX_SUBTITLE_MS
      5. Hard floor: MIN_SUBTITLE_MS

    Returns the number of events written.
    """
    subs = pysubs2.SSAFile()
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

        an = alignment_from_area(spu["x1"], spu["y1"], spu["x2"], spu["y2"])
        body = text.replace("\n", "\\N")
        if an != 2:
            body = f"{{\\an{an}}}{body}"

        subs.events.append(pysubs2.SSAEvent(
            start=start_ms, end=end_ms, text=body))

    subs.save(str(output_path))
    return len(subs.events)
