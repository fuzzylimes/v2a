#!/usr/bin/env python3
"""
v2a.py — Extract VobSub subtitles from MKV(s), OCR them with Tesseract,
                and write a styled ASS file alongside each source video.

System dependencies:
    sudo apt install mkvtoolnix ffmpeg tesseract-ocr

Python dependencies:
    pip install pytesseract Pillow pysubs2

Usage (single file):
    python3 v2a.py movie.mkv

Usage (season folder — processes all MKVs):
    python3 v2a.py -d /path/to/season/

Flags:
    -d, --dir DIR         Process all .mkv files in DIR instead of a single file.
    -l, --language LANG   Auto-select the subtitle track matching this language
                          code (e.g. 'eng', 'en'). Falls back to prompting if not
                          found, or auto-selects when only one track exists.
"""

import sys
import os
import json
import re
import shutil
import struct
import subprocess
import tempfile
import argparse
from pathlib import Path


# ── Dependency check ──────────────────────────────────────────────────────────

def check_deps():
    missing = [t for t in ("mkvmerge", "mkvextract", "ffmpeg", "tesseract")
               if not shutil.which(t)]
    if missing:
        print(f"[error] Missing system tools: {', '.join(missing)}")
        print("        Run: sudo apt install mkvtoolnix ffmpeg tesseract-ocr")
        sys.exit(1)
    try:
        import pytesseract  # noqa
        from PIL import Image  # noqa
        import pysubs2  # noqa
    except ImportError as exc:
        print(f"[error] Missing Python package: {exc}")
        print("        Run: pip install pytesseract Pillow pysubs2")
        sys.exit(1)

check_deps()

import pytesseract
from PIL import Image, ImageEnhance
import pysubs2


# ── Style ─────────────────────────────────────────────────────────────────────
# Edit here to change the look of all output subtitles.
# pysubs2.Color(r, g, b, a) — alpha: 0 = fully opaque, 255 = fully transparent.

def make_style() -> pysubs2.SSAStyle:
    s = pysubs2.SSAStyle()
    s.fontname       = "Arial"
    s.fontsize       = 52
    s.primarycolor   = pysubs2.Color(255, 255, 255,   0)   # white text
    s.secondarycolor = pysubs2.Color(255, 255, 255,   0)
    s.outlinecolor   = pysubs2.Color(  0,   0,   0,   0)   # black border
    s.backcolor      = pysubs2.Color(  0,   0,   0, 160)   # soft shadow
    s.bold           = False
    s.italic         = False
    s.outline        = 2.5
    s.shadow         = 1.5
    s.alignment      = 2      # \an2 = bottom-center (overridden per-event for signs)
    s.marginl        = 40
    s.marginr        = 40
    s.marginv        = 30
    return s


# ── SPU binary parsing ────────────────────────────────────────────────────────
#
# Each subtitle image in the .sub file is a VobSub SPU (Sub Picture Unit).
# Its header encodes two things we care about:
#
#   1. Stop time  — encoded in the Display Control Sequence (DCSQ) that carries
#                   the STP_DSP (0x02) command. The DCSQ's delay field gives the
#                   offset from the packet's PTS in units of 1024/90 ms.
#
#   2. Display area — the SET_DAREA (0x05) command stores four 12-bit values
#                   packed into 6 bytes: x1, x2, y1, y2 (inclusive pixel coords).
#
# The .sub file is a stream of MPEG-PS private-stream-1 packets. Each entry in
# the .idx file gives the byte offset (filepos) of the start of one such packet.
# We skip the PS wrapper to reach the raw SPU payload.

CMD_FSTA_DSP  = 0x00   # forced start display      (0 params)
CMD_STA_DSP   = 0x01   # start display              (0 params)
CMD_STP_DSP   = 0x02   # stop display               (0 params)
CMD_SET_COLOR = 0x03   # set palette indices        (2 bytes)
CMD_SET_ALPHA = 0x04   # set alpha values           (2 bytes)
CMD_SET_DAREA = 0x05   # set display area           (6 bytes)
CMD_SET_DSPXA = 0x06   # set pixel data offsets     (4 bytes)
CMD_END       = 0xFF


def _skip_ps_header(data: bytes) -> int:
    """
    Return the byte offset where the raw SPU payload starts inside a PS packet.
    The .sub file uses MPEG-PS private_stream_1 (0xBD) packets.
    """
    if len(data) < 9 or data[:3] != b'\x00\x00\x01' or data[3] != 0xBD:
        return 0
    # Byte 8 holds the PES header data length (N).
    # After the fixed 9-byte header, skip N optional PES bytes, then 1 sub-stream ID byte.
    pes_header_len = data[8]
    return 9 + pes_header_len + 1


def parse_spu(data: bytes) -> dict:
    """
    Parse a raw SPU payload (PS header already stripped).
    Returns a dict with:
        end_ms        : stop-time offset in ms relative to the subtitle's start PTS.
                        None if STP_DSP command is absent — caller applies MAX cap.
        x1, y1, x2, y2 : bounding box in DVD pixel space. All None if not found.
    """
    result = dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)
    if len(data) < 4:
        return result

    # SPU layout:
    #   offset 0: SP_SIZE       (2 bytes) total SPU size
    #   offset 2: SP_DCSQ_START (2 bytes) offset to the first DCSQ from SPU start
    dcsq_offset = struct.unpack_from('>H', data, 2)[0]

    while dcsq_offset + 4 <= len(data):
        # DCSQ block:
        #   2 bytes: delay in 1024/90 ms units from this packet's PTS
        #   2 bytes: offset of the next DCSQ (same value = this is the last)
        delay_units = struct.unpack_from('>H', data, dcsq_offset)[0]
        next_offset = struct.unpack_from('>H', data, dcsq_offset + 2)[0]
        delay_ms    = int(delay_units * 1024 / 90)
        cmd_offset  = dcsq_offset + 4

        while cmd_offset < len(data):
            cmd = data[cmd_offset]
            cmd_offset += 1

            if cmd == CMD_END:
                break
            elif cmd in (CMD_FSTA_DSP, CMD_STA_DSP):
                pass
            elif cmd == CMD_STP_DSP:
                result['end_ms'] = delay_ms
            elif cmd in (CMD_SET_COLOR, CMD_SET_ALPHA):
                cmd_offset += 2
            elif cmd == CMD_SET_DAREA:
                if cmd_offset + 6 <= len(data):
                    b = data[cmd_offset: cmd_offset + 6]
                    result['x1'] = (b[0] << 4) | (b[1] >> 4)
                    result['x2'] = ((b[1] & 0x0F) << 8) | b[2]
                    result['y1'] = (b[3] << 4) | (b[4] >> 4)
                    result['y2'] = ((b[4] & 0x0F) << 8) | b[5]
                cmd_offset += 6
            elif cmd == CMD_SET_DSPXA:
                cmd_offset += 4
            else:
                break   # unknown command — stop to avoid reading garbage

        if next_offset <= dcsq_offset:
            break
        dcsq_offset = next_offset

    return result


def read_spu_at(sub_path: Path, filepos: int) -> dict:
    """Seek to filepos in the .sub file and parse the SPU packet found there."""
    try:
        with open(sub_path, 'rb') as f:
            f.seek(filepos)
            blob = f.read(2048)   # header + control data; we don't need image RLE
        spu_start = _skip_ps_header(blob)
        return parse_spu(blob[spu_start:])
    except Exception:
        return dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)


# ── Position → ASS alignment ──────────────────────────────────────────────────
# DVD dimensions (height varies by standard; thresholds work for both).
DVD_WIDTH  = 720
DVD_HEIGHT = 576   # PAL; NTSC is 480 — close enough for zone detection

def alignment_from_area(x1, y1, x2, y2) -> int:
    """
    Map a subtitle bounding box to an ASS \\an value (numpad layout 1–9).
    Normal dialogue sits in the bottom zone and gets \\an2 (bottom-center).
    Signs and forced subs in the top or middle get a more appropriate anchor.
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
        ('top','left'):7, ('top','center'):8, ('top','right'):9,
        ('mid','left'):4, ('mid','center'):5, ('mid','right'):6,
        ('bot','left'):1, ('bot','center'):2, ('bot','right'):3,
    }[(vert, horiz)]


# ── Track identification ──────────────────────────────────────────────────────

def identify_vobsub_tracks(mkv_path: Path) -> list[dict]:
    result = subprocess.run(
        ["mkvmerge", "--identify", "--identification-format", "json", str(mkv_path)],
        capture_output=True, text=True, check=True,
    )
    data   = json.loads(result.stdout)
    tracks = []
    sub_stream_idx = 0
    for track in data.get("tracks", []):
        if track.get("type") == "subtitles":
            if track.get("codec") == "VobSub":
                props = track.get("properties", {})
                tracks.append({
                    "mkv_id":    track["id"],
                    "sub_index": sub_stream_idx,
                    "language":  props.get("language", "und"),
                    "name":      props.get("track_name", ""),
                })
            sub_stream_idx += 1
    return tracks


def select_track(tracks: list[dict], lang_hint: str | None, batch: bool) -> dict | None:
    """
    Pick a VobSub track.
    Priority: language match > only-one-track > batch auto-first > interactive prompt.

    When multiple tracks share the same language code, the LAST one is chosen.
    This follows a common disc authoring convention where the first same-language
    track is signs/forced-only and the second is the full dialogue track.
    """
    if not tracks:
        return None

    if lang_hint:
        lang_hint = lang_hint.lower()
        matches = [t for t in tracks if t["language"].lower() == lang_hint]
        if matches:
            if len(matches) > 1:
                print(f"  [info] {len(matches)} tracks match language '{lang_hint}' "
                      f"— selecting the last one (ID {matches[-1]['mkv_id']}), "
                      "assumed to be the full dialogue track.")
            return matches[-1]
        print(f"  [warn] No track matching language '{lang_hint}'.")

    if len(tracks) == 1:
        return tracks[0]

    if batch:
        print(f"  [warn] Multiple VobSub tracks — auto-selecting first "
              f"(ID {tracks[0]['mkv_id']}, lang={tracks[0]['language']}).")
        print("         Use -l/--language to be explicit.")
        return tracks[0]

    print("\n  VobSub subtitle tracks:")
    for i, t in enumerate(tracks):
        label = f"    [{i}]  Track ID {t['mkv_id']}  |  lang={t['language']}"
        if t["name"]:
            label += f"  |  \"{t['name']}\""
        print(label)
    while True:
        raw = input("  Select track [number]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(tracks):
            return tracks[int(raw)]
        print(f"  Enter a number between 0 and {len(tracks) - 1}.")


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_vobsub(mkv_path: Path, mkv_track_id: int, out_dir: Path) -> tuple[Path, Path]:
    subprocess.run(
        ["mkvextract", "tracks", str(mkv_path), f"{mkv_track_id}:{out_dir / 'subs'}"],
        check=True, capture_output=True,
    )
    idx, sub = out_dir / "subs.idx", out_dir / "subs.sub"
    if not idx.exists() or not sub.exists():
        raise RuntimeError("mkvextract did not produce the expected .idx/.sub pair.")
    return idx, sub


def parse_idx(idx_path: Path) -> list[tuple[int, int]]:
    """
    Returns a list of (start_ms, filepos) tuples from the .idx file.
    filepos is the hex byte offset into .sub for that subtitle's PS packet.
    """
    entries = []
    pat = re.compile(
        r"^timestamp:\s*(\d{2}):(\d{2}):(\d{2}):(\d{3}),\s*filepos:\s*([0-9a-fA-F]+)"
    )
    with open(idx_path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                h, mn, s, ms = map(int, m.groups()[:4])
                filepos  = int(m.group(5), 16)
                start_ms = h * 3_600_000 + mn * 60_000 + s * 1_000 + ms
                entries.append((start_ms, filepos))
    return entries


def extract_frames(idx_path: Path, out_dir: Path) -> list[Path]:
    """Render each subtitle bitmap to a PNG using ffmpeg's VobSub demuxer."""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir()
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-i", str(idx_path),
            "-vsync", "0",
            str(frames_dir / "frame_%06d.png"),
        ],
        check=True,
    )
    return sorted(frames_dir.glob("frame_*.png"))


# ── OCR ───────────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image) -> Image.Image:
    """
    Prepare a VobSub frame for Tesseract.
    Flatten transparent background → black, convert to grayscale, upscale 3×,
    then boost contrast. The upscale is important: DVD subtitle strips are
    typically ~720×60px, well below Tesseract's comfortable resolution.
    """
    bg   = Image.new("RGBA", img.size, (0, 0, 0, 255))
    flat = Image.alpha_composite(bg, img.convert("RGBA")).convert("L")
    w, h = flat.size
    flat = flat.resize((w * 3, h * 3), Image.LANCZOS)
    return ImageEnhance.Contrast(flat).enhance(1.8)


def ocr_frames(frame_paths: list[Path]) -> list[str]:
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


# ── ASS assembly ──────────────────────────────────────────────────────────────

# A subtitle that runs longer than this is almost certainly a bad end-time from
# a malformed or missing STP_DSP command. Cap it so nothing freezes on screen.
MAX_SUBTITLE_MS  = 8_000
MIN_SUBTITLE_MS  =   500
LAST_FALLBACK_MS = 3_000   # used only for the very last entry if STP_DSP is absent


def build_ass(
    entries:     list[tuple[int, int]],
    texts:       list[str],
    sub_path:    Path,
    output_path: Path,
) -> int:
    subs = pysubs2.SSAFile()
    subs.styles["Default"] = make_style()

    total = len(entries)

    for i, ((start_ms, filepos), text) in enumerate(zip(entries, texts)):
        if not text:
            continue

        spu = read_spu_at(sub_path, filepos)

        # End time: SPU's own stop offset > next-start heuristic > last-entry fallback
        if spu["end_ms"] is not None:
            end_ms = start_ms + spu["end_ms"]
        elif i + 1 < total:
            end_ms = entries[i + 1][0] - 100
        else:
            end_ms = start_ms + LAST_FALLBACK_MS

        end_ms = min(end_ms, start_ms + MAX_SUBTITLE_MS)
        end_ms = max(end_ms, start_ms + MIN_SUBTITLE_MS)

        # Alignment: derive from bounding box so signs land in the right zone
        an   = alignment_from_area(spu["x1"], spu["y1"], spu["x2"], spu["y2"])
        body = text.replace("\n", "\\N")
        if an != 2:
            body = f"{{\\an{an}}}{body}"

        subs.events.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=body))

    subs.save(str(output_path))
    return len(subs.events)


# ── Per-file driver ───────────────────────────────────────────────────────────

def process_file(mkv_path: Path, lang_hint: str | None, batch: bool) -> bool:
    print(f"\n{'─' * 60}")
    print(f"File : {mkv_path.name}")

    tracks = identify_vobsub_tracks(mkv_path)
    track  = select_track(tracks, lang_hint, batch)

    if track is None:
        print("  [skip] No VobSub tracks found.")
        return False

    print(f"  Track ID {track['mkv_id']}  |  lang={track['language']}")

    with tempfile.TemporaryDirectory(prefix="v2a_") as tmp:
        tmp_dir = Path(tmp)

        print("  [1/4] Extracting VobSub...")
        try:
            idx_path, sub_path = extract_vobsub(mkv_path, track["mkv_id"], tmp_dir)
        except RuntimeError as exc:
            print(f"  [error] {exc}")
            return False

        print("  [2/4] Parsing .idx timestamps and file positions...")
        entries = parse_idx(idx_path)
        print(f"        {len(entries)} subtitle entries.")

        print("  [3/4] Rendering subtitle bitmaps via ffmpeg...")
        frames = extract_frames(idx_path, tmp_dir)
        print(f"        {len(frames)} frames rendered.")

        if len(frames) != len(entries):
            print(f"  [warn] Frame/entry mismatch ({len(frames)} vs {len(entries)}). "
                  "Truncating to shorter list.")
            n       = min(len(frames), len(entries))
            frames  = frames[:n]
            entries = entries[:n]

        print("  [4/4] Running OCR...")
        texts = ocr_frames(frames)

        # sub_path must be read before the tempdir is cleaned up
        out_path = mkv_path.parent / f"{mkv_path.stem}.{track['language']}.ass"
        count    = build_ass(entries, texts, sub_path, out_path)

    print(f"  Done  →  {out_path.name}  ({count} events written)")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert VobSub subtitles in MKV files to styled ASS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("mkv", nargs="?", metavar="FILE", help="Single .mkv to process.")
    group.add_argument("-d", "--dir",    metavar="DIR",  help="Directory of .mkv files.")
    parser.add_argument(
        "-l", "--language", metavar="LANG",
        help="Language code to auto-select (e.g. eng, en). "
             "Recommended when using -d with files that have multiple VobSub tracks.",
    )
    args = parser.parse_args()

    if args.dir:
        folder    = Path(args.dir).resolve()
        if not folder.is_dir():
            print(f"[error] Not a directory: {folder}")
            sys.exit(1)
        mkv_files = sorted(folder.glob("*.mkv"))
        if not mkv_files:
            print(f"[error] No .mkv files found in {folder}")
            sys.exit(1)
        print(f"Found {len(mkv_files)} MKV file(s) in {folder.name}/")
        ok = skip = 0
        for mkv in mkv_files:
            if process_file(mkv, args.language, batch=True):
                ok += 1
            else:
                skip += 1
        print(f"\n{'─' * 60}")
        print(f"Batch complete — {ok} converted, {skip} skipped.")
    else:
        mkv_path = Path(args.mkv).resolve()
        if not mkv_path.exists():
            print(f"[error] File not found: {mkv_path}")
            sys.exit(1)
        process_file(mkv_path, args.language, batch=False)


if __name__ == "__main__":
    main()
