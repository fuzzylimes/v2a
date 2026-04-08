"""VobSub extraction, .idx parsing, and bitmap frame rendering."""

import re
import subprocess
from pathlib import Path


def extract_vobsub(mkv_path: Path, mkv_track_id: int, out_dir: Path) -> tuple[Path, Path]:
    """
    Extract a single VobSub track from mkv_path into out_dir.

    Produces out_dir/subs.idx and out_dir/subs.sub.
    Raises RuntimeError if either file is missing after extraction.
    """
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
    Parse the plain-text .idx file for subtitle timing and position data.

    Returns a list of (start_ms, filepos) tuples where filepos is the hex byte
    offset into the .sub file for that subtitle's MPEG-PS packet.
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


def _parse_palette(idx_path: Path) -> list[tuple[int, int, int]]:
    """Extract the 16-colour RGB palette from the .idx file's 'palette:' line."""
    with open(idx_path) as f:
        for line in f:
            if line.startswith('palette:'):
                parts = line.split(':', 1)[1].strip().split(',')
                result = []
                for p in parts:
                    p = p.strip()
                    if len(p) == 6:
                        result.append((int(p[0:2], 16), int(p[2:4], 16), int(p[4:6], 16)))
                return result
    return [(i * 17, i * 17, i * 17) for i in range(16)]


def extract_frames(
    idx_path: Path,
    sub_path: Path,
    entries: list[tuple[int, int]],
    out_dir: Path,
) -> list[Path]:
    """
    Render each VobSub bitmap to a numbered PNG by decoding the SPU binary data.

    Returns one Path per entry (entries whose bounding box is missing get a
    blank 4×4 image so the frame/entry lists stay aligned).
    """
    from PIL import Image
    from .spu import read_spu_bytes, decode_spu_image, parse_spu

    frames_dir = out_dir / "frames"
    frames_dir.mkdir()
    palette = _parse_palette(idx_path)
    frames = []
    for i, (_, filepos) in enumerate(entries, 1):
        spu_data = read_spu_bytes(sub_path, filepos)
        meta = parse_spu(spu_data)
        path = frames_dir / f"frame_{i:06d}.png"
        if None not in (meta['x1'], meta['y1'], meta['x2'], meta['y2']):
            img = decode_spu_image(
                spu_data, meta['x1'], meta['y1'], meta['x2'], meta['y2'], palette)
        else:
            img = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
        img.save(path)
        frames.append(path)
    return frames
