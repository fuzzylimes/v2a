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


def extract_frames(idx_path: Path, out_dir: Path) -> list[Path]:
    """
    Render each VobSub bitmap to a numbered PNG using ffmpeg's VobSub demuxer.

    Output files are written to out_dir/frames/ as frame_000001.png, etc.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir()
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-i", str(idx_path),
            "-fps_mode", "passthrough",
            str(frames_dir / "frame_%06d.png"),
        ],
        check=True, capture_output=True,
    )
    return sorted(frames_dir.glob("frame_*.png"))
