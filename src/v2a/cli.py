"""
v2a — Extract bitmap subtitles from MKV(s) — VobSub (DVD) or PGS (Blu-ray) —
      OCR them with Tesseract, and write a styled ASS file alongside each
      source video. The source type is detected automatically per track.

System dependencies:
    sudo apt install mkvtoolnix ffmpeg tesseract-ocr

Usage (single file):
    v2a movie.mkv

Usage (season folder — processes all MKVs):
    v2a -d /path/to/season/ -l eng

Flags:
    -d, --dir DIR         Process all .mkv files in DIR instead of a single file.
    -l, --language LANG   Auto-select the subtitle track matching this language
                          code (e.g. 'eng', 'en'). Falls back to prompting if not
                          found, or auto-selects when only one track exists.
    --force               Re-process files that already have a .ass output.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_deps() -> None:
    """Abort with a clear message if any required system tool or Python package is missing."""
    missing = [t for t in ("mkvmerge", "mkvextract", "tesseract")
               if not shutil.which(t)]
    if missing:
        print(f"[error] Missing system tools: {', '.join(missing)}")
        print("        Run: sudo apt install mkvtoolnix tesseract-ocr")
        sys.exit(1)
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        import pysubs2  # noqa: F401
    except ImportError as exc:
        print(f"[error] Missing Python package: {exc}")
        print("        Run: pip install pytesseract Pillow pysubs2")
        sys.exit(1)


def process_file(mkv_path: Path, lang_hint: str | None, batch: bool, force: bool, keep_frames: bool = False, verbose: bool = False) -> bool:
    """
    Run the full pipeline on a single MKV file.

    Every selected subtitle track is converted (not just one); same-language
    tracks are disambiguated by cue count and named for Jellyfin (see
    tracks.plan_outputs / tracks.output_filename).

    Returns True if at least one .ass file was written, False if the file was
    skipped entirely or an error prevented any output.
    """
    from .tracks import (
        identify_subtitle_tracks, select_tracks, plan_outputs, output_filename,
        needs_interactive_prompt,
    )

    print(f"\n{'─' * 60}")
    print(f"File : {mkv_path.name}")

    print("  Identifying subtitle tracks (mkvmerge --identify)...")
    try:
        tracks = identify_subtitle_tracks(mkv_path)
    except subprocess.TimeoutExpired:
        print("  [error] mkvmerge --identify timed out — file may be unreadable "
              "or on a stalled mount. Skipping.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"  [error] mkvmerge failed: {exc}")
        return False

    if not tracks:
        print("  [skip] No VobSub or PGS subtitle tracks found.")
        return False

    summary = ", ".join(
        f"ID {t['mkv_id']} ({t['kind']}/{t['language']})" for t in tracks)
    print(f"  Found {len(tracks)} bitmap subtitle track(s): {summary}")

    with tempfile.TemporaryDirectory(prefix="v2a_") as tmp:
        tmp_dir = Path(tmp)
        extracted: dict[int, dict] = {}

        def prepare(track: dict) -> bool:
            """Extract a track once and record its cue count. False on failure."""
            if track["mkv_id"] in extracted:
                return True
            info = _extract_and_count(mkv_path, track, tmp_dir)
            if info is None:
                return False
            extracted[track["mkv_id"]] = info
            return True

        # The interactive menu shows cue counts, so when we will prompt we must
        # extract every track up front. Otherwise we only touch the chosen set.
        if needs_interactive_prompt(tracks, lang_hint, batch):
            print(f"  Extracting all {len(tracks)} track(s) up front to count cues "
                  "for the selection menu...")
            for t in tracks:
                prepare(t)
            chosen = select_tracks(tracks, lang_hint, batch)
        else:
            chosen = select_tracks(tracks, lang_hint, batch)
            chosen = [t for t in chosen if prepare(t)]

        if not chosen:
            print("  [skip] No tracks selected.")
            return False

        wrote = 0
        for p in plan_outputs(chosen):
            info = extracted.get(p["mkv_id"])
            if info is None:
                continue
            out_path = mkv_path.parent / output_filename(
                mkv_path.stem, p["lang"], p["title"], p["flags"])

            print(f"\n  Track ID {p['mkv_id']}  |  lang={p['language']}  |  "
                  f"{p['kind']}  |  {p.get('count', 0)} cues  →  {out_path.name}")

            if not force and out_path.exists():
                print(f"  [skip] Output already exists (use --force to overwrite).")
                continue

            if info["kind"] == "pgs":
                count = _convert_pgs(info["sup"], out_path, info["dir"], verbose)
            else:
                count = _convert_vobsub(info["idx"], info["sub"], out_path, info["dir"], verbose)
            if count is None:
                continue

            if keep_frames:
                _save_frames(info["dir"], out_path)

            print(f"  Done  →  {out_path.name}  ({count} events written)")
            wrote += 1

    return wrote > 0


def _extract_and_count(mkv_path: Path, track: dict, tmp_dir: Path) -> dict | None:
    """
    Extract one track into its own subdir and record its cue count on the track.

    Returns a dict describing the extracted files (keys: kind, dir, and either
    'sup' or 'idx'/'sub'), or None on extraction failure. Sets track['count'].
    """
    from .extraction import extract_pgs, extract_vobsub, parse_idx
    from .pgs import count_sup

    track_dir = tmp_dir / f"track_{track['mkv_id']}"
    track_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting track {track['mkv_id']} ({track['kind']}) "
          "with mkvextract... (this can take a minute on large files)")
    try:
        if track["kind"] == "pgs":
            sup = extract_pgs(mkv_path, track["mkv_id"], track_dir)
            track["count"] = count_sup(sup)
            info = {"kind": "pgs", "dir": track_dir, "sup": sup}
        else:
            idx, sub = extract_vobsub(mkv_path, track["mkv_id"], track_dir)
            track["count"] = len(parse_idx(idx))
            info = {"kind": "vobsub", "dir": track_dir, "idx": idx, "sub": sub}
    except subprocess.TimeoutExpired:
        print(f"  [error] mkvextract timed out for track {track['mkv_id']} — skipping.")
        return None
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"  [error] Extraction failed for track {track['mkv_id']}: {exc}")
        return None

    if track["count"] == 0:
        print(f"  [warn] Track {track['mkv_id']} extracted but has 0 cues "
              "(empty subtitle stream).")
    else:
        print(f"        Track {track['mkv_id']}: {track['count']} cues.")
    return info


def _save_frames(track_dir: Path, out_path: Path) -> None:
    """Copy a track's decoded frames next to its output for inspection."""
    import shutil as _shutil
    src = track_dir / "frames"
    if not src.exists():
        return
    dest = out_path.parent / (out_path.stem + ".frames")
    if dest.exists():
        _shutil.rmtree(dest)
    _shutil.copytree(src, dest)
    print(f"  Frames saved → {dest}")


def _convert_vobsub(idx_path: Path, sub_path: Path, out_path: Path,
                    track_dir: Path, verbose: bool) -> int | None:
    """Run the VobSub (DVD) pipeline on already-extracted files. Returns event count or None."""
    from .extraction import parse_idx, extract_frames
    from .ocr import ocr_frames
    from .ass import build_ass

    print("  [1/3] Parsing .idx and decoding subtitle bitmaps...")
    try:
        entries = parse_idx(idx_path)
        frames = extract_frames(idx_path, sub_path, entries, track_dir)
    except Exception as exc:
        print(f"  [error] Bitmap decoding failed: {exc}")
        return None
    print(f"        {len(frames)} frames rendered.")

    if len(frames) != len(entries):
        print(f"  [warn] Frame/entry mismatch ({len(frames)} vs {len(entries)}). "
              "Truncating to shorter list.")
        n       = min(len(frames), len(entries))
        frames  = frames[:n]
        entries = entries[:n]

    print("  [2/3] Running OCR...")
    texts = ocr_frames(frames)

    print("  [3/3] Writing ASS...")
    # sub_path must be read before the tempdir is cleaned up
    return build_ass(entries, texts, sub_path, out_path, frames=frames, verbose=verbose)


def _convert_pgs(sup_path: Path, out_path: Path,
                 track_dir: Path, verbose: bool) -> int | None:
    """Run the PGS (Blu-ray) pipeline on an already-extracted .sup. Returns event count or None."""
    from .pgs import parse_sup
    from .ocr import ocr_frames
    from .ass import build_ass_pgs

    print("  [1/3] Decoding PGS subtitle bitmaps and timing...")
    try:
        records = parse_sup(sup_path, track_dir / "frames")
    except Exception as exc:
        print(f"  [error] PGS decoding failed: {exc}")
        return None
    print(f"        {len(records)} subtitle entries.")

    print("  [2/3] Running OCR...")
    texts = ocr_frames([r["image_path"] for r in records])

    print("  [3/3] Writing ASS...")
    return build_ass_pgs(records, texts, out_path, verbose=verbose)


def main() -> None:
    check_deps()

    parser = argparse.ArgumentParser(
        description="Convert VobSub (DVD) or PGS (Blu-ray) subtitles in MKV files to styled ASS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("mkv", nargs="?", metavar="FILE", help="Single .mkv to process.")
    group.add_argument("-d", "--dir", metavar="DIR", help="Directory of .mkv files.")
    parser.add_argument(
        "-l", "--language", metavar="LANG",
        help="Language code to auto-select (e.g. eng, en). "
             "Recommended when using -d with files that have multiple VobSub tracks.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recurse into subdirectories when using -d/--dir (e.g. a full show library).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process and overwrite existing .ass files (default: skip).",
    )
    parser.add_argument(
        "--keep-frames", action="store_true",
        help="Save decoded subtitle frames to a frames/ subdirectory next to the output for inspection.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print SPU bounding-box coordinates and alignment tag for each subtitle (useful for debugging positioning).",
    )
    args = parser.parse_args()

    if args.dir:
        folder = Path(args.dir).resolve()
        if not folder.is_dir():
            print(f"[error] Not a directory: {folder}")
            sys.exit(1)
        mkv_files = sorted(folder.rglob("*.mkv") if args.recursive else folder.glob("*.mkv"))
        if not mkv_files:
            print(f"[error] No .mkv files found in {folder}")
            sys.exit(1)
        print(f"Found {len(mkv_files)} MKV file(s) in {folder.name}/")
        ok = skip = 0
        for mkv in mkv_files:
            if process_file(mkv, args.language, batch=True, force=args.force, keep_frames=args.keep_frames, verbose=args.verbose):
                ok += 1
            else:
                skip += 1
        print(f"\n{'─' * 60}")
        print(f"Batch complete — {ok} converted, {skip} skipped.")
    else:
        if args.mkv is None:
            parser.error("FILE is required when -d/--dir is not specified.")
        mkv_path = Path(args.mkv).resolve()
        if not mkv_path.exists():
            print(f"[error] File not found: {mkv_path}")
            sys.exit(1)
        process_file(mkv_path, args.language, batch=False, force=args.force, keep_frames=args.keep_frames, verbose=args.verbose)


if __name__ == "__main__":
    main()
