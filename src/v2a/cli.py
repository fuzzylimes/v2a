"""
v2a — Extract VobSub subtitles from MKV(s), OCR them with Tesseract,
      and write a styled ASS file alongside each source video.

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

    Returns True if an .ass file was written, False if the file was skipped or
    an error prevented completion.
    """
    from .tracks import identify_vobsub_tracks, select_track
    from .extraction import extract_vobsub, parse_idx, extract_frames
    from .ocr import ocr_frames
    from .ass import build_ass

    print(f"\n{'─' * 60}")
    print(f"File : {mkv_path.name}")

    try:
        tracks = identify_vobsub_tracks(mkv_path)
    except subprocess.CalledProcessError as exc:
        print(f"  [error] mkvmerge failed: {exc}")
        return False

    track = select_track(tracks, lang_hint, batch)
    if track is None:
        print("  [skip] No VobSub tracks found.")
        return False

    print(f"  Track ID {track['mkv_id']}  |  lang={track['language']}")

    lang_label = lang_hint or track['language']
    out_path = mkv_path.parent / f"{mkv_path.stem}.{lang_label}.ass"
    if not force and out_path.exists():
        print(f"  [skip] Output already exists: {out_path.name}  (use --force to overwrite)")
        return False

    with tempfile.TemporaryDirectory(prefix="v2a_") as tmp:
        tmp_dir = Path(tmp)

        print("  [1/4] Extracting VobSub...")
        try:
            idx_path, sub_path = extract_vobsub(mkv_path, track["mkv_id"], tmp_dir)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"  [error] Extraction failed: {exc}")
            return False

        print("  [2/4] Parsing .idx timestamps and file positions...")
        try:
            entries = parse_idx(idx_path)
        except Exception as exc:
            print(f"  [error] Failed to parse .idx: {exc}")
            return False
        print(f"        {len(entries)} subtitle entries.")

        print("  [3/4] Decoding subtitle bitmaps...")
        try:
            frames = extract_frames(idx_path, sub_path, entries, tmp_dir)
        except Exception as exc:
            print(f"  [error] Bitmap decoding failed: {exc}")
            return False
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
        count = build_ass(entries, texts, sub_path, out_path, frames=frames, verbose=verbose)

        if keep_frames:
            import shutil as _shutil
            dest = out_path.parent / (out_path.stem + ".frames")
            if dest.exists():
                _shutil.rmtree(dest)
            _shutil.copytree(tmp_dir / "frames", dest)
            print(f"  Frames saved → {dest}")

    print(f"  Done  →  {out_path.name}  ({count} events written)")
    return True


def main() -> None:
    check_deps()

    parser = argparse.ArgumentParser(
        description="Convert VobSub subtitles in MKV files to styled ASS.",
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
        mkv_files = sorted(folder.glob("*.mkv"))
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
