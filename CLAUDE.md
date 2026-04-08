# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`v2a` converts VobSub bitmap subtitles embedded in MKV files into styled ASS subtitle files. It's a CLI tool designed for DVD backup libraries served by Jellyfin.

**Pipeline:** `mkvmerge` identifies VobSub tracks → `mkvextract` pulls the `.idx`/`.sub` pair → SPU binary decoder renders bitmaps to PNGs → Tesseract OCRs each frame → `pysubs2` writes a styled `.ass` file next to the source MKV.

## System dependencies (must be on PATH)

```bash
sudo apt install mkvtoolnix ffmpeg tesseract-ocr
```

## Development commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
# or with uv:
uv sync

# Run all tests
pytest

# Run a single test file
pytest tests/test_spu.py

# Run a specific test class or function
pytest tests/test_spu.py::TestParseSpu::test_extracts_end_ms_and_bounding_box

# Run the CLI directly
v2a movie.mkv
v2a -d /path/to/season/ -l eng
```

## Architecture

All source lives in `src/v2a/`. Each module has a single responsibility:

| Module | Role |
|--------|------|
| `cli.py` | Entry point. Parses args, checks system deps, orchestrates `process_file()` for single or batch mode. |
| `tracks.py` | Calls `mkvmerge --identify` (JSON output) to list VobSub tracks; handles interactive/batch track selection. |
| `extraction.py` | Runs `mkvextract` to produce `subs.idx`/`subs.sub`; parses the `.idx` text file for `(start_ms, filepos)` pairs; runs `ffmpeg` to render bitmaps as numbered PNGs. |
| `ocr.py` | Preprocesses each PNG (flatten alpha → grayscale → 3x upscale → contrast boost) then calls Tesseract (`--psm 6 --oem 3`). |
| `spu.py` | Binary parser for MPEG-PS `private_stream_1` packets in the `.sub` file. Extracts `STP_DSP` end-time offset and `SET_DAREA` bounding box from each SPU control sequence. |
| `ass.py` | Assembles a `pysubs2.SSAFile` from timing entries, OCR texts, and SPU data. Handles end-time priority logic and maps bounding boxes to ASS `\an` alignment tags. |

### End-time priority (in `ass.py::build_ass`)

1. SPU's own `STP_DSP` offset (most accurate)
2. Next subtitle's start time − 100 ms gap
3. Fixed 3-second fallback (last entry only)
4. Hard cap: 8 000 ms (prevents text freezing across scene breaks)
5. Hard floor: 500 ms

### Sign positioning

`alignment_from_area()` divides the DVD frame (720x576) into a 3x3 zone grid and maps each subtitle's bounding-box center to an ASS `\an` numpad value. Bottom-center (normal dialogue) gets `\an2` and no override tag is written; all other zones get an explicit `{\an#}` tag prepended.

### Track selection convention

When multiple VobSub tracks need to be disambiguated (same language code, or no language hint in batch mode), `select_track()` picks the one with the most `num_index_entries`. This reliably selects the full dialogue track over signs/credits tracks regardless of language-code labeling errors (a common DVD authoring issue where the full dialogue track gets mislabeled with the wrong language code).

## Testing notes

Tests use `unittest.mock` to patch `subprocess.run` for track identification, and `tmp_path` fixtures for file I/O tests. The SPU tests construct binary payloads by hand using `struct.pack` — see `tests/test_spu.py` helper functions `_make_spu` and `_make_ps_packet` for the canonical way to build test data.
