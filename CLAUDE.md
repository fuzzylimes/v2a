# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`v2a` converts bitmap subtitles embedded in MKV files into styled ASS subtitle files. It handles both **VobSub** (DVD sources) and **PGS** (Blu-ray sources), and is a CLI tool designed for disc backup libraries served by Jellyfin.

**VobSub pipeline (DVD):** `mkvmerge` identifies VobSub tracks → `mkvextract` pulls the `.idx`/`.sub` pair → SPU binary decoder renders bitmaps to PNGs → Tesseract OCRs each frame → `pysubs2` writes a styled `.ass` file next to the source MKV.

**PGS pipeline (Blu-ray):** `mkvmerge` identifies PGS tracks → `mkvextract` pulls a `.sup` file → the PGS binary decoder (`pgs.py`) renders each subtitle to a PNG and extracts its timing + bounding box in one pass → Tesseract OCRs each frame → `pysubs2` writes the same styled `.ass`.

The source type is detected automatically per track (via `codec_id`), and `process_file()` branches to the matching pipeline. Output naming and styling are identical for both.

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
| `cli.py` | Entry point. Parses args, checks system deps, orchestrates `process_file()`, which branches to `_convert_vobsub()` or `_convert_pgs()` based on the selected track's `kind`. |
| `tracks.py` | Calls `mkvmerge --identify` (JSON output). `identify_subtitle_tracks()` lists both VobSub and PGS tracks (each tagged with `kind`); handles interactive/batch track selection. |
| `extraction.py` | Runs `mkvextract` to produce `subs.idx`/`subs.sub` (VobSub) or `subs.sup` (PGS); parses the `.idx` text file for `(start_ms, filepos)` pairs and renders VobSub bitmaps as numbered PNGs. |
| `ocr.py` | Preprocesses each PNG (flatten alpha → grayscale → upscale → contrast boost; DVD adds 4x upscale + Otsu threshold + quiet-zone border + black-on-white inversion via `preprocess(dvd=True)`), calls Tesseract (`--psm 6 --oem 3`), then repairs leftover tall-glyph misreads (`|`/`¦`, `/`, `1`, `[`/`]` → `I`/`l`) and rebuilds stranded contractions whose `I` was dropped (`'ll` → `I'll`). |
| `spu.py` | Binary parser for MPEG-PS `private_stream_1` packets in the VobSub `.sub` file. Extracts `STP_DSP` end-time offset and `SET_DAREA` bounding box from each SPU control sequence. |
| `pgs.py` | Binary parser for Blu-ray PGS `.sup` files. Walks PG segments (PCS/PDS/ODS), decodes the RLE bitmaps, and returns per-subtitle records (start/end ms, rendered PNG, bounding box, canvas size). |
| `ass.py` | Assembles a `pysubs2.SSAFile`. `build_ass()` handles VobSub end-time priority + SPU data and maps boxes to 9-zone `\an` tags; `build_ass_pgs()` trusts PGS timing and positions each line exactly with `\pos()`. Both share `make_style()`. |

### End-time priority (in `ass.py::build_ass`, VobSub only)

1. SPU's own `STP_DSP` offset (most accurate)
2. Next subtitle's start time − 100 ms gap
3. Fixed 3-second fallback (last entry only)
4. Hard cap: 8 000 ms (prevents text freezing across scene breaks)
5. Hard floor: 500 ms

PGS (`build_ass_pgs`) is simpler: the `.sup` stream carries reliable start *and* end times (the presentation PCS and the following clear PCS), so timing is taken directly from each record — only the 500 ms floor applies, plus the 3-second fallback for a trailing subtitle the stream never explicitly clears. The 8 000 ms cap and the `STP_DSP` fallback chain are not used.

### Positioning

**VobSub (DVD):** `alignment_from_area()` divides the DVD frame (720x576) into a 3x3 zone grid and maps each subtitle's bounding-box center to an ASS `\an` numpad value. Bottom-center (normal dialogue) gets `\an2` and no override tag is written; all other zones get an explicit `{\an#}` tag prepended. This zone bucketing is a workaround for VobSub's frequently-useless `SET_DAREA` box (often set to the full frame), where exact position isn't reliably available.

**PGS (Blu-ray):** PGS objects carry an exact on-screen position, so `build_ass_pgs()` skips zones entirely and anchors every line at its bounding-box center with `{\an5\pos(cx,cy)}`, using the native canvas (e.g. 1920x1080) as PlayRes. This reproduces the disc's real placement uniformly — bottom dialogue, raised dialogue, and signs — with no guessing. Size-related style metrics (fontsize, outline, shadow) are scaled by `canvas_h / 576` so 1080p output matches the DVD on-screen size; font, colors, and weight are unchanged (`_scaled_style()`).

### Track selection convention

When multiple VobSub tracks need to be disambiguated (same language code, or no language hint in batch mode), `select_track()` picks the one with the most `num_index_entries`. This reliably selects the full dialogue track over signs/credits tracks regardless of language-code labeling errors (a common DVD authoring issue where the full dialogue track gets mislabeled with the wrong language code).

## Testing notes

Tests use `unittest.mock` to patch `subprocess.run` for track identification, and `tmp_path` fixtures for file I/O tests. The SPU tests construct binary payloads by hand using `struct.pack` — see `tests/test_spu.py` helper functions `_make_spu` and `_make_ps_packet` for the canonical way to build test data.
