# v2a — Project Summary

## What It Is
A single-file Python CLI tool that extracts VobSub bitmap subtitles from MKV files, OCRs them with Tesseract, and writes a styled ASS subtitle file alongside the source video. Intended for converting a DVD backup library hosted on Jellyfin (Ubuntu 24 LTS).

## System Dependencies
```
sudo apt install mkvtoolnix ffmpeg tesseract-ocr
pip install pytesseract Pillow pysubs2
```

## Usage
```bash
# Single file (interactive track selection)
python3 v2a.py movie.mkv

# Batch — full season folder
python3 v2a.py -d /path/to/season/ -l eng
```

## Pipeline (in order)
1. **`identify_vobsub_tracks`** — runs `mkvmerge --identify` (JSON output) to list all VobSub tracks in the file
2. **`select_track`** — picks a track by language code; when multiple tracks match the same language, picks the **last** one (convention: first = signs-only, last = full dialogue); falls back to interactive prompt in single-file mode
3. **`extract_vobsub`** — runs `mkvextract tracks` to produce a `.idx`/`.sub` pair in a temp directory
4. **`parse_idx`** — parses the plain-text `.idx` file for `(start_ms, filepos)` tuples; `filepos` is the hex byte offset into `.sub` for each subtitle's MPEG-PS packet
5. **`extract_frames`** — runs `ffmpeg` against the `.idx` to render each subtitle bitmap as a numbered PNG
6. **`read_spu_at` / `parse_spu`** — seeks to each `filepos` in the `.sub` binary and parses the SPU (Sub Picture Unit) packet to extract:
   - **True end time** from the `STP_DSP` control command (stored as a delay in `1024/90 ms` units from the subtitle's PTS)
   - **Bounding box** (`x1, y1, x2, y2`) from the `SET_DAREA` control command
7. **`ocr_frames`** — preprocesses each PNG (flatten alpha to black, upscale 3×, contrast boost) then runs Tesseract (`--psm 6 --oem 3`)
8. **`alignment_from_area`** — maps the DVD bounding box to an ASS `\an` numpad value (1–9) by dividing the 720×576 frame into a 3×3 zone grid; emits `{\an N}` override tag for anything that isn't bottom-center (`\an2`)
9. **`build_ass`** — assembles a `pysubs2.SSAFile` with a single named style ("Default") and writes the `.ass` file next to the source MKV

## End Time Logic (priority order)
1. SPU's own `STP_DSP` offset (most accurate)
2. Next subtitle's start time minus 100 ms gap (fallback)
3. 3000 ms fixed duration (last entry only)
4. Hard cap: `MAX_SUBTITLE_MS = 8000` — prevents frozen text across scene breaks
5. Hard floor: `MIN_SUBTITLE_MS = 500`

## ASS Style (defined in `make_style`)
- Font: Arial 52pt
- White text, black outline (2.5px), soft shadow (1.5px, 160 alpha)
- Bottom-center alignment (`\an2`) as default; per-event `\an` override for signs
- Margins: 40px left/right, 30px vertical

## Output
- File is written as `{source_stem}.{language}.ass` in the same directory as the MKV
- Jellyfin picks it up automatically as an external subtitle track
- Source MKV is never modified

## Known Limitations / Potential Future Work
- Sign positioning uses zone-based `\an` tags rather than exact `\pos(x,y)` coordinates — pixel-perfect sign placement would require remapping DVD pixel space to the video's actual resolution
- OCR accuracy degrades on italicized or stylized fonts; a manual review pass is recommended before considering the output final
- No recursive directory walking — `-d` processes one flat folder, not a full show tree
