"""Subtitle track identification and selection (VobSub for DVD, PGS for Blu-ray)."""

import json
import subprocess
from pathlib import Path

# Map a track's codec to the pipeline that handles it, keyed by both the
# canonical Matroska codec_id and mkvmerge's human-readable codec name. VobSub
# comes from DVD sources; PGS (Presentation Graphic Stream) from Blu-ray.
_CODEC_KIND = {
    "S_VOBSUB": "vobsub",
    "VobSub": "vobsub",
    "S_HDMV/PGS": "pgs",
    "HDMV PGS": "pgs",
    "PGS": "pgs",
}


def _track_kind(track: dict, props: dict) -> str | None:
    """Resolve a subtitle track's pipeline kind from its codec_id or codec name."""
    return (_CODEC_KIND.get(props.get("codec_id", ""))
            or _CODEC_KIND.get(track.get("codec", "")))


def identify_subtitle_tracks(mkv_path: Path) -> list[dict]:
    """
    Return the bitmap subtitle tracks v2a can convert (VobSub and PGS).

    Each track is a dict with:
        mkv_id   : track ID used by mkvextract
        language : BCP-47 language code (e.g. 'eng', 'und')
        name     : human-readable track name (may be empty)
        entries  : number of index entries (0 for PGS — not reported by mkvmerge)
        kind     : 'vobsub' or 'pgs' — selects the extraction/decode pipeline
    """
    result = subprocess.run(
        ["mkvmerge", "--identify", "--identification-format", "json", str(mkv_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    tracks = []
    for track in data.get("tracks", []):
        if track.get("type") != "subtitles":
            continue
        props = track.get("properties", {})
        kind = _track_kind(track, props)
        if kind is None:
            continue
        tracks.append({
            "mkv_id":   track["id"],
            "language": props.get("language", "und"),
            "name":     props.get("track_name", ""),
            "entries":  props.get("num_index_entries", 0),
            "kind":     kind,
        })
    return tracks


def identify_vobsub_tracks(mkv_path: Path) -> list[dict]:
    """
    Return only the VobSub tracks found in mkv_path (DVD sources).

    Kept for callers that only handle VobSub; the dicts omit the 'kind' key.
    """
    return [
        {k: t[k] for k in ("mkv_id", "language", "name", "entries")}
        for t in identify_subtitle_tracks(mkv_path)
        if t["kind"] == "vobsub"
    ]


def select_track(tracks: list[dict], lang_hint: str | None, batch: bool) -> dict | None:
    """
    Pick a VobSub track.

    Priority: language match > only-one-track > most-entries auto-select > interactive prompt.

    When multiple tracks match (same language or no lang_hint in batch mode) the
    one with the most index entries is chosen. This reliably picks the full dialogue
    track over signs/credits tracks, regardless of language-code labeling errors.
    """
    if not tracks:
        return None

    if lang_hint:
        lang_hint = lang_hint.lower()
        matches = [t for t in tracks if t["language"].lower() == lang_hint]
        if matches:
            best = max(matches, key=lambda t: t["entries"])
            if len(matches) > 1:
                print(f"  [info] {len(matches)} tracks match language '{lang_hint}' "
                      f"— selecting largest (ID {best['mkv_id']}, "
                      f"{best['entries']} entries).")

            # If another track is significantly larger, it is probably the real
            # full-dialogue track regardless of language labeling.
            non_matches = [t for t in tracks if t["language"].lower() != lang_hint]
            if non_matches:
                dominant = max(non_matches, key=lambda t: t["entries"])
                threshold = max(best["entries"] * 3, 1)
                if dominant["entries"] >= threshold:
                    print(f"  [warn] Requested language '{lang_hint}' track "
                          f"(ID {best['mkv_id']}, {best['entries']} entries) is much "
                          f"smaller than '{dominant['language']}' track "
                          f"(ID {dominant['mkv_id']}, {dominant['entries']} entries).")
                    print(f"         Selecting larger track — it likely contains the "
                          f"full subtitles. Specify -l {dominant['language']} to silence this.")
                    return dominant

            return best
        print(f"  [warn] No track matching language '{lang_hint}'.")

    if len(tracks) == 1:
        return tracks[0]

    if batch:
        best = max(tracks, key=lambda t: t["entries"])
        print(f"  [warn] Multiple VobSub tracks — auto-selecting largest "
              f"(ID {best['mkv_id']}, lang={best['language']}, {best['entries']} entries).")
        print("         Use -l/--language to be explicit.")
        return best

    print("\n  VobSub subtitle tracks:")
    for i, t in enumerate(tracks):
        label = f"    [{i}]  Track ID {t['mkv_id']}  |  lang={t['language']}  |  {t['entries']} entries"
        if t["name"]:
            label += f"  |  \"{t['name']}\""
        print(label)
    while True:
        raw = input("  Select track [number]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(tracks):
            return tracks[int(raw)]
        print(f"  Enter a number between 0 and {len(tracks) - 1}.")
