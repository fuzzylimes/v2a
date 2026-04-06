"""VobSub track identification and selection."""

import json
import subprocess
from pathlib import Path


def identify_vobsub_tracks(mkv_path: Path) -> list[dict]:
    """
    Return a list of VobSub tracks found in mkv_path, each as a dict with:
        mkv_id   : track ID used by mkvextract
        language : BCP-47 language code (e.g. 'eng', 'und')
        name     : human-readable track name (may be empty)
    """
    result = subprocess.run(
        ["mkvmerge", "--identify", "--identification-format", "json", str(mkv_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    tracks = []
    for track in data.get("tracks", []):
        if track.get("type") == "subtitles" and track.get("codec") == "VobSub":
            props = track.get("properties", {})
            tracks.append({
                "mkv_id":   track["id"],
                "language": props.get("language", "und"),
                "name":     props.get("track_name", ""),
            })
    return tracks


def select_track(tracks: list[dict], lang_hint: str | None, batch: bool) -> dict | None:
    """
    Pick a VobSub track.

    Priority: language match > only-one-track > batch auto-first > interactive prompt.

    When multiple tracks share the same language code the LAST one is returned.
    This follows a common disc-authoring convention where the first same-language
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
