#!/usr/bin/env python3
"""
tag_genres.py — Auto-fill MP3 genre tags with emotion/vibe labels using Claude.

Usage:
    python tag_genres.py /path/to/your/music/folder

Requirements:
    pip install mutagen anthropic

Set your Anthropic API key:
    export ANTHROPIC_API_KEY=sk-ant-...

Each song gets 2-3 emotion/vibe tags written as separate entries in the
ID3 TCON frame. Native phone music players (iOS Music, Samsung Music) read
each entry as a distinct genre — sort/filter by genre works out of the box.
"""

import os
import sys
import json
import time
from pathlib import Path

try:
    from mutagen.id3 import ID3, TCON, ID3NoHeaderError
except ImportError:
    print("Missing dependency. Run: pip install mutagen")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run: pip install anthropic")
    sys.exit(1)


# ── Vibe palette ─────────────────────────────────────────────────────────────
# Pure emotion/energy — how a song FEELS, not where you'd play it.
# Claude picks 2-3 per song. Each tag becomes a separate genre in the MP3.
#
# Energy spectrum so similar vibes cluster when sorted alphabetically:
#   Acoustic Feels · Chill · Mellow  →  Feel Good · Upbeat  →  Hype · Party Banger
#
# ALL tags are in English regardless of song language (so Korean
# songs show English tags — easier to sort on any phone).

VIBE_TAGS = {
    "English": [
        # High energy
        "Party Banger",
        "Hype Track",
        "Dance Hit",
        "Summer Energy",
        "Upbeat Pop",
        # Mid / positive
        "Feel Good",
        "Carefree Bop",
        "Confident Pop",
        "Indie Vibes",
        # Romantic / emotional
        "Romantic",
        "Soft Love Song",
        "Heartbreak",
        "Emotional",
        # Chill / slow
        "Chill Pop",
        "Mellow",
        "Late Night Chill",
        "Acoustic Feels",
        # Era (use when the era IS the identity of the song)
        "2000s Throwback",
        "Chill 2010s",
        "2020s Pop",
        "90s Oldie",           # only for actual 90s songs
        # Artist-era vibes
        "Backstreet Era",      # BSB / early 2000s boyband
        "Bieber Era",
        "Shawn Mendes Soft",
        "Ed Sheeran Acoustic",
        "Taylor Swift Pop",
        "R&B Smooth",
        "Hip-Hop Energy",
    ],

    "Hindi": [
        # High energy
        "Party Banger",
        "Desi Hype",
        "Punjabi Energy",
        "Item Track",
        "Sangeet Hit",
        # Romantic / emotional
        "Bollywood Romantic",
        "Heartbreak Hindi",
        "Sufi Feel",
        "Emotional Arijit",    # slow emotional Bollywood / Arijit sound
        # Chill / nostalgic
        "Chill Bollywood",
        "Late Night Hindi",
        "Filmi Nostalgia",
        # Era
        "2000s Bollywood",
        "2010s Bollywood",
        "2020s Hindi Pop",
        "90s Bollywood",       # only for actual 90s songs
        # Artist-era
        "AR Rahman Classic",
        "Pritam Feels",
        "Badshah Energy",
        "Atif Aslam Soft",
    ],

    "Korean": [
        # Intentionally small — library is 90% BTS
        # BTS specific
        "BTS Hype",
        "BTS Upbeat",
        "BTS Slow",
        "BTS Emotional",
        # Generic K-Pop for non-BTS
        "K-Pop Upbeat",
        "K-Pop Slow",
        "K-Drama OST",
        "K-Pop Girl Group",
        # Korean artist singing fully in English (e.g. Jungkook - Yes or No)
        "K-Artist English",
    ],
}

BATCH_SIZE = 20
DELAY_BETWEEN_BATCHES = 1.5


def read_tags(mp3_path: Path) -> dict | None:
    """Use the raw filename (no extension) as the song identifier.
    Claude will parse song name, artist, movie title etc. from whatever is there."""
    return {"path": mp3_path, "filename": mp3_path.stem}


def write_genres(mp3_path: Path, genres: list[str]):
    """
    Write multiple genre strings into the ID3 TCON frame as a proper list.
    Each item in text= becomes a separate genre entry — iOS Music and
    Samsung Music both expose these as individual filterable genres.
    """
    try:
        try:
            tags = ID3(str(mp3_path))
        except ID3NoHeaderError:
            tags = ID3()
        tags["TCON"] = TCON(encoding=3, text=genres)
        tags.save(str(mp3_path))
    except Exception as e:
        print(f"  ⚠  Could not write genres to {mp3_path.name}: {e}")


def build_prompt(songs: list[dict]) -> str:
    eng  = "\n".join(f"    - {t}" for t in VIBE_TAGS["English"])
    hin  = "\n".join(f"    - {t}" for t in VIBE_TAGS["Hindi"])
    kor  = "\n".join(f"    - {t}" for t in VIBE_TAGS["Korean"])

    songs_str = "\n".join(
        f'{i+1}. {s["filename"]}'
        for i, s in enumerate(songs)
    )

    return f"""You are a music tagger for a multilingual library (English, Hindi, Korean).

For each song assign exactly 2 or 3 vibe tags that capture the song's EMOTION and ENERGY.
Focus on how the song feels — not where you'd play it.

Step 1: Parse the filename to figure out song name, artist(s), and movie/album if present.
         Then detect the language — English, Hindi, or Korean.
Step 2: Pick 2-3 tags strictly from the matching list below.

English tags:
{eng}

Hindi tags:
{hin}

Korean tags:
{kor}

Rules:
- Exactly 2 or 3 tags. Never 1, never 4.
- Only use tags from the correct language list. No inventing new tags.
- "90s Oldie" / "90s Bollywood" only for songs genuinely from the 1990s.
- Artist-era tags only if the song truly fits that artist's sound.
- High-energy songs → energy tags. Slow/sad songs → emotional/chill tags.
- "Party Banger" appears in both English and Hindi lists — fine to use for either.
- Return ONLY a JSON array where each element is an array of 2-3 tag strings.
  Same order as input songs. No explanation, no markdown.

Song filenames (raw — parse them yourself):
{songs_str}

Example output for 5 filenames (English, Hindi, Korean, Hindi, English):
[
  ["Upbeat Pop", "Feel Good", "Chill 2010s"],
  ["Heartbreak Hindi", "Emotional Arijit", "Late Night Hindi"],
  ["BTS Hype", "BTS Upbeat"],
  ["Party Banger", "Punjabi Energy", "Sangeet Hit"],
  ["Acoustic Feels", "Romantic", "Shawn Mendes Soft"]
]
"""


def get_genres_from_claude(client: anthropic.Anthropic, songs: list[dict]) -> list[list[str]]:
    prompt = build_prompt(songs)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    if len(result) != len(songs):
        raise ValueError(f"Got {len(result)} entries for {len(songs)} songs")

    cleaned = []
    for entry in result:
        if isinstance(entry, str):
            cleaned.append([entry])
        elif isinstance(entry, list):
            cleaned.append([str(t) for t in entry[:3]])  # cap at 3
        else:
            cleaned.append(["Unknown"])
    return cleaned


def process_folder(folder: Path):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    mp3_files = sorted(folder.rglob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {folder}")
        sys.exit(1)

    print(f"Found {len(mp3_files)} MP3 files in {folder}\n")

    songs = [info for f in mp3_files if (info := read_tags(f))]

    total     = len(songs)
    processed = 0
    skipped   = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch         = songs[batch_start : batch_start + BATCH_SIZE]
        batch_num     = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Batch {batch_num}/{total_batches}  ({len(batch)} songs)...")

        try:
            all_genres = get_genres_from_claude(client, batch)
        except Exception as e:
            print(f"  ✗ API error on batch {batch_num}: {e}")
            skipped += len(batch)
            continue

        for song, genres in zip(batch, all_genres):
            write_genres(song["path"], genres)
            tag_str = "  /  ".join(genres)
            print(f"  ✓  {song['filename'][:55]:<55}  →  {tag_str}")
            processed += 1

        if batch_start + BATCH_SIZE < total:
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\nDone!  {processed} tagged,  {skipped} skipped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tag_genres.py /path/to/music/folder")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    process_folder(folder)
