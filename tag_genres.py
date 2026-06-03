#!/usr/bin/env python3
"""
tag_genres.py — Auto-fill MP3 genre tags with emotion/vibe labels using GPT.

Usage:
    python tag_genres.py            # tags MUSIC_FOLDER (hardcoded below)
    python tag_genres.py --dry-run  # preview without writing files

Requirements:
    pip install mutagen openai tqdm

API key and music folder are hardcoded below (local-only script).

Each song gets 2-3 emotion/vibe tags written as separate entries in the
ID3 TCON frame. Native phone music players (iOS Music, Samsung Music) read
each entry as a distinct genre — sort/filter by genre works out of the box.
"""

import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    from mutagen import MutagenError
    from mutagen.id3 import ID3, TCON, ID3NoHeaderError
except ImportError:
    print("Missing dependency. Run: pip install mutagen")
    sys.exit(1)

try:
    from openai import OpenAI, APIError, APITimeoutError, RateLimitError
except ImportError:
    print("Missing dependency. Run: pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependency. Run: pip install tqdm")
    sys.exit(1)


# ── API key + music folder (local-only script — hardcoded by design) ────────
# OPENAI_API_KEY is imported from a sibling `key.py` (gitignored).
# That file should contain a single line:  OPENAI_API_KEY = "sk-..."
try:
    from key import OPENAI_API_KEY
except ImportError:
    print("Missing key.py next to this script. Create it with:")
    print('    OPENAI_API_KEY = "sk-..."')
    sys.exit(1)

OPENAI_MODEL = "gpt-5.4-mini"
MUSIC_FOLDER = Path(r"C:\Users\arnav\Music\Music")


# ── Daily token budget ───────────────────────────────────────────────────────
# Account cap is 2M (input+output combined) per UTC day. Stay below it.
DAILY_TOKEN_LIMIT = 2_000_000
DAILY_SAFETY_MARGIN = 50_000          # leave headroom for other usage today
USAGE_FILE = Path.home() / ".tag_genres_usage.json"
ESTIMATED_TOKENS_PER_BATCH = 5_000    # conservative pre-flight estimate (used before any batch has run)


def load_today_usage() -> int:
    """Return tokens already spent today (UTC). 0 if file missing/stale/corrupt."""
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        with open(USAGE_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    if data.get("date") != today:
        return 0
    return int(data.get("tokens", 0))


def save_today_usage(tokens: int) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump({"date": today, "tokens": tokens}, f)
    except OSError as e:
        print(f"  ⚠  Could not persist token usage: {e}")


# ── Vibe palette ─────────────────────────────────────────────────────────────
# Compact, language-prefixed vibes. GPT picks 1-3 per song.
# Each tag becomes a separate genre entry in the MP3.
#
# English: 1 energy tag (party/feelgood/slow) + optional era tag (modern/oldschool)
#          + optional gym flag (energy-gym for workout/motivational tracks)
#          modern  = Bieber, Sheeran, Mendes, Taylor, Drake-era (≈2010+)
#          oldschool = BSB, MJ, Bryan Adams, 90s/early-2000s
# Hindi:   pick 1-3 tags freely from the Hindi list (no era axis)
#          + optional gym flag (energy-gym for workout/motivational tracks)
# Kpop:    1 energy tag + optional flag(s): bts, girlgroup, english
#          english = Korean artist singing fully in English (e.g. Jungkook "Yes or No")

VIBE_TAGS = {
    "English": [
        "english-party",
        "english-feelgood",
        "english-slow",
        "english-modern",       # era flag — combine with party/feelgood/slow
        "english-oldschool",    # era flag — combine with party/feelgood/slow
        "english-energy-gym",   # flag — workout / motivational (Remember the Name, Eye of the Tiger)
    ],

    "Hindi": [
        "hindi-party",             # regular Bollywood party
        "hindi-punjabi-party",     # Punjabi-flavored bangers
        "hindi-upbeat",            # non-party upbeat / feel-good high energy
        "hindi-roadtrip",          # mid-tempo journey vibe (Hum Jo Chalne Lage, Ve Haaniya)
        "hindi-rainy-acoustic",    # soft acoustic monsoon-mood (Iktara, Kabhi Kabhi Aditi)
        "hindi-slow-romantic",     # slow romantic ballads (Tum Hi Ho, Tum Se Hi)
        "hindi-heartbreak",        # sad / breakup / longing (Channa Mereya, Agar Tum Saath Ho)
        "hindi-energy-gym",        # flag — workout / motivational (Bhaag Milkha Bhaag, Sultan)
    ],

    "Korean": [
        "kpop-upbeat",      # high-energy K-pop
        "kpop-slow",        # ballad / slow K-pop
        "kpop-bts",         # flag — apply on top of upbeat/slow when it's BTS
        "kpop-girlgroup",   # flag — apply on top of upbeat/slow for girl groups
        "kpop-english",     # Korean artist singing fully in English
    ],
}

BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 1.5
REQUEST_TIMEOUT = 60.0          # seconds per OpenAI request
MAX_RETRIES = 3                 # attempts per batch
RETRY_BACKOFF_BASE = 2.0        # seconds; doubles each retry

# Flat set of every legal tag — used to validate model output.
ALL_VIBE_TAGS = frozenset(t for tags in VIBE_TAGS.values() for t in tags)


def read_tags(mp3_path: Path) -> dict | None:
    """Use the raw filename (no extension) as the song identifier.
    GPT will parse song name, artist, movie title etc. from whatever is there.

    Returns None to SKIP this file when:
      - the file is unreadable (resume-friendly: don't crash the run)
      - it already has a TCON frame populated entirely from our vibe palette
        (i.e. this script tagged it on a previous run)
    """
    try:
        try:
            existing = ID3(str(mp3_path))
        except ID3NoHeaderError:
            existing = None
    except (OSError, MutagenError) as e:
        print(f"  ⚠  Could not read {mp3_path.name}: {e}")
        return None

    if existing is not None and "TCON" in existing:
        current = [str(t) for t in existing["TCON"].text if str(t).strip()]
        if current and all(t in ALL_VIBE_TAGS for t in current):
            return None  # already tagged by us — resume

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
    except (OSError, MutagenError) as e:
        print(f"  ⚠  Could not write genres to {mp3_path.name}: {e}")


def build_prompt(songs: list[dict]) -> str:
    eng = "\n".join(f"    - {t}" for t in VIBE_TAGS["English"])
    hin = "\n".join(f"    - {t}" for t in VIBE_TAGS["Hindi"])
    kor = "\n".join(f"    - {t}" for t in VIBE_TAGS["Korean"])

    songs_str = "\n".join(
        f'{i+1}. {s["filename"]}'
        for i, s in enumerate(songs)
    )

    return f"""You are a music tagger for a multilingual library (English, Hindi, Korean).

For each song, assign 1 to 3 vibe tags that capture how the song FEELS
(emotion + energy), not where you'd play it.

Step 1: Parse the filename to figure out song name, artist(s), and movie/album if present.
         Then detect the language: English, Hindi, or Korean.
Step 2: Pick 1-3 tags strictly from the matching list below.

English tags:
{eng}

Hindi tags:
{hin}

Korean (kpop) tags:
{kor}

How to combine tags within each language:

ENGLISH — pick 1 energy tag + (optionally) 1 era tag + (optionally) the gym flag.
  Energy: english-party | english-feelgood | english-slow
  Era flag (optional): english-modern (Bieber, Sheeran, Mendes, Taylor, Drake-era ≈2010+)
                       english-oldschool (BSB, MJ, Bryan Adams, 90s/early-2000s)
  Gym flag (optional): english-energy-gym — workout/motivational (Remember the Name,
                       Eye of the Tiger, Stronger). Add on TOP of the energy tag.
  Examples:
    "I Want It That Way - Backstreet Boys" → ["english-slow", "english-oldschool"]
    "Shape of You - Ed Sheeran"           → ["english-feelgood", "english-modern"]
    "Macarena"                            → ["english-party", "english-oldschool"]
    "Blinding Lights - The Weeknd"        → ["english-feelgood", "english-modern"]
    "Remember the Name - Fort Minor"      → ["english-party", "english-energy-gym"]
    "Eye of the Tiger - Survivor"         → ["english-party", "english-oldschool", "english-energy-gym"]

HINDI — pick 1-3 tags freely from the Hindi list. No era axis.
  Hindi-slow-romantic = slow romantic ballads (Tum Hi Ho).
  Hindi-heartbreak    = sad / breakup / longing (Channa Mereya, Agar Tum Saath Ho).
                        Use heartbreak instead of slow-romantic when the song's
                        core emotion is sadness, not love.
  Hindi-roadtrip      = mid-tempo journey vibe (Hum Jo Chalne Lage, Ve Haaniya).
  Hindi-rainy-acoustic = soft acoustic monsoon mood (Iktara, Kabhi Kabhi Aditi).
  Hindi-upbeat        = high energy that ISN'T party (Ilahi, Nashe Si Chadh Gayi).
  Hindi-energy-gym    = workout/motivational (Bhaag Milkha Bhaag, Sultan, Zinda).
                        Add on TOP of the energy tag (usually with hindi-upbeat).
  Examples:
    "Tum Hi Ho - Arijit Singh"        → ["hindi-slow-romantic"]
    "Channa Mereya - Arijit Singh"    → ["hindi-heartbreak"]
    "Lamberghini - The Doorbeen"      → ["hindi-punjabi-party"]
    "Ilahi - Yeh Jawaani Hai Deewani" → ["hindi-upbeat", "hindi-roadtrip"]
    "Iktara - Wake Up Sid"            → ["hindi-rainy-acoustic", "hindi-slow-romantic"]
    "Bhaag Milkha Bhaag - title"      → ["hindi-upbeat", "hindi-energy-gym"]

KPOP — pick 1 energy tag + optional flag(s) (bts, girlgroup, english).
  Energy: kpop-upbeat | kpop-slow
  Flags (optional, can combine):
    kpop-bts       — song is by BTS or a BTS member
    kpop-girlgroup — song is by a K-pop girl group (BLACKPINK, NewJeans, ITZY, etc.)
    kpop-english   — Korean artist singing fully in English
  Examples:
    "Dynamite - BTS"           → ["kpop-upbeat", "kpop-bts", "kpop-english"]
    "Spring Day - BTS"         → ["kpop-slow", "kpop-bts"]
    "How You Like That - BLACKPINK" → ["kpop-upbeat", "kpop-girlgroup"]
    "Yes or No - Jungkook"     → ["kpop-upbeat", "kpop-bts", "kpop-english"]

Rules:
- 1 to 3 tags per song. Never more than 3.
- Only use tags from the correct language list. No inventing new tags.
- Tag names must match EXACTLY (lowercase, hyphenated, as listed).
- Return ONLY a JSON array where each element is an array of 1-3 tag strings.
  Same order as input songs. No explanation, no markdown.

Song filenames (raw — parse them yourself):
{songs_str}

Example output for 5 filenames (English, Hindi, Korean, Hindi, English):
[
  ["english-feelgood", "english-modern"],
  ["hindi-heartbreak"],
  ["kpop-upbeat", "kpop-bts"],
  ["hindi-punjabi-party"],
  ["english-slow", "english-oldschool"]
]
"""


def _validate_tags(entry, lineno: int) -> list[str]:
    """Coerce one model output into a clean list of 1-3 known tags.
    Drops anything not in ALL_VIBE_TAGS (no hallucinated genres written to disk)."""
    if isinstance(entry, str):
        candidates = [entry]
    elif isinstance(entry, list):
        candidates = [str(t).strip() for t in entry]
    else:
        candidates = []

    valid = [t for t in candidates if t in ALL_VIBE_TAGS]
    # de-dupe while keeping order, cap at 3
    seen = set()
    deduped = []
    for t in valid:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    deduped = deduped[:3]

    if not deduped:
        dropped = [t for t in candidates if t not in ALL_VIBE_TAGS]
        if dropped:
            print(f"  ⚠  song #{lineno}: dropped invalid tags {dropped}")
    return deduped


def get_genres_from_gpt(client: OpenAI, songs: list[dict]) -> tuple[list[list[str]], int]:
    """Call the model with timeout + retry.
    Returns (tags-per-song, tokens-used-this-call).
    Raises RuntimeError on permanent failure."""
    prompt = build_prompt(songs)

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_completion_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
                timeout=REQUEST_TIMEOUT,
            )
            if not response.choices or response.choices[0].message.content is None:
                raise RuntimeError("empty response from model")
            raw = response.choices[0].message.content.strip()

            tokens_used = response.usage.total_tokens if response.usage else 0

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                result = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"model returned non-JSON: {e}; first 200 chars: {raw[:200]!r}")

            if not isinstance(result, list) or len(result) != len(songs):
                raise RuntimeError(f"got {len(result) if isinstance(result, list) else 'non-list'} entries for {len(songs)} songs")

            return [_validate_tags(entry, i + 1) for i, entry in enumerate(result)], tokens_used

        except (APITimeoutError, RateLimitError, APIError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE ** (attempt - 1)
                print(f"  … API error (attempt {attempt}/{MAX_RETRIES}): {e}; retrying in {wait:.1f}s")
                time.sleep(wait)
            continue
        except RuntimeError:
            # Parsing/shape problems aren't worth retrying — model gave a bad response.
            raise

    raise RuntimeError(f"OpenAI failed after {MAX_RETRIES} attempts: {last_err}")


def process_folder(folder: Path, dry_run: bool = False):
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        print("Error: set OPENAI_API_KEY in key.py.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    mp3_files = sorted(folder.rglob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {folder}")
        sys.exit(1)

    print(f"Found {len(mp3_files)} MP3 files in {folder}")
    if dry_run:
        print("DRY RUN — no files will be modified.\n")
    else:
        print()

    # read_tags returns None for files we should skip (already-tagged or unreadable),
    # so the walrus filter actually filters now.
    songs = [info for f in mp3_files if (info := read_tags(f))]
    already_tagged = len(mp3_files) - len(songs)
    if already_tagged:
        print(f"Skipping {already_tagged} file(s) already tagged or unreadable.\n")

    total     = len(songs)
    processed = 0
    skipped   = 0

    # Daily token budget tracking
    tokens_today_start = load_today_usage()
    tokens_today = tokens_today_start
    tokens_this_run = 0
    budget_cap = DAILY_TOKEN_LIMIT - DAILY_SAFETY_MARGIN

    print(f"Daily token usage so far today (UTC): {tokens_today:,} / {DAILY_TOKEN_LIMIT:,}\n")
    if tokens_today >= budget_cap:
        print(f"Already at/near daily limit ({tokens_today:,} ≥ {budget_cap:,}). Try again after UTC midnight.")
        return

    pbar = tqdm(total=total, desc="Tagging", unit="song", dynamic_ncols=True)
    try:
        for batch_start in range(0, total, BATCH_SIZE):
            batch         = songs[batch_start : batch_start + BATCH_SIZE]
            batch_num     = batch_start // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

            # Pre-flight budget check: use observed average if we have data, else a safe estimate.
            batches_done = batch_num - 1
            avg_per_batch = (tokens_this_run / batches_done) if batches_done else ESTIMATED_TOKENS_PER_BATCH
            if tokens_today + avg_per_batch > budget_cap:
                remaining_songs = total - batch_start
                tqdm.write(
                    f"Stopping: next batch (~{int(avg_per_batch):,} tokens) would exceed "
                    f"daily limit ({tokens_today:,}/{budget_cap:,}). "
                    f"{remaining_songs} song(s) deferred to next run."
                )
                skipped += remaining_songs
                break

            tqdm.write(f"Batch {batch_num}/{total_batches}  ({len(batch)} songs)...")

            try:
                all_genres, batch_tokens = get_genres_from_gpt(client, batch)
            except (APIError, RuntimeError) as e:
                tqdm.write(f"  ✗ batch {batch_num} failed: {e}")
                skipped += len(batch)
                pbar.update(len(batch))
                continue

            tokens_today += batch_tokens
            tokens_this_run += batch_tokens
            # Always persist — dry-run still spends real tokens against the daily cap.
            save_today_usage(tokens_today)

            for song, genres in zip(batch, all_genres):
                if not genres:
                    tqdm.write(f"  ✗  {song['filename'][:55]:<55}  →  no valid tags returned, skipped")
                    skipped += 1
                    pbar.update(1)
                    continue
                if not dry_run:
                    write_genres(song["path"], genres)
                tag_str = "  /  ".join(genres)
                marker = "·" if dry_run else "✓"
                tqdm.write(f"  {marker}  {song['filename'][:55]:<55}  →  {tag_str}")
                processed += 1
                pbar.update(1)

            tqdm.write(f"  ↳ batch tokens: {batch_tokens:,}  |  today: {tokens_today:,}/{DAILY_TOKEN_LIMIT:,}")

            if batch_start + BATCH_SIZE < total:
                time.sleep(DELAY_BETWEEN_BATCHES)
    finally:
        pbar.close()

    verb = "would tag" if dry_run else "tagged"
    print(
        f"\nDone!  {processed} {verb},  {skipped} skipped.  "
        f"Tokens this run: {tokens_this_run:,}.  Today total: {tokens_today:,}/{DAILY_TOKEN_LIMIT:,}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-tag MP3 files with vibe/emotion genres via GPT.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be tagged without modifying any files.")
    args = parser.parse_args()

    if not MUSIC_FOLDER.is_dir():
        print(f"Not a directory: {MUSIC_FOLDER}")
        sys.exit(1)

    process_folder(MUSIC_FOLDER, dry_run=args.dry_run)
