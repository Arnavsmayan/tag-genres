# tag-genres

Auto-fill MP3 genre tags with emotion/vibe labels using Claude AI.

Scans a folder of MP3 files, detects the song's language (English, Hindi, or Korean) from the filename, and writes 2–3 mood/energy tags into the ID3 genre frame. Native music players (iOS Music, Samsung Music, etc.) treat each tag as a separate filterable genre — no special apps needed.

## How it works

1. Reads MP3 filenames and batches them (20 at a time).
2. Sends each batch to Claude, which parses artist/title/language and picks tags from a curated vibe palette.
3. Writes the tags into the MP3's ID3 TCON frame using mutagen.

## Vibe palette (examples)

| Language | Sample tags |
|----------|-------------|
| English  | Party Banger, Acoustic Feels, Chill 2010s, Taylor Swift Pop |
| Hindi    | Bollywood Romantic, Punjabi Energy, Emotional Arijit, Sufi Feel |
| Korean   | BTS Hype, K-Pop Upbeat, K-Drama OST |

## Setup

```bash
pip install mutagen anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
python tag_genres.py /path/to/your/music/folder
```

The script recursively finds all `.mp3` files, tags them, and prints results:

```
Found 47 MP3 files in /Music

Batch 1/3  (20 songs)...
  ✓  Shape of You                                         →  Upbeat Pop  /  Feel Good
  ✓  Tum Hi Ho                                            →  Bollywood Romantic  /  Emotional Arijit
  ...

Done!  47 tagged,  0 skipped.
```

## Requirements

- Python 3.10+
- [mutagen](https://mutagen.readthedocs.io/) — MP3 tag editing
- [anthropic](https://docs.anthropic.com/en/docs/sdks) — Claude API client
- An Anthropic API key

## License

MIT
