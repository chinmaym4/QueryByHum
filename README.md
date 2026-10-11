# earworm (QueryByHum)

Hum a melody, find the song — searching only your own music library (up to ~1,000 songs).
Design: [SPEC.md](SPEC.md) · project conventions: [CLAUDE.md](CLAUDE.md).

Current milestone: **M1** — matching engine + evaluation harness, CLI only.

## Setup

1. **ffmpeg** must be on your PATH (used to decode every input format):
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: `winget install ffmpeg`
   - Check: `ffmpeg -version`
2. **uv** (https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install: `uv sync` (creates `.venv` with Python 3.11, torch, torchcrepe, demucs, …)
4. Run tests: `uv run pytest`. The first run renders the synthetic library and runs CREPE
   over it (~5 min on CPU); the CREPE output is cached in `.pytest_cache`, so later runs take seconds.

On first ingest, demucs downloads the `htdemucs` weights (~80 MB) from `dl.fbaipublicfiles.com`.
CUDA or Apple MPS is used automatically when available (`[device]` in `config.toml`);
everything also runs on CPU, where CREPE uses the `tiny` model.

## Commands

```
uv run earworm ingest --folder ~/Music/some-folder        # local audio files
uv run earworm ingest --csv playlist.csv --source deezer  # 30 s Deezer previews
uv run earworm search hum.m4a [--artist Drake] [--k 5] [--matcher all|dtw|contour|rank]
uv run earworm explain hum.m4a SONG_ID [--out plot.png]   # alignment plot per matcher
uv run earworm eval [--weights-search]                    # real-hum evaluation report
uv run earworm stats [--list]                             # library size, sources, failures, song ids
```

- **Local folder:** title/artist come from tags (mutagen), else from an `Artist - Title.ext` filename.
- **Playlist CSV:** columns `title,artist[,added_at]`, or an [Exportify](https://exportify.net)
  export (`Track Name`, `Artist Name(s)`, `Added At`). Each row is looked up on Deezer; rows
  without a confident match (rapidfuzz ≥ 85) or without a preview are written to `data/unmatched.csv`.
- Ingestion is idempotent: songs whose features already exist are skipped unless you pass `--force`.
- Everything generated (audio cache, stems, features, `index.db`) lives in `data/` (gitignored).
- `search` prints rank, title, artist, fused score, the z-score from each matcher, and the
  offset in the song where the match starts.

CPU cost: ingestion runs demucs and then CREPE on two stems, which is slow on CPU (roughly
a minute or more per full-length song). Previews (30 s) are much cheaper.

## Recording and labeling hums for `earworm eval`

1. Record 10–20 s of humming per song with any recorder (phone voice memo is fine; m4a, mp3,
   wav, webm/opus all work). Use one clear melodic line ("da da" or "hmm"), no lyrics needed.
   Start right on the melody; leading/trailing silence is trimmed automatically.
2. Put the files in `eval/hums/` (audio files there are gitignored).
3. Add one row per hum to `eval/labels.csv`:

   ```
   hum_file,title,artist,part,in_preview,notes
   hum01.m4a,Hotline Bling,Drake,chorus,y,
   hum02.m4a,Blinding Lights,The Weeknd,hook_instrumental,unknown,synth riff
   ```

   - `title`, `artist`: as in your library (fuzzy matched, so small differences are fine)
   - `part`: `chorus | verse | hook_instrumental | rap | intro | other`
   - `in_preview`: `y | n | unknown`: whether the part you hummed falls inside the indexed
     audio (matters for Deezer's 30 s previews)
4. Run `uv run earworm eval`. It prints:
   - top-1/3/5 accuracy (with a 95% interval on top-5) and MRR for each matcher and the fused ranking;
   - a **leave-one-out** table: fused accuracy with each matcher dropped, plus how many hums got
     better or worse without it. Use the per-hum counts, not small accuracy differences, to judge
     whether a matcher earns its place;
   - the same metrics broken down by `part` and `in_preview`;
   - every hum's rank of the correct song under each matcher, and which library song its label
     matched (check for `AMBIGUOUS` warnings, e.g. a remix and the original both in the library);
   - excluded hums, by reason (song not in library / hum unreadable).

   The report is saved to `eval/results/<timestamp>.csv` (per hum) and `<timestamp>_metrics.csv`.
   Ties count against the correct song. `--weights-search` grid-searches the fusion weights; with
   a small eval set that overfits, so treat the result as a hint.

   Caveats at small scale: 30 hums give roughly +/-15 points of uncertainty on top-5; offsets are
   not evaluated (labels have no true start time); and an `in_preview` column that is mostly
   `unknown` makes that breakdown uninterpretable, so fill it in.

When a hum is ranked badly, run `earworm explain HUM SONG_ID` to see where each matcher aligned it.

## How matching works (short version)

Audio → 16 kHz mono → torchcrepe pitch (10 ms) → cleaned MIDI curve → notes. Songs are first
split by demucs into `vocals` and `other` stems, and each song scores as the max over its stems.
Three matchers score every song:

- **dtw:** subsequence DTW on the continuous pitch curve (key- and tempo-invariant)
- **contour:** alignment of quantized up/down/repeat intervals plus rhythm tokens
- **rank:** Kendall's tau between time-normalized pitch samples

Their scores are z-scored across the library and summed with the weights in `config.toml`.
All tunable parameters are in `config.toml`.
