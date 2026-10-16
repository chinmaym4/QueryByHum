# earworm (QueryByHum)

Hum a melody, find the song — searching only your own music library (up to ~1,000 songs).
Design: [SPEC.md](SPEC.md) · project conventions: [CLAUDE.md](CLAUDE.md).

Current milestone: **M1** — matching engine + evaluation harness, CLI only (no web app yet).

**Status:** the pipeline, three matchers, fusion, ingestion (local folder and Deezer previews),
eval harness and CLI are implemented, and the synthetic test suite passes (fused top-1 100% on
synthetic queries, every matcher at least 91% top-3). **Nothing has been measured on real hums
yet**: that is the next step, and the workflow is below.

## Architecture

```
ingest:  folder / CSV+Deezer ─► demucs (vocals, other) ─► torchcrepe pitch ─► notes
                                      └──────────── data/index.db + data/features/*.npz
search:  hum ─► ffmpeg ─► pitch ─► notes ─► dtw ┐
                                           contour ├─► z-score fusion ─► top-K + offset
                                           rank ┘      (offset from DTW)
```

Code lives in `earworm/` (`audio`, `separate`, `pitch`, `notes`, `features`, `library`, `ingest`,
`sources/`, `matchers/`, `fusion`, `search`, `evaluation`, `plots`, `cli`). All tunable values
are in `config.toml`; [SPEC.md](SPEC.md) has the full design, the measured results and the
known limitations.

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
uv run earworm check-hum hum.m4a [more files]                 # preflight a recording (advisory)
```

- **Local folder:** title/artist come from tags (mutagen), else from an `Artist - Title.ext` filename.
- **Playlist CSV:** columns `title,artist[,added_at]`, or an [Exportify](https://exportify.net)
  export (`Track Name`, `Artist Name(s)`, `Added At`; the first listed artist is used). Each row
  is looked up on Deezer; rows without a confident match (rapidfuzz ≥ 85) or without a preview
  are written to `data/unmatched.csv`. Deezer's structured `artist:"…" track:"…"` search currently
  returns nothing, so the client falls back to a plain `artist title` query, which works. Previews
  are 30 s MP3s on expiring signed URLs, so they are downloaded during ingest.
- After a CSV ingest, run `uv run earworm stats` and check `data/unmatched.csv` to see which
  songs are missing from the library (and so excluded from eval).
- Ingestion is idempotent: songs whose features already exist are skipped unless you pass `--force`.
- Everything generated (audio cache, stems, features, `index.db`) lives in `data/` (gitignored).
- `search` prints rank, title, artist, fused score, the z-score from each matcher, and the
  offset in the song where the match starts.

Cost (measured on a 4-core CPU container, no GPU):

- **Ingest:** about 20-25 s per 30 s preview (demucs 13-18 s, pitch and notes for both stems
  7 s), so roughly 60-75 minutes for 175 songs. Full-length songs are much slower on CPU.
- **Search:** about 1.1 s to match one hum against 175 previews, about 4 s against 1,000
  previews, about 20 s against 1,000 full-length songs, plus about 4 s of pitch extraction on the
  hum itself.

## Recording and labeling hums for `earworm eval`

1. Record 10–20 s of humming per song with any recorder (phone voice memo is fine; m4a, mp3,
   wav, webm/opus all work). Use one clear melodic line ("da da" or "hmm"), no lyrics needed.
   Start right on the melody; leading/trailing silence is trimmed automatically.
2. Check each take before labeling it: `uv run earworm check-hum eval/hums/*.m4a` reports level,
   clipping, voiced audio, note count and pitch range, and warns about takes that are too short,
   too quiet, clipped, mostly unvoiced or not melodic. The warnings are advisory (thresholds in
   `config.toml` `[checkhum]`) and exist so a bad recording is not mistaken for a matcher failure.
   Humming "da da" or "ta ta" gives clearer note starts than "mmm".
3. Put the files in `eval/hums/` (audio files there are gitignored).
4. Add one row per hum to `eval/labels.csv`:

   ```
   hum_file,title,artist,part,in_preview,notes,true_offset_s
   hum01.m4a,Hotline Bling,Drake,chorus,y,,12.5
   hum02.m4a,Blinding Lights,The Weeknd,hook_instrumental,unknown,synth riff,
   ```

   - `title`, `artist`: as in your library (fuzzy matched, so small differences are fine)
   - `part`: `chorus | verse | hook_instrumental | rap | intro | other`
   - `in_preview`: `y | n | unknown`: whether the part you hummed falls inside the indexed
     audio (matters for Deezer's 30 s previews)
   - `true_offset_s` (optional): where the hummed part starts, **in seconds into the indexed
     audio**, i.e. into the 30 s preview file in `data/audio/<song_id>.mp3` for Deezer songs, not
     into the full track (a preview is an excerpt from an unknown point of the song, so a
     full-track timestamp would be meaningless). Leave it blank for hums with `in_preview=n`.
     A non-numeric or negative value is an error, not silently ignored.
5. Run `uv run earworm eval`. It prints:
   - top-1/3/5 accuracy (with a 95% interval on top-5) and MRR for each matcher and the fused ranking;
   - a **leave-one-out** table: fused accuracy with each matcher dropped, plus how many hums got
     better or worse without it. Use the per-hum counts, not small accuracy differences, to judge
     whether a matcher earns its place;
   - the same metrics broken down by `part` and `in_preview`;
   - **offset accuracy** for hums that have `true_offset_s`: the fraction within
     `[eval] offset_tolerance_s` (1 s) of the truth and the median error, per matcher and fused,
     for all such hums and for those where fused ranked the song first;
   - every hum's offset under each matcher and the fused offset, with the **maximum spread**
     between the three matchers (a large spread means they disagree about where the hum matches);
   - that spread summarised separately for hums where fused ranked the song first and for all
     other hums (count, median, max, and how many exceed `offset_tolerance_s`);
   - every hum's rank of the correct song under each matcher, and which library song its label
     matched (check for `AMBIGUOUS` warnings, e.g. a remix and the original both in the library);
   - excluded hums, by reason (song not in library / hum unreadable).

   The report is saved to `eval/results/<timestamp>.csv` (per hum) and `<timestamp>_metrics.csv`.
   Ties count against the correct song. `--weights-search` grid-searches the fusion weights; with
   a small eval set that overfits, so treat the result as a hint.

   Caveats at small scale: 30 hums give roughly +/-15 points of uncertainty on top-5; offsets are
   not evaluated (labels have no true start time); and an `in_preview` column that is mostly
   `unknown` makes that breakdown uninterpretable, so fill it in.

To choose which songs to hum, `uv run python eval/pick_hums.py` draws a seeded random sample
(40 songs, seed 42 by default) from `eval/playlist.csv`, writes `eval/hum_sample.csv` and prints the
list with the seed and the playlist checksum. The draw never looks at what ingested or matched.

When a hum is ranked badly, run `earworm explain HUM SONG_ID` to see where each matcher aligned it.

## How matching works (short version)

Audio → 16 kHz mono → torchcrepe pitch (10 ms) → cleaned MIDI curve → notes. Songs are first
split by demucs into `vocals` and `other` stems, and each song scores as the max over its stems.
Three matchers score every song:

- **dtw:** subsequence DTW on the continuous pitch curve, tolerant of key (whole-semitone
  search, then a fractional key correction from the DTW path) and tempo (0.5-2x). Libraries of
  up to 300 songs (`fine_all_max_songs`) get the full-resolution pass on every song.
- **contour:** alignment of quantized up/down/repeat intervals plus rhythm tokens
- **rank:** Kendall's tau between time-normalized pitch samples (15 tempo scales, near-equal
  pitches counted as ties, best windows re-scanned at a fine start step)

Their scores are z-scored across the library and summed with the weights in `config.toml`.
The match offset shown by `search` comes from DTW. All tunable parameters are in `config.toml`.

## Tests

`uv run pytest` runs unit tests plus the synthetic accuracy suites (random melodies rendered to
audio and degraded with transposition, tempo change, wrong notes and note splits; a second set uses
continuous transpositions and per-note intonation error, because whole-semitone transpositions
cannot expose key-estimation errors). The accuracy thresholds come from SPEC.md and are never
loosened to make a run pass.
