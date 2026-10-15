# Earworm MVP Spec

Identify a song from a user's hum, searching only the user's own library (up to ~1,000 songs).

## Goal of this MVP

Answer one question with data: **can hummed melodies be matched to songs in a personal library accurately enough to be useful (correct song in top 5)?**

Build in two milestones. Finish and verify Milestone 1 before starting Milestone 2.

- **M1: Matching engine + evaluation harness (CLI only).** Three melody matchers, score fusion, synthetic tests, real-hum eval report.
- **M2: Minimal web app.** Record a hum in the browser, see top 5 results with playback at the matched timestamp.

## Non-goals (do not build)

Spotify/Apple OAuth, LLM voice agent, lyrics matching, user accounts, deployment, mobile app. Keep interfaces open for them (see "Future hooks").

## Stack

- Python 3.11, managed with `uv` and `pyproject.toml`
- `torchcrepe` (pitch tracking), `demucs` (source separation, model `htdemucs`)
- `librosa`, `soundfile`, `numpy`, `scipy`, `numba` (DTW and alignment loops must be numba-jitted)
- `mutagen` (audio tags), `rapidfuzz` (fuzzy artist/title matching), `requests`
- `fastapi`, `uvicorn` (M2)
- `matplotlib` (debug plots)
- `ffmpeg` must be on PATH; use it to decode any input format (mp3, m4a, webm/opus) to 16 kHz mono float32
- `pytest`
- Detect CUDA/MPS and use it when available; everything must still run on CPU

## Project layout

```
earworm/
  pyproject.toml
  config.toml                # all tunable parameters live here
  earworm/
    config.py
    audio.py                 # ffmpeg decode, resample, normalize
    separate.py              # demucs wrapper, stem caching
    pitch.py                 # torchcrepe -> cleaned MIDI pitch curve
    notes.py                 # note segmentation from pitch curve
    features.py              # Features bundle (pitch_10ms, pitch_50ms, notes); audio -> Features
    library.py               # Song model, index build/load (SQLite + .npz features)
    ingest.py                # per-song pipeline: separate -> features -> index
    sources/
      local.py               # ingest a folder of audio files
      playlist_csv.py        # ingest a CSV of title,artist[,added_at]
      deezer.py              # fetch 30s preview MP3s by title+artist
    matchers/
      base.py                # Matcher protocol
      dtw.py                 # Matcher A
      contour.py             # Matcher B
      rank.py                # Matcher C
    fusion.py
    search.py                # query -> ranked results
    evaluation.py            # eval metrics, leave-one-out, label matching (run_eval.py wraps it)
    plots.py                 # explain plots
    cli.py
    api.py                   # M2
  web/index.html             # M2
  eval/
    hums/                    # user-recorded hum files
    labels.csv
    playlist.csv             # optional: your playlist (title,artist[,added_at] or Exportify)
    results/                 # gitignored: per-run reports
    run_eval.py
  tests/
  data/                      # gitignored: audio cache, stems, features, index.db
```

## Data model

```
Song:
  id: str            # stable slug from artist + title
  title: str
  artist: str
  added_at: str|None # ISO date, if known
  audio_path: str
  duration_s: float
  source: "local" | "deezer_preview"

Features (per song, per stem, stored in data/features/<id>__<stem>.npz):
  pitch_10ms: float32[]   # MIDI semitones, NaN where unvoiced, 10 ms hop
  pitch_50ms: float32[]   # downsampled (nanmedian over 5 frames)
  notes: float32[n, 3]    # onset_s, duration_s, midi_pitch
```

Index metadata in SQLite (`data/index.db`). Ingestion is idempotent: skip songs whose features already exist unless `--force`.

## Ingestion

Three sources:

1. **Local folder:** `earworm ingest --folder PATH`. Read title/artist from tags via mutagen, fall back to parsing `Artist - Title` from the filename.
2. **Playlist CSV:** `earworm ingest --csv playlist.csv --source deezer`. CSV columns: `title,artist[,added_at]`. Accept the column names produced by Exportify (Spotify playlist exporter) as well: map `Track Name`, `Artist Name(s)`, `Added At`.
3. **Deezer previews:** for each CSV row, query `https://api.deezer.com/search?q=artist:"{artist}" track:"{title}"`, pick the best result by rapidfuzz score on title+artist (threshold 85), download the 30 s `preview` MP3. Verify the endpoint works at implementation time; if it fails, log clearly and skip the song. Rate limit to ~5 requests/s. Log unmatched rows to `data/unmatched.csv`.
   - **Verified behavior (Oct 2026):** the search and preview download work without authentication, but the structured `artist:"X" track:"Y"` query returns zero results for every song tried (HTTP 200, `total: 0`). The client therefore tries it first and falls back to the plain query `"{artist} {title}"`, which matched all test songs at score 100. Only the first listed artist is used when a row has several. Previews are 30 s, 128 kbps stereo MP3 (about 480 kB) on signed URLs that expire, so they are downloaded at ingest time and never re-fetched from a stored URL.

Per song:

1. Decode to 44.1 kHz stereo for demucs.
2. Run demucs `htdemucs`, keep `vocals` and `other` stems. Cache stems as 16 kHz mono FLAC in `data/stems/`.
3. Extract features (below) for each stem.

Measured on a 4-core CPU container (`tiny` CREPE): about 13-18 s demucs plus 7 s pitch/notes per 30 s preview, i.e. 20-25 s per song (about 60-75 min for 175 previews).

The `other` stem catches instrumental hooks (synth lines, riffs). Search scores a song as the max over its stems. Make stem list configurable.

## Pitch extraction (shared by songs and queries)

1. Decode to 16 kHz mono.
2. `torchcrepe` with model `full` on GPU, `tiny` on CPU (configurable), hop 10 ms, fmin 65 Hz, fmax 1000 Hz.
3. Voicing: frames with periodicity < 0.5 become NaN (configurable).
4. Convert Hz to MIDI: `69 + 12*log2(f/440)`.
5. Median filter, width 5 frames, on voiced runs only.
6. Octave error fix: if a voiced run jumps by 11 to 13 semitones for under 150 ms and returns, fold it back by 12.
7. Drop voiced runs shorter than 50 ms.
8. Produce `pitch_50ms` by nanmedian over blocks of 5 frames.

Queries skip demucs (hums are already monophonic). Trim leading/trailing silence.

## Note segmentation (`notes.py`)

From `pitch_10ms`:

- A new note starts after an unvoiced gap > 60 ms, or when pitch moves > 0.7 semitones from the running note median and stays there for >= 40 ms.
- Merge notes shorter than 80 ms into the neighbor with the closest pitch.
- Note pitch = median of its frames.

## Matchers

All matchers implement:

```python
class Matcher(Protocol):
    name: str
    def score(self, query: Features, song: Features) -> MatchResult: ...

@dataclass
class MatchResult:
    score: float        # higher = better
    offset_s: float     # where in the song the best match starts
    detail: dict        # matcher-specific extras (transposition, end time, ...) used by `explain`
```

A matcher may additionally provide `score_many(query, songs)` for a faster batch search over the whole library (DTW does; others are looped).

### Matcher A: Pitch DTW (`dtw.py`)

Frame-level, continuous pitch. Most information, least robust to bad singing.

- Inputs: `pitch_50ms` of query and song with unvoiced frames removed. Keep an index map back to original time so `offset_s` is correct.
- Key invariance: center the query on its median. For the song, try transpositions so the query median aligns with song median + t, for t in -12..+12 semitones. Keep the best.
- Subsequence DTW: query must be fully consumed, song start and end are free.
- Step pattern: (1,1), (1,2), (2,1) to bound tempo ratio between 0.5x and 2x.
- Local cost: `min(|q - s|, 2.0)` (clipped so single wrong notes don't dominate).
- Score: `-(accumulated cost / query length)`.
- Performance: implement in numba. Target: under 10 s for a 1,000-song library on a laptop CPU, under 2 s for 100 songs.
- Search is three passes:
  1. Coarse: every song at 100 ms hop with transpositions in steps of 2.
  2. Full resolution (50 ms, integer transpositions -12..+12): the top 50 songs by coarse score, or every song when the library has at most `fine_all_max_songs` (default 300) songs, so that DTW scores are never a mix of coarse and fine.
  3. Fractional key correction (`key_refine`), on the same songs as pass 2: set the shift to `t + median(song - shifted hum)` over the aligned frame pairs of the best path, rescore, and keep the better of the two. Hums are never on the song's semitone grid, so integer shifts alone leave up to 0.5 st of error on every frame (measured: self-excerpt cost 0.35 -> 0.14 on one song, and a mean gap to a brute-force 0.05-semitone search of 0.039 -> 0.001).

### Matcher B: Quantized contour + rhythm (`contour.py`)

Note-level, symbolic. Robust to wrong interval sizes.

- For consecutive notes compute interval `d = p[i] - p[i-1]` and quantize to 5 classes:
  - `|d| < 0.5`: R (repeat)
  - `0.5 <= |d| < 2.5`: u / d (small up/down)
  - `|d| >= 2.5`: U / D (large up/down)
- Rhythm: `r = log2(dur[i] / dur[i-1])`, quantize to 3 classes: shorter (< -0.4), same, longer (> 0.4).
- Token = (interval_class, rhythm_class).
- Semi-global alignment (query fully aligned, song free ends), numba.
  - Substitution cost: interval part 0 if equal, 0.5 if adjacent classes (U/u, u/R, R/d, d/D), 1 otherwise; rhythm part 0 if equal, 0.3 otherwise.
  - Insertion/deletion cost: 1.0 (absorbs note split/merge errors from segmentation).
- Score: `-(alignment cost / query token count)`.

### Matcher C: Rank contour (`rank.py`)

Compares the ordering of note heights, not their sizes. A single wrong note only slightly disturbs the ranking.

- Let `D` = query voiced duration. For each song window of duration `D * s`, s in {0.70, 0.75, ..., 1.40} (15 scales, step 0.05), hop 0.25 s:
  - Sample both query and window `pitch_50ms` at L = 24 evenly spaced voiced points (time-normalized; this absorbs tempo and note-count differences).
  - Compute Kendall's tau between the two 24-point sequences. The window scan is a numba kernel (no per-window scipy calls), verified in tests against a vectorized numpy tau-b, which is itself verified against scipy.
  - Ties: pitch differences smaller than `tie_tolerance` = 0.5 semitones count as ties (tau-b). This keeps CREPE jitter within a held note from turning same-note sample pairs into random signs.
- Refinement: re-scan the best 5 grid windows at a 1-frame (50 ms) start step and at scale +/- 0.025. Tau is sensitive to start shifts of a few frames, so the 0.25 s grid alone can miss the true alignment.
- Score: max tau over all windows. `offset_s` = start of the best window.

## Fusion (`fusion.py`)

- For each matcher, z-score its scores across all songs in the library for this query.
- Fused score = weighted sum of z-scores. Default weights equal; configurable in `config.toml`.
- Add priors (see Future hooks): `fused += prior_boost[song_id]`.
- Return top K (default 5) with per-matcher z-scores and the offset from DTW (configurable `offset_matcher`); fall back to the matcher with the highest z-score for that song when DTW was not run or found no match.

## CLI

```
earworm ingest --folder PATH
earworm ingest --csv FILE --source deezer
earworm search HUM_FILE [--artist NAME] [--k 5] [--matcher all|dtw|contour|rank]
earworm explain HUM_FILE SONG_ID     # saves PNG: query contour overlaid on best-matching song segment, per matcher
earworm eval [--weights-search]
earworm stats                         # library size, songs per source, ingest failures
```

`search` prints a table: rank, title, artist, fused score, per-matcher z-scores, offset.

## Evaluation harness (`eval/`)

`eval/labels.csv`:

```
hum_file,title,artist,part,in_preview,notes,true_offset_s
hum01.wav,Hotline Bling,Drake,chorus,y,,12.5
hum02.wav,...,...,verse,unknown,,
```

- `part`: chorus | verse | hook_instrumental | rap | intro | other
- `in_preview`: y | n | unknown (whether the hummed part falls inside the indexed audio)
- `true_offset_s` (optional): start of the hummed part in seconds **into the indexed audio** (for Deezer songs, into the 30 s preview, not the full track). Blank for hums outside the indexed audio. Invalid values are an error.

`run_eval.py` output:

- Per matcher and fused: top-1, top-3, top-5 accuracy and MRR
- Same metrics broken down by `part` and by `in_preview`
- Per-hum table: rank of the correct song under each matcher
- Saved to `eval/results/<timestamp>.csv` and printed as a table
- `--weights-search`: grid search fusion weights in steps of 0.1 and report the best. Print a warning that with a small eval set this overfits.

If a labeled song is not in the library, report it and exclude it from metrics (excluded hums are listed by reason: not in library vs unreadable hum, and the report header shows how many were evaluated vs excluded).

Report details:

- Ranks are pessimistic: songs tied with the correct one count as ranked above it, for every matcher, the fused ranking, the weights search and leave-one-out.
- Top-5 accuracy carries a 95% Wilson interval (at 30 hums it is roughly +/-15 points).
- Leave-one-out: fused accuracy with all three matchers and with each pair (dropped matcher weight 0), plus per-hum counts of hums whose rank improved/worsened and that entered/left the top 5. Judge a matcher by those counts, not by small accuracy differences.
- Label rows map to library songs by rapidfuzz (`[eval] label_match_threshold`, default 85); the matched song is printed per hum, with a warning when another song scores within `label_ambiguity_margin` (e.g. a remix next to the original).
- Offsets: for hums with `true_offset_s`, report per matcher and fused the fraction within `[eval] offset_tolerance_s` (default 1.0 s) and the median absolute error, over all such hums and over those where fused ranked the song first. A matcher with no offset counts as a miss. Warn when a labeled hum has `in_preview=n` or an offset beyond the indexed audio. Per hum, log each matcher's offset (of the true song), the fused offset and which matcher it came from, and the maximum spread (max - min) between the three matchers.
- Hum selection: `eval/pick_hums.py` draws a seeded random sample of the playlist (default 40 songs, seed 42) into `eval/hum_sample.csv`, independent of ingest results.

## Tests (`tests/`)

Synthetic tests that need no real recordings:

- Generate a library of 50 random melodies (MIDI note lists, 40 to 80 notes each, stepwise-biased random walk, realistic durations), rendered as sine tones with light harmonics at 16 kHz.
- Generate queries from a random 12 to 20 note excerpt of a library melody with:
  - random transposition (-7..+7 semitones)
  - tempo change (0.75x to 1.3x)
  - pitch noise: 25% of notes off by +/-1 semitone
  - one random note split or merge
- Assertions:
  - Each matcher: correct melody in top 3 for >= 80% of 100 queries
  - Fused: correct melody top 1 for >= 85%
  - `offset_s` within 1.0 s of the true excerpt start for >= 80% of correct matches
- A second query set with continuous transpositions (e.g. +2.37) and per-note intonation error (sd 0.25 st), with the same accuracy and offset thresholds, plus: DTW's per-frame cost on the true song is within 0.01 (mean) / 0.03 (p90) of a brute-force 0.05-semitone transposition search.
- Unit tests: Hz to MIDI conversion, octave fix, note segmentation on a known sine sequence, alignment cost on hand-built token strings, Kendall tau against scipy on random inputs.

If thresholds fail, investigate and fix the matcher rather than lowering thresholds. Report what was changed.

## M1 implementation status and notes

Synthetic suite (50 melodies, 100 queries each, real ffmpeg + CREPE), top-3 per matcher / fused top-1:

| | spec queries | fractional-key queries |
|---|---|---|
| dtw | 0.99 | 0.98 |
| contour | 1.00 | 0.98 |
| rank | 0.94 | 0.91 |
| fused top-1 | 1.00 | 1.00 |

Offsets within 1 s of the true start: 0.99 (fused, DTW offsets). Measured search time (CPU, 2 stems per song): 175 previews about 1.1 s; 1,000 previews about 4 s; 1,000 full-length (210 s) songs about 20 s total (DTW 4.6 s, rank 14.5 s, contour 0.4 s), plus about 4 s of CREPE on the hum.

Spec values changed after measurement (with the evidence):

- Rank scales: the spec's {0.7, 0.85, 1.0, 1.2, 1.4} reached only 0.79 top-3 even on perfect pitch; 15 scales (step 0.05) reach 0.93. A 0.5 st tie tolerance in Kendall's tau lifted the CREPE result from 0.63 to 0.83 top-3 (CREPE jitter inside held notes made same-note pairs random).
- DTW key search is fractional (see Matcher A), and rank re-scans the best windows at a 1-frame start step. A Queen self-excerpt exposed both: integer shifts left a cost of 0.3 on exact matches, and the 0.25 s start grid picked a window 2.25 s early because tau fell from 0.99 to 0.70 for a 2-frame start shift.
- Fused offset comes from DTW rather than the highest-z matcher, after a bad rank offset leaked into the result.

Known limitations:

- Nothing is measured on real hums yet. Real audio so far: 3 Deezer previews, queried with excerpts of their own vocals stems.
- Rank offsets are found in voiced-frame space, so a start error of a few frames across an unvoiced gap can still become seconds of wall-clock error (rank no longer sets the fused offset).
- Self-excerpt DTW cost is about 0.14 rather than 0 on real songs; suspected cause (unverified): different 50 ms block phase between hum and song.
- Above `fine_all_max_songs`, songs outside the coarse top 50 keep coarse DTW scores in the z-score.

## Milestone 2: Web app

`earworm serve` starts FastAPI on localhost.

API:

- `POST /search`: multipart `audio` file plus optional JSON `hints` (`{"artist": "Drake"}`). Returns:
  ```json
  {"results": [{"song_id": "", "title": "", "artist": "", "score": 0.0,
                "offset_s": 0.0, "per_matcher": {"dtw": 0.0, "contour": 0.0, "rank": 0.0}}],
   "timings_ms": {"pitch": 0, "match": 0},
   "debug": {"query_pitch": [], "best_segment_pitch": []}}
  ```
- `GET /audio/{song_id}`: serves the song audio with HTTP Range support.
- `GET /library`: song count and list.

`web/index.html`, single file, no build step:

- Record button using MediaRecorder (webm/opus), 20 s max with countdown, stop button. Also a file upload fallback.
- Optional artist hint text field.
- Results list: top 5 with title, artist, score, and an audio player that starts at `offset_s`.
- Canvas plot of query pitch vs best match segment (from `debug`).
- Thumbs up/down per result, appended to `data/feedback.csv` (useful later as labeled data).

## Future hooks (design for, don't build)

- `search()` accepts `priors: dict[song_id, float]`. M2 computes these only from the artist hint (rapidfuzz >= 85 on artist gives +0.5). Later an LLM will turn spoken context ("added a while back, probably Drake") into priors using `artist` and `added_at`.
- Lyrics matcher will implement the same `Matcher` protocol on Whisper transcripts and plug into fusion.
- Spotify playlist import will produce the same rows as `playlist_csv.py`.

## Acceptance criteria

M1 is done when:

1. `pytest` passes, including synthetic accuracy thresholds.
2. `earworm ingest` works on a folder of 20+ real songs and on an Exportify CSV via Deezer previews.
3. `earworm eval` runs end-to-end on real hums and writes the report.
4. `earworm explain` produces a readable plot.
5. README documents setup (including ffmpeg), commands, and how to record and label hums.

M2 is done when a hum recorded in the browser returns results in under 10 s on a 100-song library and playback starts at the matched offset.
