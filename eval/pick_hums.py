"""Pick which songs to hum for the real eval: a seeded random sample of the playlist.

    uv run python eval/pick_hums.py [--n 40] [--seed 42] [--playlist eval/playlist.csv]

Writes eval/hum_sample.csv and prints the list with its provenance (seed, playlist
checksum). The draw depends only on the seed and the playlist file, never on which
songs ingested or matched, so re-running with the same inputs gives the same list.
"""

import argparse
import csv
import hashlib
from pathlib import Path

from earworm.evaluation import sample_playlist
from earworm.sources.playlist_csv import read_playlist

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--playlist", default=str(HERE / "playlist.csv"))
    ap.add_argument("--out", default=str(HERE / "hum_sample.csv"))
    args = ap.parse_args()

    rows = read_playlist(args.playlist)
    sha = hashlib.sha256(Path(args.playlist).read_bytes()).hexdigest()
    sample = sample_playlist(rows, args.n, args.seed)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["playlist_row", "title", "artist"])
        for i, r in sample:
            w.writerow([i, r.title, r.artist])
    print(f"sample of {args.n} from {len(rows)} playlist rows | seed={args.seed} | "
          f"playlist sha256={sha[:16]} | method=random.Random(seed).sample(range(n_rows), n), sorted by row")
    for k, (i, r) in enumerate(sample, 1):
        print(f"{k:2d}. row {i:3d}  {r.title}  -  {r.artist}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
