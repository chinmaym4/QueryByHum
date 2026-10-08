"""earworm command line."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from .config import get_config

log = logging.getLogger("earworm")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _query_features(path: str, cfg):
    from .audio import decode_mono16k
    from .features import features_from_audio

    return features_from_audio(decode_mono16k(path, int(cfg.audio.sample_rate)), cfg, is_query=True)


def _load_entries(cfg):
    from .library import Library

    lib = Library(cfg)
    entries = lib.entries()
    lib.close()
    if not entries:
        sys.exit("library is empty: run `earworm ingest` first")
    return entries


def cmd_ingest(args, cfg) -> int:
    from .ingest import ingest_songs
    from .library import Library

    lib = Library(cfg)
    if args.folder:
        from .sources.local import scan_folder

        songs = list(scan_folder(args.folder))
        log.info("found %d audio files in %s", len(songs), args.folder)
    else:
        from .sources.playlist_csv import read_playlist

        if args.source != "deezer":
            sys.exit("--csv currently requires --source deezer")
        from .sources.deezer import fetch_previews

        rows = read_playlist(args.csv)
        log.info("read %d rows from %s", len(rows), args.csv)
        songs = fetch_previews(rows, cfg, library=lib, force=args.force)
    counts = ingest_songs(songs, lib, cfg, force=args.force)
    print(f"processed={counts['processed']} skipped={counts['skipped']} failed={counts['failed']}")
    lib.close()
    return 0 if counts["failed"] == 0 else 1


def cmd_search(args, cfg) -> int:
    from .search import MATCHER_NAMES, artist_priors, search

    entries = _load_entries(cfg)
    t0 = time.perf_counter()
    q = _query_features(args.hum_file, cfg)
    t1 = time.perf_counter()
    matchers = MATCHER_NAMES if args.matcher == "all" else (args.matcher,)
    out = search(q, entries, cfg, matchers=matchers, priors=artist_priors(entries, args.artist, cfg))
    t2 = time.perf_counter()
    names = list(matchers)
    print(f"{'rank':>4s}  {'title':30s} {'artist':22s} {'fused':>7s} " + " ".join(f"{m:>8s}" for m in names)
          + f" {'offset':>8s}")
    for r in out.results[: args.k]:
        print(f"{r.rank:>4d}  {r.entry.title[:30]:30s} {r.entry.artist[:22]:22s} {r.fused:>7.2f} "
              + " ".join(f"{r.z[m]:>8.2f}" for m in names) + f" {r.offset_s:>7.1f}s")
    print(f"(pitch {1000 * (t1 - t0):.0f} ms, match {1000 * (t2 - t1):.0f} ms, {len(entries)} songs; "
          f"per-matcher columns are z-scores)")
    return 0


def cmd_explain(args, cfg) -> int:
    from .plots import explain_plot
    from .search import search

    entries = _load_entries(cfg)
    try:
        target = next(e for e in entries if e.song_id == args.song_id)
    except StopIteration:
        sys.exit(f"no song with id {args.song_id!r} (see `earworm stats --list`)")
    q = _query_features(args.hum_file, cfg)
    out = search(q, entries, cfg)
    r = next(x for x in out.results if x.entry is target)
    # plot each matcher on the stem it matched best
    stems = r.stem
    from .search import make_matchers

    results = {}
    song_feats = {}
    for m in make_matchers(cfg):
        f = target.stems[stems[m.name]]
        results[m.name] = r.per_matcher[m.name] if m.name != "dtw" else m.score(q, f)
        song_feats[m.name] = f
    # one figure per stem used keeps the song curve consistent within a panel
    out_path = Path(args.out or f"explain_{Path(args.hum_file).stem}__{target.song_id}.png")
    by_stem: dict[str, dict] = {}
    for name, res in results.items():
        by_stem.setdefault(stems[name], {})[name] = res
    paths = []
    for stem, res in by_stem.items():
        p = out_path if len(by_stem) == 1 else out_path.with_name(f"{out_path.stem}__{stem}{out_path.suffix}")
        explain_plot(q, target.stems[stem], res, f"{target.artist} - {target.title} [{stem}]  "
                     f"(fused rank {r.rank}, score {r.fused:.2f})", p, cfg, z=r.z)
        paths.append(p)
    for p in paths:
        print(f"saved {p}")
    return 0


def cmd_eval(args, cfg) -> int:
    from .evaluation import run_eval

    entries = _load_entries(cfg)
    eval_dir = Path(args.eval_dir) if args.eval_dir else REPO_ROOT / "eval"
    print(run_eval(cfg, entries, eval_dir / "labels.csv", eval_dir / "hums", eval_dir / "results",
                   do_weights_search=args.weights_search))
    return 0


def cmd_stats(args, cfg) -> int:
    from .library import Library

    lib = Library(cfg)
    songs = lib.songs()
    fails = lib.failures()
    print(f"library: {len(songs)} songs ({cfg.paths.data_dir})")
    for src, n in sorted(Counter(s.source for s in songs).items()):
        print(f"  {src}: {n}")
    missing = [s for s in songs if not lib.has_features(s.id)]
    if missing:
        print(f"  {len(missing)} songs missing features for some stems")
    print(f"ingest failures: {len(fails)}")
    for f in fails:
        print(f"  [{f['stage']}] {f['artist']} - {f['title']} ({f['source']}): {f['error'][:160]}")
    unmatched = Path(cfg.paths.data_dir) / "unmatched.csv"
    if unmatched.exists():
        n = max(0, sum(1 for _ in open(unmatched)) - 1)
        print(f"deezer unmatched rows (last CSV ingest): {n} (see {unmatched})")
    if args.list:
        for s in songs:
            print(f"  {s.id:50s} {s.artist} - {s.title} [{s.source}]")
    lib.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="earworm", description="Query-by-humming over your own music library")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="add songs to the library")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder", help="folder of audio files")
    g.add_argument("--csv", help="playlist CSV (title,artist[,added_at] or Exportify)")
    s.add_argument("--source", default="deezer", choices=["deezer"], help="audio source for --csv rows")
    s.add_argument("--force", action="store_true", help="recompute songs that already have features")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", help="find the song you hummed")
    s.add_argument("hum_file")
    s.add_argument("--artist", help="artist hint (adds a prior)")
    s.add_argument("--k", type=int, default=None)
    s.add_argument("--matcher", default="all", choices=["all", "dtw", "contour", "rank"])
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("explain", help="plot how a hum aligns to one song, per matcher")
    s.add_argument("hum_file")
    s.add_argument("song_id")
    s.add_argument("--out", help="output PNG path")
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("eval", help="run the real-hum evaluation")
    s.add_argument("--weights-search", action="store_true")
    s.add_argument("--eval-dir", help="directory with labels.csv and hums/ (default: ./eval)")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("stats", help="library size, sources, ingest failures")
    s.add_argument("--list", action="store_true", help="also list song ids")
    s.set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.config:
        from .config import load_config

        cfg = load_config(args.config)
    else:
        cfg = get_config()
    if getattr(args, "k", "unset") is None:
        args.k = int(cfg.fusion.top_k)
    return int(args.func(args, cfg) or 0)


if __name__ == "__main__":
    sys.exit(main())
