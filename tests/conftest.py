import copy
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import synth  # noqa: E402

from earworm.config import get_config  # noqa: E402

N_SONGS = 50
N_QUERIES = 100
SEED = 1234


@pytest.fixture(scope="session")
def cfg():
    return copy.deepcopy(get_config())


@dataclass
class SyntheticSet:
    library: list            # list[synth.Melody]
    queries: list            # list[synth.Query]
    song_crepe: list         # list[(hz, periodicity)]
    query_crepe: list


def _cache_key(cfg) -> str:
    p = cfg.pitch
    h = hashlib.sha1()
    h.update(Path(synth.__file__).read_bytes())
    h.update(repr((SEED, N_SONGS, N_QUERIES, p.model, p.hop_s, p.fmin, p.fmax)).encode())
    return h.hexdigest()[:12]


@pytest.fixture(scope="session")
def synthetic_set(cfg, request) -> SyntheticSet:
    """50 rendered melodies + 100 degraded queries, with raw CREPE output.

    CREPE output is cached in .pytest_cache (keyed on synth.py and the CREPE
    parameters) since it dominates runtime on CPU; every later stage (pitch
    cleaning, notes, matching) is recomputed on every run.
    """
    from earworm.pitch import run_crepe

    rng = np.random.default_rng(SEED)
    library = [synth.random_melody(rng) for _ in range(N_SONGS)]
    queries = [synth.make_query(rng, library, int(rng.integers(0, N_SONGS))) for _ in range(N_QUERIES)]

    cache_dir = Path(request.config.cache.mkdir("earworm_synth"))
    path = cache_dir / f"crepe_{_cache_key(cfg)}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return SyntheticSet(library, queries, list(z["songs"]), list(z["queries"]))

    noise = np.random.default_rng(SEED + 1)
    song_crepe = [run_crepe(synth.render(m, rng=noise), cfg) for m in library]
    query_crepe = [run_crepe(synth.render(q.melody, rng=noise), cfg) for q in queries]
    songs_arr = np.empty(len(song_crepe), dtype=object)
    songs_arr[:] = song_crepe
    q_arr = np.empty(len(query_crepe), dtype=object)
    q_arr[:] = query_crepe
    np.savez(path, songs=songs_arr, queries=q_arr)
    return SyntheticSet(library, queries, song_crepe, query_crepe)
