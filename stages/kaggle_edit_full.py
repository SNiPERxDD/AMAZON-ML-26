"""Fit the band edit model (H+words) on every hold-out and training-fold band pair, on a many-core or multi-GPU host.

Inputs are the tables written by ``kaggle_edit_prep.py`` (a directory given by ``--root``). Features are built once, in
row chunks, by a process pool. Fits, all scored on hold-out band pairs of the other S1-hash half:

- ``r2``: the hold-out half on ``refine2`` context, the ``probe_edit`` setup, as the reference (400 rounds);
- ``holdout``: the hold-out half on refit ``q`` context;
- ``both``: that half plus every training-fold pair;
- ``trainfolds``: the training-fold pairs alone, scoring the whole hold-out band;
- ``final``: every hold-out and training-fold pair, scoring the test US and India band.

The ``q`` fits run 800 rounds and are read at 400 and 800. Printed: F0.5 per country at 0.5/0.85 and at the best
grid rule. Written to ``--out/<mode>``: the hold-out scores of every fit and the test scores of ``final``.

Modes: ``scores`` is the plain model. ``scores_crossed`` gives the ``q`` fits the three crossed-word logits of
``crossed_words_probe.py`` (name words, address words, name 3-grams) as extra features, out of fold by ``fold``:
hold-out pairs from a regression on the training folds, each training fold from a regression on the other training
folds (so no hold-out label enters a logit used on the hold-out), and test pairs from a regression on every ``q`` band
pair. The ``r2`` fits run in the plain mode only.

Backends: ``lightgbm`` runs eight fits at once on the CPU, each with an equal share of the cores. ``xgboost`` runs one
fit at a time per device (by default every GPU that ``nvidia-smi`` lists), with the settings of ``PARAMS`` translated
to ``XGB_PARAMS``; training rows are streamed to the device in chunks, and each fit goes to the device with the least
queued work. The crossed logits are computed on the CPU while the plain fits run.

Run from the repository root: ``PYTHONPATH=. python stages/kaggle_edit_full.py --root data/kaggle_edit --out <dir>``
(add ``--backend xgboost --modes scores,scores_crossed`` on a GPU host).
"""

import argparse
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

try:
    import xgboost as xgb
except ImportError:  # local runs use only the LightGBM helpers
    xgb = None
from scipy import sparse

from stages.band_edit_probe import (
    BASE,
    BITS,
    PARAMS,
    edit_features,
    evaluate,
    hashed,
    word_features,
)
from stages.crossed_words_probe import BLOCKS, auc, crossed, logistic, logit

ROUNDS = (400, 800)
CHUNK = 200_000  # upper bound; chunks shrink so every feature worker gets about four
BANDS = {"r2": "train", "q": "train", "test": "test"}
CACHE: dict = {}
FEED = 1_000_000  # rows per chunk streamed to XGBoost and per scoring call
# The LightGBM settings of PARAMS in XGBoost terms; min_child_weight is a hessian sum, so 100 rows is not matched exactly.
XGB_PARAMS = {"objective": "binary:logistic", "eta": PARAMS["learning_rate"], "tree_method": "hist", "grow_policy": "lossguide",
              "max_leaves": PARAMS["num_leaves"], "max_depth": 0, "min_child_weight": 5, "colsample_bytree": PARAMS["feature_fraction"],
              "max_bin": 256, "seed": 42}


def band_path(root: str, name: str) -> str:
    """Return the path of one band table."""
    return f"{root}/{'test' if name == 'test' else name}_band.parquet"


def cached(root: str, name: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the pairs of one band and the raw S1 and S2/S3 sides (S2/S3 limited to the band), kept per process."""
    if name not in CACHE:
        CACHE.clear()
        split = BANDS[name]
        band = pl.read_parquet(band_path(root, name), columns=["s1", "s23"])
        s1 = pl.read_parquet(f"{root}/{split}_s1_raw.parquet")
        s23 = pl.read_parquet(f"{root}/{split}_s23_raw.parquet").join(band.select(idx="s23").unique(), on="idx", how="semi")
        CACHE[name] = (band, s1, s23)
    return CACHE[name]


def chunk_crossed(root: str, name: str, start: int, stride: int, block: str) -> sparse.csr_matrix:
    """Return the hashed one-side and crossed tokens of ``block`` for one chunk of a band."""
    band, s1, s23 = cached(root, name)
    return crossed(band.slice(start, stride), s1, s23, block)


def chunk_features(root: str, name: str, start: int, stride: int) -> tuple[np.ndarray, sparse.csr_matrix, list[str]]:
    """Return the dense text features, the hashed 3-gram and word matrix, and the dense names of one chunk of a band.

    The word rarity of ``word_features`` is taken over every S1 record and the S2/S3 records of the whole band, as in
    ``band_edit_probe.py``, so it does not depend on the chunking.
    """
    band, s1, s23 = cached(root, name)
    part = band.slice(start, stride)
    edit = edit_features(part, s1, s23)
    wdense, wmatrix = word_features(part, s1, s23)
    dense = np.hstack([edit.cast(pl.Float32).to_numpy(), wdense.to_numpy()])
    return dense, sparse.hstack([hashed(part, s1, s23), wmatrix], format="csr", dtype=np.float32), [*edit.columns, *wdense.columns]


def stride_of(height: int, workers: int) -> int:
    """Return the chunk size that gives every feature worker about four chunks."""
    return max(10_000, min(CHUNK, -(-height // (4 * workers))))


def crossed_logits(root: str, bands: dict, pool: ProcessPoolExecutor, workers: int) -> dict[str, np.ndarray]:
    """Return, for the ``q`` and ``test`` bands, the out-of-fold crossed-word logits of every block as an (n, 3) array."""
    started = time.time()
    fold, label = bands["q"]["fold"].to_numpy(), bands["q"]["t"].to_numpy().astype(np.float64)
    matrices = {}
    for block in BLOCKS:
        for name in [n for n in ("q", "test") if n in bands]:
            stride = stride_of(bands[name].height, workers)
            jobs = [pool.submit(chunk_crossed, root, name, start, stride, block) for start in range(0, bands[name].height, stride)]
            matrices[(name, block)] = stack_rows([job.result() for job in jobs])
    print(f"phase: crossed tokens done in {(time.time() - started) / 60:.1f} min", flush=True)
    train_folds = sorted(set(fold.tolist()) - {0})
    tasks = [(block, f) for block in BLOCKS for f in [0, *train_folds, *([None] if "test" in bands else [])]]

    def regress(block: str, f: int | None) -> tuple[str, int | None, np.ndarray]:
        """Fit one regression: ``f`` 0 scores the hold-out, a training fold scores that fold, ``None`` scores the test band."""
        rows = np.ones(len(fold), bool) if f is None else (fold != 0) & (fold != f)
        x = matrices[("q", block)]
        w = logistic(x[rows], label[rows])
        target = matrices[("test", block)] if f is None else x[fold == f]
        return block, f, logit(target, w).astype(np.float32)

    out = {"q": np.zeros((len(fold), len(BLOCKS)), np.float32)}
    if "test" in bands:
        out["test"] = np.zeros((bands["test"].height, len(BLOCKS)), np.float32)
    with ThreadPoolExecutor(3) as threads:  # each regression copies its training rows; 18 at once would not fit in 31 GB
        for block, f, values in threads.map(lambda t: regress(*t), tasks):
            if f is None:
                out["test"][:, BLOCKS.index(block)] = values
            else:
                out["q"][fold == f, BLOCKS.index(block)] = values
    hold = fold == 0
    print(f"phase: crossed logits done in {(time.time() - started) / 60:.1f} min; hold-out AUC "
          + " ".join(f"{b} {auc(label[hold].astype(np.int8), out['q'][hold, j]):.4f}" for j, b in enumerate(BLOCKS)), flush=True)
    return out


def stack_rows(blocks: list[sparse.csr_matrix]) -> sparse.csr_matrix:
    """Stack CSR blocks with equal column counts by concatenating their arrays, emptying ``blocks`` as it goes."""
    shape = (sum(b.shape[0] for b in blocks), blocks[0].shape[1])
    offsets = np.cumsum([0] + [b.nnz for b in blocks])
    indptr = np.concatenate([[0]] + [b.indptr[1:].astype(np.int64) + offsets[i] for i, b in enumerate(blocks)])
    data = np.concatenate([b.data for b in blocks])
    indices = np.concatenate([b.indices for b in blocks])
    blocks.clear()
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def chunk_to_disk(root: str, name: str, start: int, stride: int, spill: str) -> tuple[str, int, int, list[str]]:
    """Write one chunk's features (context, ``us``, text features, 3-grams, words) as a CSR ``.npz`` under ``spill``.

    Returns the file, its rows, its non-zeros and the dense names. Only the path crosses the process boundary, so the parent
    never holds more than one chunk besides the band's final arrays.
    """
    dense, matrix, dense_names = chunk_features(root, name, start, stride)
    context = pl.scan_parquet(band_path(root, name)).select(*BASE, "us").slice(start, stride).collect().cast(pl.Float32).to_numpy()
    block = sparse.hstack([sparse.csr_matrix(np.hstack([context, dense])), matrix], format="csr", dtype=np.float32)
    path = f"{spill}/{name}_{start}.npz"
    np.savez(path, data=block.data, indices=block.indices.astype(np.int32), indptr=block.indptr.astype(np.int64), cols=block.shape[1])
    return path, block.shape[0], block.nnz, dense_names


def build(root: str, name: str, band: pl.DataFrame, pool: ProcessPoolExecutor, workers: int, spill: str) -> tuple[sparse.csr_matrix, list[str]]:
    """Return the feature matrix of one band (context, ``us``, text features, 3-grams, words) and its column names.

    The workers spill their chunks to disk; the parent then fills preallocated ``data``/``indices``/``indptr`` arrays one
    chunk at a time. Returning chunks through the pool and concatenating them held about two copies of the band, which ran
    the 31 GB Kaggle host out of memory on the q band twice (v4, v5).
    """
    Path(spill).mkdir(parents=True, exist_ok=True)
    stride = stride_of(band.height, workers)
    parts = [job.result() for job in [pool.submit(chunk_to_disk, root, name, start, stride, spill) for start in range(0, band.height, stride)]]
    nnz = sum(p[2] for p in parts)
    data, indices, indptr = np.empty(nnz, np.float32), np.empty(nnz, np.int32), np.zeros(band.height + 1, np.int64)
    row = at = 0
    for path, rows, count, _ in parts:
        with np.load(path) as chunk:
            data[at:at + count], indices[at:at + count] = chunk["data"], chunk["indices"]
            indptr[row + 1:row + rows + 1] = chunk["indptr"][1:] + at
            cols = int(chunk["cols"])
        os.remove(path)
        row, at = row + rows, at + count
    x = sparse.csr_matrix((data, indices, indptr), shape=(band.height, cols))
    names = [*BASE, "us", *parts[0][3]] + [f"g{i}" for i in range(1 << BITS)] + [f"w{i}" for i in range(1 << BITS)]
    print(f"phase: features {name} done, {x.shape[0]} rows, {x.nnz} non-zeros", flush=True)
    return x, names


def cuda_devices() -> list[str]:
    """Return one ``cuda:<i>`` name per GPU listed by ``nvidia-smi -L``, or an empty list without one."""
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [f"cuda:{i}" for i, _ in enumerate(line for line in listing.splitlines() if line.startswith("GPU "))]


class Rows(xgb.DataIter if xgb else object):
    """Feed ``x[rows]`` to XGBoost in chunks of ``FEED`` rows, so the host never holds a copy of the whole subset."""

    def __init__(self, x: sparse.csr_matrix, label: np.ndarray, rows: np.ndarray) -> None:
        self.x, self.label, self.index, self.at = x, label, np.flatnonzero(rows), 0
        super().__init__()

    def next(self, input_data) -> bool:
        """Pass the next chunk; return False after the last one."""
        if self.at >= len(self.index):
            return False
        part = self.index[self.at:self.at + FEED]
        input_data(data=self.x[part], label=self.label[part])
        self.at += FEED
        return True

    def reset(self) -> None:
        """Restart from the first chunk."""
        self.at = 0


def fit_lightgbm(x: sparse.csr_matrix, label: np.ndarray, names: list[str], rows: np.ndarray, rounds: int, threads: int):
    """Return a LightGBM booster fitted on ``rows`` and a function that scores a matrix after ``r`` rounds."""
    booster = lgb.train(PARAMS | {"num_threads": threads}, lgb.Dataset(x[rows], label[rows], feature_name=names, params={"num_threads": threads}),
                        rounds)
    return lambda matrix, r: booster.predict(matrix, num_iteration=r, num_threads=threads)


def fit_xgboost(x: sparse.csr_matrix, label: np.ndarray, rows: np.ndarray, rounds: int, threads: int, device: str):
    """Return an XGBoost booster fitted on ``rows`` on ``device`` and a function that scores a matrix after ``r`` rounds.

    The training rows are streamed into a ``QuantileDMatrix``. If this XGBoost cannot train on a GPU from a
    host-built quantile matrix, the subset is copied into a plain ``DMatrix`` instead.
    """
    params = XGB_PARAMS | {"device": device, "nthread": threads}
    try:
        booster = xgb.train(params, xgb.QuantileDMatrix(Rows(x, label, rows), max_bin=XGB_PARAMS["max_bin"], nthread=threads), rounds)
    except xgb.core.XGBoostError as error:
        print(f"phase: quantile matrix failed on {device} ({str(error)[:120]}); plain DMatrix", flush=True)
        booster = xgb.train(params, xgb.DMatrix(x[rows], label[rows], nthread=threads), rounds)
    return lambda matrix, r: booster.predict(xgb.DMatrix(matrix, nthread=threads), iteration_range=(0, r))


def fit(spec: tuple, backend: str, device: str, threads: int) -> dict:
    """Fit one spec and return, per target name and round count, the scores of that target's rows.

    ``spec`` is (name, matrix, labels, column names, training rows, {target: (matrix, rows)}, rounds). Targets are
    scored in chunks of ``FEED`` rows.
    """
    name, x, label, names, rows, targets, rounds = spec
    started = time.time()
    if backend == "xgboost":
        score = fit_xgboost(x, label, rows, rounds, threads, device)
    else:
        score = fit_lightgbm(x, label, names, rows, rounds, threads)
    out = {}
    for target, (matrix, keep) in targets.items():
        index = np.flatnonzero(keep)
        for r in [k for k in ROUNDS if k <= rounds]:
            out[(target, r)] = np.concatenate([score(matrix[index[s:s + FEED]], r) for s in range(0, len(index), FEED)]).astype(np.float32)
    print(f"phase: fit {name} done on {rows.sum()} pairs, {device}, {(time.time() - started) / 60:.1f} min", flush=True)
    return out


class Scheduler:
    """Run fits on a fixed set of slots, one fit per slot at a time, each fit on the slot with the least queued work."""

    def __init__(self, backend: str, devices: list[str], threads: int) -> None:
        self.backend, self.devices, self.threads = backend, devices, threads
        self.pools = [ThreadPoolExecutor(1) for _ in devices]
        self.load = [0.0] * len(devices)

    def submit(self, spec: tuple) -> Future:
        """Queue ``spec`` on the slot with the least work (training rows times rounds)."""
        slot = int(np.argmin(self.load))
        self.load[slot] += spec[4].sum() * spec[6]
        return self.pools[slot].submit(fit, spec, self.backend, self.devices[slot], self.threads)


def specs_of(mode: str, bands: dict, x_r2: sparse.csr_matrix | None, x_q: sparse.csr_matrix, x_test: sparse.csr_matrix | None, names: list[str],
             names_q: list[str]) -> list[tuple]:
    """Return the fits of one mode, largest first; the ``r2`` fits do not depend on the logits and run in the plain mode only."""
    r2, q = bands["r2"], bands["q"]
    hold = q["fold"].to_numpy() == 0
    q_half, r2_half = q["half"].to_numpy(), r2["half"].to_numpy()
    y_q, y_r2 = q["t"].to_numpy(), r2["t"].to_numpy()
    specs = [] if x_test is None else [("final", x_q, y_q, names_q, np.ones(q.height, bool), {"test": (x_test, np.ones(x_test.shape[0], bool))}, ROUNDS[-1])]
    specs.append(("trainfolds", x_q, y_q, names_q, ~hold, {"q all": (x_q, hold)}, ROUNDS[-1]))
    for h in (0, 1):
        target = {f"q {h}": (x_q, hold & (q_half != h))}
        specs.append((f"both half {h}", x_q, y_q, names_q, (hold & (q_half == h)) | ~hold, target, ROUNDS[-1]))
    for h in (0, 1):
        specs.append((f"holdout half {h}", x_q, y_q, names_q, hold & (q_half == h), {f"q {h}": (x_q, hold & (q_half != h))}, ROUNDS[-1]))
    if mode == "scores" and x_r2 is not None:
        specs += [(f"r2 half {h}", x_r2, y_r2, names, r2_half == h, {f"r2 {h}": (x_r2, r2_half != h)}, ROUNDS[0]) for h in (0, 1)]
    return specs


def finish(out: str, specs: list[tuple], futures: list[Future], bands: dict, root: str, ids: pl.DataFrame, truth: pl.DataFrame) -> None:
    """Collect the scores of one mode, print F0.5 per fit, and write the hold-out and test scores to ``out``."""
    Path(out).mkdir(parents=True, exist_ok=True)
    r2, q = bands["r2"], bands["q"]
    hold = q["fold"].to_numpy() == 0
    q_half, r2_half = q["half"].to_numpy(), r2["half"].to_numpy()
    scored = {}
    for spec, future in zip(specs, futures, strict=True):
        for (target, r), values in future.result().items():
            if target == "test":
                scored[f"e{r}"] = values
                continue
            kind = spec[0].split(" ")[0]
            key = f"{kind} {r}"
            scored.setdefault(key, np.zeros(r2.height if kind == "r2" else q.height, np.float32))
            if target == "q all":
                scored[key][hold] = values
            elif kind == "r2":
                scored[key][r2_half != int(target[-1])] = values
            else:
                scored[key][hold & (q_half != int(target[-1]))] = values
    print(f"phase: {Path(out).name} fits collected", flush=True)
    r2_low, q_low = pl.read_parquet(f"{root}/r2_low.parquet"), pl.read_parquet(f"{root}/q_low.parquet")
    if any(k.startswith("r2") for k in scored):
        report("refine2 as is", r2, np.ones(r2.height, bool), r2["p"].to_numpy(), r2_low, ids, truth)
    report("refit q as is", q, hold, q["p"].to_numpy(), q_low, ids, truth)
    for key in sorted(k for k in scored if not k.startswith("e")):
        if key.startswith("r2"):
            report(key, r2, np.ones(r2.height, bool), scored[key], r2_low, ids, truth)
        else:
            report(key, q, hold, scored[key], q_low, ids, truth)
    if "holdout 800" in scored and "trainfolds 800" in scored:
        report("mean holdout+trainfolds 800", q, hold, (scored["holdout 800"] + scored["trainfolds 800"]) / 2, q_low, ids, truth)
    if any(k.startswith("r2") for k in scored):
        r2.select("s1", "s23").with_columns(**{k: pl.Series(v) for k, v in scored.items() if k.startswith("r2")}).write_parquet(f"{out}/r2_scores.parquet")
    q.filter(pl.Series(hold)).select("s1", "s23").with_columns(
        **{k: pl.Series(v[hold]) for k, v in scored.items() if not k.startswith(("r2", "e"))}).write_parquet(f"{out}/holdout_scores.parquet")
    if "test" in bands:
        bands["test"].select("s1", "s23").with_columns(**{k: pl.Series(v) for k, v in scored.items() if k.startswith("e")}).write_parquet(
            f"{out}/test_scores.parquet")
    print(f"phase: {Path(out).name} written", flush=True)


def report(name: str, band: pl.DataFrame, mask: np.ndarray, score: np.ndarray, low: pl.DataFrame, ids: pl.DataFrame, truth: pl.DataFrame) -> None:
    """Print F0.5 per country of the band pairs in ``mask`` scored by ``score`` plus the pairs below the band."""
    for country in ("US", "India"):
        cids = ids.filter(pl.col("country") == country).select("s1")
        keep = mask & (band["country"].to_numpy() == country)
        candidate = pl.concat([band.filter(pl.Series(keep)).select("s1", "s23", p=pl.Series(score[keep])), low.join(cids, on="s1", how="semi")])
        evaluate(name, candidate, truth.join(cids, on="s1", how="semi"), cids, country)


def main() -> None:
    """Build the features once, queue the fits of every mode on the backend, print the hold-out scores and write the score files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="directory of the kaggle_edit_prep.py tables")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--sample", type=int, default=1, help="keep S1 entities with s1 %% sample == 0 (smoke tests)")
    parser.add_argument("--workers", type=int, default=0, help="feature processes (0: from cores and memory)")
    parser.add_argument("--spill", default="", help="directory for the feature chunks (default: <out>/spill)")
    parser.add_argument("--skip-r2", action="store_true", help="skip the r2 band (its fits only reproduce the hold-out-only models)")
    parser.add_argument("--share", type=float, default=1.0, help="keep this share of training-fold S1 entities in the q band (by hash)")
    parser.add_argument("--no-test", action="store_true", help="skip the test band (no final fit, no test scores); hold-out comparison only")
    parser.add_argument("--slots", type=int, default=8, help="concurrent LightGBM fits")
    parser.add_argument("--threads", type=int, default=0, help="threads per fit (0: cores / number of slots)")
    parser.add_argument("--modes", default="scores", help="comma list of scores (plain) and scores_crossed (q fits with the crossed logits)")
    parser.add_argument("--backend", choices=("lightgbm", "xgboost"), default="lightgbm", help="lightgbm: eight concurrent CPU fits; "
                        "xgboost: one fit at a time per device")
    parser.add_argument("--devices", default="", help="xgboost devices, e.g. cuda:0,cuda:1 or cpu,cpu (default: every GPU, else cpu)")
    args = parser.parse_args()
    cores = os.cpu_count() or 8
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    workers = args.workers or max(1, min(cores // 4, int(ram_gb // 6), 48))
    print(f"phase: start, {cores} cores, {ram_gb:.0f} GB, {workers} feature workers", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    root = args.root
    if args.sample > 1:
        root = f"{args.out}/sample"
        Path(root).mkdir(exist_ok=True)
        paths = sorted(Path(args.root).glob("*.parquet"), key=lambda p: "raw" in p.name)
        for path in paths:
            frame = pl.scan_parquet(path)
            if path.name.endswith("s1_raw.parquet"):
                frame = frame.filter(pl.col("idx") % args.sample == 0)
            elif path.name.endswith("s23_raw.parquet"):
                split = path.name.split("_")[0]
                kept = pl.concat([pl.scan_parquet(f"{root}/{n}_band.parquet").select(idx="s23") for n, s in BANDS.items() if s == split])
                frame = frame.join(kept.unique(), on="idx", how="semi")
            elif "s1" in frame.collect_schema().names():
                frame = frame.filter(pl.col("s1") % args.sample == 0)
            frame.collect().write_parquet(f"{root}/{path.name}")
    if args.share < 1:
        shared = f"{args.out}/share"
        Path(shared).mkdir(exist_ok=True)
        for path in Path(root).glob("*.parquet"):
            if path.name == "q_band.parquet":
                pl.scan_parquet(path).filter((pl.col("fold") == 0) | (pl.col("s1").hash(13) % 1000 < int(args.share * 1000))).sink_parquet(
                    f"{shared}/{path.name}")
            elif not Path(f"{shared}/{path.name}").exists():
                os.symlink(path.resolve(), f"{shared}/{path.name}")
        root = shared
    ids = pl.read_parquet(f"{root}/holdout_ids.parquet")
    truth = pl.read_parquet(f"{root}/truth.parquet")
    bands = {name: pl.read_parquet(band_path(root, name)) for name in BANDS if not (args.no_test and name == "test")}
    modes = args.modes.split(",")
    if args.backend == "xgboost":
        devices = args.devices.split(",") if args.devices else (cuda_devices() or ["cpu"])
        threads = args.threads or max(1, cores // len(devices))
    else:
        devices = ["cpu"] * args.slots
        threads = args.threads or max(2, cores // len(devices))
    scheduler = Scheduler(args.backend, devices, threads)
    runs = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        spill = args.spill or f"{args.out}/spill"
        matrices = {name: build(root, name, band, pool, workers, spill) for name, band in bands.items() if not (args.skip_r2 and name == "r2")}
        x_q, names = matrices["q"]
        x_test = matrices["test"][0] if "test" in matrices else None
        x_r2 = matrices["r2"][0] if "r2" in matrices else None
        del matrices
        if "scores" in modes:
            specs = specs_of("scores", bands, x_r2, x_q, x_test, names, names)
            print(f"phase: fitting {len(specs)} scores models, {args.backend} on {','.join(sorted(set(devices)))} x{len(devices)}, "
                  f"{threads} threads each", flush=True)
            runs.append(("scores", specs, [scheduler.submit(s) for s in specs]))
        # The logits are computed on the CPU while the plain fits run on the devices.
        logits = crossed_logits(root, bands, pool, workers) if "scores_crossed" in modes else None
    if logits is not None:
        x_qc = sparse.hstack([x_q, sparse.csr_matrix(logits["q"])], format="csr")
        x_testc = sparse.hstack([x_test, sparse.csr_matrix(logits["test"])], format="csr") if x_test is not None else None
        specs = specs_of("scores_crossed", bands, x_r2, x_qc, x_testc, names, [*names, *(f"{b}_logit" for b in BLOCKS)])
        print(f"phase: fitting {len(specs)} scores_crossed models", flush=True)
        runs.append(("scores_crossed", specs, [scheduler.submit(s) for s in specs]))
    for mode, specs, futures in runs:
        finish(f"{args.out}/{mode}", specs, futures, bands, root, ids, truth)
    print("phase: done", flush=True)


if __name__ == "__main__":
    main()
