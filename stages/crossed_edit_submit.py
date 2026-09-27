"""Write test submissions from the band edit model (H+words) fitted on the whole hold-out band, with and without the crossed-word logits.

This is the ``probe_edit`` setup (``refine2`` context, US and India hold-out band pairs with ``p >= LOW``, 400 rounds)
with two changes measured on the hold-out: word features (H+words) and the three crossed-word logits of
``crossed_words_probe.py`` (name words, address words, name 3-grams; +0.00023 US / +0.00046 India over H+words at
0.5/0.85). The hold-out logits are cross-fitted over the two S1-hash halves; the test logits come from a regression on
the whole hold-out band.

Features are built by a process pool with the helpers of ``kaggle_edit_full.py``; the two fits run concurrently. The test
band is featurised and scored in slabs, so its matrix is never held whole. The inputs live in ``data/kaggle_edit_r2/``:
hard links to the ``kaggle_edit_prep.py`` tables plus a test band on ``refine2`` context.

Written: ``output/probes/probe_hwords/`` (H+words) and ``output/probes/probe_xedit/`` (H+words + crossed logits), each
with ``matching_results.tsv`` and ``candidate_pairs.tsv`` linked from ``output/probes/probe_reverse/``. France keeps its
``refine2`` probabilities; the rule is ``submit.TEST_RULES``. Printed: kept links per S1 by country, and links added and
dropped against ``probe_edit``.

Run from the repository root: ``PYTHONPATH=. python stages/crossed_edit_submit.py [--workers 4]``.
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import sparse

from pipeline import features, records, refine2, submit, train
from stages import kaggle_edit_full as kef
from stages.band_edit_probe import BASE, LOW, PARAMS, ROUNDS, context, raw_side
from stages.crossed_words_probe import BLOCKS, auc, logistic, logit
from stages.kaggle_edit_prep import KEYS

D = features.FEATURE_DIR
SOURCE = Path("data/kaggle_edit")
ROOT = Path("data/kaggle_edit_r2")
LINKED = ("r2_band", "holdout_ids", "truth", "train_s1_raw", "train_s23_raw", "test_s1_raw")
BASELINE = "output/probes/probe_reverse/"
REFERENCE = "output/probes/probe_edit/matching_results.tsv"
SLAB = 500_000


def prepare(ids: pl.DataFrame) -> None:
    """Link the hold-out tables into ``ROOT`` and write the test band on ``refine2`` context with its raw S2/S3 side."""
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in LINKED:
        if not (ROOT / f"{name}.parquet").exists():
            os.link(SOURCE / f"{name}.parquet", ROOT / f"{name}.parquet")
    if (ROOT / "test_s23_raw.parquet").exists():
        return
    preds = context(train.prob_scan("test", D, refine2.PREFIX).join(ids.lazy(), on="s1", how="semi").collect())
    band = preds.filter(pl.col("p") >= LOW).select(KEYS).join(ids, on="s1").with_columns(us=(pl.col("country") == "US").cast(pl.Int8))
    band.write_parquet(ROOT / "test_band.parquet")
    raw_side("s23", "test").join(band.select(idx="s23").unique(), on="idx", how="semi").write_parquet(ROOT / "test_s23_raw.parquet")
    print(f"test band on refine2 context: {band.height} pairs", flush=True)


def slab(pool: ProcessPoolExecutor, band: pl.DataFrame, start: int, stop: int, workers: int) -> tuple[sparse.csr_matrix, dict]:
    """Return the H+words matrix and the crossed token matrices of rows ``start:stop`` of the test band."""
    stride = kef.stride_of(stop - start, workers)
    starts = range(start, stop, stride)
    lengths = [min(stride, stop - s) for s in starts]
    feats = [pool.submit(kef.chunk_features, str(ROOT), "test", s, n) for s, n in zip(starts, lengths, strict=True)]
    cross = {b: [pool.submit(kef.chunk_crossed, str(ROOT), "test", s, n, b) for s, n in zip(starts, lengths, strict=True)] for b in BLOCKS}
    parts = [job.result() for job in feats]
    dense = np.hstack([band.slice(start, stop - start).select(*BASE, "us").cast(pl.Float32).to_numpy(), np.vstack([p[0] for p in parts])])
    x = sparse.hstack([sparse.csr_matrix(dense), sparse.vstack([p[1] for p in parts], format="csr")], format="csr", dtype=np.float32)
    return x, {b: sparse.vstack([job.result() for job in jobs], format="csr") for b, jobs in cross.items()}


def write_probe(out: str, frame: pl.DataFrame, rule: dict, test_country: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> None:
    """Write ``matching_results.tsv`` for ``frame`` to ``out``, link the candidate pairs, and print kept links and the change against ``probe_edit``."""
    kept = submit.decide_by_country(train.with_maxima(frame.lazy()).collect(), rule, submit.TEST_RULES)
    per = kept.join(test_country, on="s1").group_by("country").len().join(test_country.group_by("country").len(), on="country", suffix="_s1")
    Path(out).mkdir(parents=True, exist_ok=True)
    lists = submit.id_lists(kept, s1.select(s1="idx", source1_entity_id="entity_id"), s23.select(s23="idx", entity_id="entity_id"), "matched_entity_ids")
    lists.write_csv(f"{out}/matching_results.tsv", separator="\t", quote_style="never")
    if not Path(f"{out}/candidate_pairs.tsv").exists():
        os.link(f"{BASELINE}candidate_pairs.tsv", f"{out}/candidate_pairs.tsv")
    links = [pl.read_csv(path, separator="\t", infer_schema=False).select("source1_entity_id", e=pl.col("matched_entity_ids").str.split(",")).explode("e").drop_nulls()
             for path in (f"{out}/matching_results.tsv", REFERENCE)]
    added, dropped = links[0].join(links[1], on=["source1_entity_id", "e"], how="anti").height, links[1].join(links[0], on=["source1_entity_id", "e"], how="anti").height
    print(f"{out}: " + " ".join(f"{c}={n / m:.4f}" for c, n, m in per.sort("country").iter_rows())
          + f"; kept links {kept.height}; against probe_edit +{added} -{dropped}", flush=True)


def main() -> None:
    """Fit both models on the hold-out band, score the test band in slabs, and write the two probes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4, help="feature processes (each caches about 1.5 GB)")
    parser.add_argument("--xedit-only", action="store_true", help="fit and score only the xedit model (less memory)")
    parser.add_argument("--tag", default="", help="suffix of the probe directories and of the saved test scores")
    parser.add_argument("--slab", type=int, default=SLAB, help="test pairs scored per slab")
    parser.add_argument("--resume", action="store_true", help="with --xedit-only: load the saved xedit booster and weights, skip the fits")
    args = parser.parse_args()
    started = time.time()
    cores = os.cpu_count() or 8
    test_country = pl.read_parquet(records.records_path("test", "s1"), columns=["idx", "country"]).rename({"idx": "s1"})
    prepare(test_country.filter(pl.col("country").is_in(["US", "India"])))
    r2, test = pl.read_parquet(ROOT / "r2_band.parquet"), pl.read_parquet(ROOT / "test_band.parquet")
    y, half = r2["t"].to_numpy(), r2["half"].to_numpy()
    saved = ROOT / f"xedit_booster{args.tag}.txt", ROOT / f"xedit_weights{args.tag}.npz"
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        if args.resume and args.xedit_only and all(f.exists() for f in saved):
            boosters = {"xedit": lgb.Booster(model_file=str(saved[0]))}
            weights = dict(np.load(saved[1]))
            print("phase: loaded saved xedit booster and weights", flush=True)
        else:
            x, names = kef.build(str(ROOT), "r2", r2, pool, args.workers, str(ROOT / "spill"))
            stride = kef.stride_of(r2.height, args.workers)
            cross = {b: sparse.vstack([job.result() for job in [pool.submit(kef.chunk_crossed, str(ROOT), "r2", s, stride, b) for s in range(0, r2.height, stride)]],
                                      format="csr") for b in BLOCKS}
            print(f"phase: hold-out features done in {(time.time() - started) / 60:.1f} min", flush=True)

            def regress(block: str, h: int | None) -> tuple[str, int | None, np.ndarray]:
                """Fit one regression on the half other than ``h`` (``None``: the whole band) and return its weights."""
                rows = np.ones(r2.height, bool) if h is None else half != h
                return block, h, logistic(cross[block][rows], y[rows].astype(np.float64))

            logits, weights = np.zeros((r2.height, len(BLOCKS)), np.float32), {}
            with ThreadPoolExecutor(3 * len(BLOCKS)) as threads:
                for block, h, w in threads.map(lambda t: regress(*t), [(b, h) for b in BLOCKS for h in (0, 1, None)]):
                    if h is None:
                        weights[block] = w
                    else:
                        logits[half == h, BLOCKS.index(block)] = logit(cross[block][half == h], w)
            cross.clear()
            print("phase: logits done; cross-fitted AUC " + " ".join(f"{b} {auc(y, logits[:, j]):.4f}" for j, b in enumerate(BLOCKS)), flush=True)
            xc = sparse.hstack([x, sparse.csr_matrix(logits)], format="csr")
            models = {"hwords": (x, names), "xedit": (xc, [*names, *(f"{b}_logit" for b in BLOCKS)])}
            if args.xedit_only:
                del models["hwords"]
            threads_each = max(1, cores // len(models))

            def fit(matrix: sparse.csr_matrix, columns: list[str]) -> lgb.Booster:
                """Fit the edit model on every hold-out band pair."""
                return lgb.train(PARAMS | {"num_threads": threads_each}, lgb.Dataset(matrix, y, feature_name=columns, params={"num_threads": threads_each}), ROUNDS)

            with ThreadPoolExecutor(len(models)) as threads:
                boosters = dict(zip(models, threads.map(lambda m: fit(*m), models.values()), strict=True))
            del x, xc, models
            print(f"phase: fits done in {(time.time() - started) / 60:.1f} min", flush=True)
            if "xedit" in boosters:
                boosters["xedit"].save_model(str(saved[0]))
                np.savez(saved[1], **weights)
        scores = {name: np.zeros(test.height, np.float32) for name in boosters}
        for start in range(0, test.height, args.slab):
            stop = min(start + args.slab, test.height)
            xt, crossed = slab(pool, test, start, stop, args.workers)
            if "hwords" in boosters:
                scores["hwords"][start:stop] = boosters["hwords"].predict(xt, num_threads=cores)
            extra = np.column_stack([logit(crossed[b], weights[b]) for b in BLOCKS]).astype(np.float32)
            scores["xedit"][start:stop] = boosters["xedit"].predict(sparse.hstack([xt, sparse.csr_matrix(extra)], format="csr"), num_threads=cores)
            print(f"phase: scored {stop} of {test.height} test pairs, {(time.time() - started) / 60:.1f} min", flush=True)
    everything = train.prob_scan("test", D, refine2.PREFIX).collect()
    rule = json.loads(Path(train.RULE_PATH).read_text())
    s1, s23 = records.load_split("test")
    test.select("s1", "s23", **{name: pl.Series(score) for name, score in scores.items()}).write_parquet(ROOT / f"test_scores{args.tag}.parquet")
    for name, score in scores.items():
        new = test.select("s1", "s23", q=pl.Series(score))
        edited = everything.join(new, on=["s1", "s23"], how="left").with_columns(p=pl.coalesce("q", "p")).drop("q")
        write_probe(f"output/probes/probe_{name}{args.tag}", edited, rule, test_country, s1, s23)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
