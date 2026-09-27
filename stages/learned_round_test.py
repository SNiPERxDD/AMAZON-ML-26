"""Test learned retrieval round: the hold-out round of ``learned_round_holdout.py`` applied to test.

Every test S2/S3 record of US or India with an address and best probability over ``BASE`` below 0.5 queries all test S1
records of its country with the saved hold-out tower; the top ``K`` neighbours that are not already candidates become new
pairs. They get the same features as the hold-out pairs (pair features, window, context, ``p1`` from the hold-out stage-1
model, base context, retrieval columns). One LightGBM model with ``reverse.PARAMS`` is fitted on all hold-out round pairs
(early stopping on a fifth of their S1) and scores them as ``q``.

The probe adds the pairs with ``q >= --cut`` (0.8 by default; the hold-out ``add`` policy at ranks 1-3 gave US +0.00132,
India +0.00254) to the links of ``--base`` and writes ``matching_results.tsv`` and ``candidate_pairs.tsv`` (the base
candidates plus the added pairs). France is unchanged. Accepted pairs per S1 are printed next to the hold-out rates
(US 0.0134, India 0.0292 at 0.8) to show the test density shift.
``--countries France`` runs the same round on France (no labels; the model fitted on US/India hold-out pairs transfers, as a
separate check showed), with caches suffixed ``_france``.

Steps, cached under ``data/learned/``: ``pairs``, ``features``, ``probe``. Run from the repository root under the guard:
``tools/run_guarded.sh --max-gb 12 --max-swap-gb 3 --log logs/learned_round_test.log -- env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 PYTHONPATH=. python stages/learned_round_test.py``.
"""

import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

from pipeline import features, namepairs, records, reverse, siblings, train
from stages.char_pair_probe import Tower
from stages.learned_retrieval_probe import encode
from stages.learned_round_holdout import (
    BASE,
    COLS,
    FEATS,
    HAS_ADDRESS,
    OUT,
    TOWER,
    K,
    byte_table,
    embed,
    search,
)

PAIRS_TEST = OUT / "test_pairs.parquet"
FEATS_TEST = OUT / "test_features.parquet"
HOLDOUT_RATE = {"US": 0.0134, "India": 0.0292}


def step_pairs(args: argparse.Namespace, started: float) -> None:
    """Retrieve the top ``K`` test S1 of every gated test S2/S3 record of ``--countries`` and keep the new pairs."""
    tower = Tower(width=128).to(args.device)
    tower.load_state_dict(torch.load(TOWER, map_location=args.device))
    best = train.prob_scan("test", features.FEATURE_DIR, BASE).group_by("s23").agg(best=pl.col("p").max())
    gated = (pl.scan_parquet(records.records_path("test", "s23")).select(COLS).filter(HAS_ADDRESS & pl.col("country").is_in(args.countries))
             .join(best, left_on="idx", right_on="s23", how="left").filter(pl.col("best").fill_null(0.0) < reverse.GATE)
             .drop("best").collect(engine="streaming"))
    print(f"phase: gated test S2/S3 {gated.height}, {(time.time() - started) / 60:.1f} min", flush=True)
    found = []
    for country in args.countries:
        c1 = byte_table("s1", split="test").filter(pl.col("country") == country)
        q = gated.filter(pl.col("country") == country)
        pool = embed(tower, encode(c1), args.device)
        c1_idx = c1["idx"].to_numpy()
        for a in range(0, q.height, args.query_chunk):
            qc = q.slice(a, args.query_chunk)
            idx, sim = search(embed(tower, encode(qc), args.device), pool, K + 1)
            for r in range(K):
                found.append(pl.DataFrame({
                    "s23": qc["idx"].to_numpy(), "s1": c1_idx[idx[:, r]], "rank": np.full(qc.height, r + 1),
                    "cos": sim[:, r], "d_top": sim[:, 0] - sim[:, r], "d_next": sim[:, r] - sim[:, r + 1], "gap12": sim[:, 0] - sim[:, 1],
                }).with_columns(pl.col("s1", "s23").cast(pl.UInt32), country=pl.lit(country), rank=pl.col("rank").cast(pl.UInt8))
                    .with_columns(pl.col("cos", "d_top", "d_next", "gap12").cast(pl.Float32)))
            del idx, sim
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"phase: {country} queries {a + qc.height}/{q.height}, {(time.time() - started) / 60:.1f} min", flush=True)
        del pool, c1, q
        gc.collect()
    known = pl.concat([
        pl.scan_parquet(f"{records.CACHE_DIR}test_pairs.parquet").select("s1", "s23"),
        pl.scan_parquet(siblings.pairs_path("test")).select("s1", "s23"),
        pl.scan_parquet(namepairs.pairs_path("test")).select("s1", "s23"),
        pl.scan_parquet(reverse.pairs_path("test")).select("s1", "s23"),
    ])
    pairs = pl.concat(found).lazy().join(known, on=["s1", "s23"], how="anti").collect(engine="streaming").sort("s1", "s23")
    pairs.write_parquet(args.pairs_path)
    print(f"phase: new test pairs {pairs.height}, per country and rank {sorted(pairs.group_by('country', 'rank').len().rows())}", flush=True)


def step_features(args: argparse.Namespace, started: float) -> None:
    """Features of the new test pairs, as ``learned_round_holdout.step_features`` without labels."""
    pairs_all = pl.read_parquet(args.pairs_path)
    flags = [c for c in pl.scan_parquet(f"{features.FEATURE_DIR}test/part_000.parquet").collect_schema().names() if c.startswith("f_")]
    per_s1, per_s23 = namepairs.base_context("test", features.FEATURE_DIR, BASE)
    n_s2 = pl.scan_parquet(f"{records.PARQUET_DIR}test_source2.parquet").select(pl.len()).collect().item()
    parts = []
    high = int(pairs_all["s1"].max()) + 1
    for low in range(0, high, reverse.CHUNK_S1):
        pairs = pairs_all.filter(pl.col("s1").is_between(low, low + reverse.CHUNK_S1, closed="left"))
        if pairs.is_empty():
            continue
        s1 = features.side_table("test", "s1", pairs.select(idx="s1").unique())
        s23 = features.side_table("test", "s23", pairs.select(idx="s23").unique())
        frame = features.pair_features(pairs.select("s1", "s23"), s1, s23, n_s2)
        frame = reverse.with_context(reverse.with_window(frame, "test", features.FEATURE_DIR), "test", features.FEATURE_DIR)
        frame = frame.with_columns(*[pl.lit(False).alias(f) for f in flags], n_families=pl.lit(0, pl.UInt8))
        frame = frame.with_columns(p1=pl.Series(reverse.stage1(frame, "test", args.threads))).drop(*flags, "n_families")
        frame = frame.join(pairs.drop("country"), on=["s1", "s23"], how="left").join(per_s1, on="s1", how="left").join(per_s23, on="s23", how="left")
        parts.append(frame)
        print(f"phase: features S1 from {low}: {pairs.height} pairs, {(time.time() - started) / 60:.1f} min", flush=True)
        del pairs, s1, s23, frame
        gc.collect()
    pl.concat(parts, how="diagonal_relaxed").write_parquet(args.feats_path)


def fit_full(threads: int) -> lgb.Booster:
    """Fit the round model on every hold-out round pair, early-stopping on a fifth of their S1."""
    frame = pl.read_parquet(FEATS).filter(pl.col("country").is_in(list(HOLDOUT_RATE))).drop("country")
    columns = [c for c in frame.columns if c not in train.NON_FEATURES]
    inner = (frame["s1"].hash(13) % 5 == 0).to_numpy()
    fit_rows, stop_rows = frame.filter(pl.Series(~inner)), frame.filter(pl.Series(inner))
    dtrain = lgb.Dataset(train.matrix(fit_rows, columns), fit_rows["label"].to_numpy(), feature_name=columns)
    dvalid = lgb.Dataset(train.matrix(stop_rows, columns), stop_rows["label"].to_numpy(), reference=dtrain)
    booster = lgb.train(reverse.PARAMS | {"num_threads": threads}, dtrain, 2000, valid_sets=[dvalid], callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"full fit: rows {fit_rows.height}, positives {fit_rows['label'].sum()}, best iteration {booster.best_iteration}", flush=True)
    return booster


def step_probe(args: argparse.Namespace) -> None:
    """Score the test pairs and write the probe: base links plus accepted pairs."""
    booster = fit_full(args.threads)
    frame = pl.read_parquet(args.feats_path)
    frame = frame.with_columns(q=pl.Series(booster.predict(train.matrix(frame, booster.feature_name()), num_iteration=booster.best_iteration,
                                                           num_threads=args.threads)))
    s1, s23 = records.load_split("test")
    s1_ids, s23_ids = s1.select(s1="idx", source1_entity_id="entity_id"), s23.select(s23="idx", entity_id="entity_id")
    for country in args.countries:
        rate = HOLDOUT_RATE.get(country, float("nan"))
        n_s1 = s1.filter(pl.col("country") == country).height
        c = frame.filter(pl.col("country") == country)
        for cut in (0.5, 0.7, 0.8, 0.9):
            n = c.filter(pl.col("q") >= cut).height
            print(f"{country} q >= {cut}: accepted {n} ({n / n_s1:.4f}/S1; hold-out at 0.8: {rate:.4f}), "
                  f"new pairs {c.height} ({c.height / n_s1:.3f}/S1)", flush=True)
    added = frame.filter(pl.col("q") >= args.cut).select("s1", "s23").join(s1_ids, on="s1").join(s23_ids, on="s23").select("source1_entity_id", e="entity_id")
    out = args.out
    Path(out).mkdir(parents=True, exist_ok=True)
    options = {"separator": "\t", "quote_style": "never"}
    for name, column in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        base = (pl.read_csv(f"{args.base}{name}", separator="\t", infer_schema=False)
                .select("source1_entity_id", e=pl.col(column).str.split(",")).explode("e").drop_nulls().filter(pl.col("e") != ""))
        both = pl.concat([base, added]).unique()
        lists = both.group_by("source1_entity_id").agg(pl.col("e").sort().str.join(",").alias(column))
        order = pl.read_csv(f"{args.base}{name}", separator="\t", infer_schema=False, columns=["source1_entity_id"])
        order.join(lists, on="source1_entity_id", how="left").with_columns(pl.col(column).fill_null("")).write_csv(f"{out}{name}", **options)
        print(f"{name}: base links {base.height}, written links {both.height} (+{both.height - base.height})", flush=True)


def main() -> None:
    """Run the requested step, or every step whose output is missing, then the probe."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", default="all", choices=["all", "pairs", "features", "probe"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--query_chunk", type=int, default=400_000)
    parser.add_argument("--cut", type=float, default=0.8)
    parser.add_argument("--base", default="output/probes/probe_xedit_fr950/")
    parser.add_argument("--out", default="output/probes/probe_learned/")
    parser.add_argument("--countries", default="US,India", help="comma-separated; caches get a suffix unless US,India")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    args.countries = args.countries.split(",")
    suffix = "" if args.countries == list(HOLDOUT_RATE) else "_" + "_".join(c.lower() for c in args.countries)
    args.pairs_path = PAIRS_TEST.with_name(f"{PAIRS_TEST.stem}{suffix}.parquet")
    args.feats_path = FEATS_TEST.with_name(f"{FEATS_TEST.stem}{suffix}.parquet")
    started = time.time()
    torch.manual_seed(42)
    for name, path, run in (("pairs", args.pairs_path, step_pairs), ("features", args.feats_path, step_features)):
        if args.step == name or (args.step == "all" and not path.exists()):
            run(args, started)
    if args.step in ("all", "probe"):
        step_probe(args)
    print(f"phase: done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
