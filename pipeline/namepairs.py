"""Third-round candidates: S2/S3 records with an empty address that share two name tokens with an S1 entity.

After blocking and second-round pairs, 3.3% of train true links are never scored; more than
half of them have an S2/S3 record with no address numbers or words, which the address keys
cannot reach. This module proposes pairs whose two records share the country and any two
name tokens (records with at most ``MAX_TOKENS`` tokens, key blocks within ``CAP``), keeps
only S2/S3 records with an empty address, drops pairs that are already candidates, scores
them with their own classifier, and writes probabilities in the same format as the other
stages (``--pred blend+sibling+namepair``).

Steps, each run from the repository root under ``tools/run_guarded.sh``:

- ``python -m pipeline.namepairs pairs train`` (and ``test``)
- ``python -m pipeline.namepairs features train`` (and ``test``): pair features from
  :func:`features.pair_features`, key counts, and base-probability context of the S1 entity
  and of the S2/S3 record over ``--base`` (default ``blend+sibling``).
- ``python -m pipeline.namepairs fit --fold k`` (k = 0–4), ``predict train --oof``, ``predict test``:
  writes ``data/features/{split}/namepair_000.parquet``.

As in :mod:`pipeline.siblings`, per-S1 context columns from :func:`features.pair_features`
are computed over these pairs only.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import features, records, siblings, stack, train

NAMEPAIR_DIR = "data/namepairs/"
MODEL_DIR = "data/models/"
MAX_TOKENS = 6
CAP = (10, 30)
PREFIX = "namepair"
CHUNK_S1 = 400_000
PARAMS = siblings.PARAMS


def pairs_path(split: str) -> str:
    """Return the name-pair file of ``split``."""
    return f"{NAMEPAIR_DIR}{split}_pairs.parquet"


def part_path(split: str, part: int) -> str:
    """Return one name-pair feature part."""
    return f"{NAMEPAIR_DIR}{split}/part_{part:03d}.parquet"


def token_pair_keys(split: str, side: str) -> pl.DataFrame:
    """Return (``idx``, ``key``) for every pair of distinct name tokens of records with at most ``MAX_TOKENS`` tokens."""
    tokens = pl.scan_parquet(records.tokens_path(split, side)).select("idx", "country", "token").unique(["idx", "token"])
    tokens = tokens.filter(pl.len().over("idx") <= MAX_TOKENS)
    return (
        tokens.join(tokens.select("idx", other="token"), on="idx")
        .filter(pl.col("token") < pl.col("other"))
        .select("idx", key=pl.concat_str("country", "token", "other", separator="|").hash(seed=1))
        .collect()
    )


def write_pairs(split: str) -> None:
    """Write new pairs from capped token-pair blocks, with the number of shared keys and the smallest block sizes."""
    k1, k23 = token_pair_keys(split, "s1"), token_pair_keys(split, "s23")
    sizes = k1.group_by("key").agg(n1=pl.len()).join(k23.group_by("key").agg(n2=pl.len()), on="key")
    sizes = sizes.filter((pl.col("n1") <= CAP[0]) & (pl.col("n2") <= CAP[1]))
    empty = (
        pl.scan_parquet(records.records_path(split, "s23"))
        .filter(pl.col("numbers").list.len().fill_null(0) + pl.col("words").list.len().fill_null(0) == 0)
        .select("idx")
        .collect()
    )
    right = k23.join(empty, on="idx", how="semi").join(sizes, on="key").rename({"idx": "s23"})
    left = k1.join(sizes.select("key"), on="key", how="semi").rename({"idx": "s1"})
    del k1, k23, empty
    found = left.join(right, on="key").group_by(code=siblings.code("s1", "s23")).agg(
        np_keys=pl.len().cast(pl.UInt8), np_n1=pl.col("n1").min().cast(pl.UInt8), np_n2=pl.col("n2").min().cast(pl.UInt8)
    )
    del left, right
    scored = pl.concat([
        pl.scan_parquet(f"{records.CACHE_DIR}{split}_pairs.parquet").select(code=siblings.code("s1", "s23")),
        pl.scan_parquet(siblings.pairs_path(split)).select(code=siblings.code("s1", "s23")),
    ])
    found = found.lazy().join(scored, on="code", how="anti").collect()
    found = found.select(
        s1=(pl.col("code") // 2**32).cast(pl.UInt32), s23=(pl.col("code") % 2**32).cast(pl.UInt32), np_keys="np_keys", np_n1="np_n1", np_n2="np_n2"
    ).sort("s1", "s23")
    Path(NAMEPAIR_DIR).mkdir(parents=True, exist_ok=True)
    found.write_parquet(pairs_path(split))
    print(f"{split}: name pairs={len(found)}", flush=True)


def base_context(split: str, out_dir: str, base: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return base-probability summaries per S1 entity and per S2/S3 record over the pairs named by ``base``."""
    preds = train.prob_scan(split, out_dir, base)
    per_s1 = preds.group_by("s1").agg(
        base_s1_max=pl.col("p").max(), base_s1_n50=(pl.col("p") >= siblings.CONFIDENT).sum().cast(pl.UInt16), base_s1_sum=pl.col("p").sum()
    )
    per_s23 = preds.group_by("s23").agg(base_s23_max=pl.col("p").max(), base_s23_n50=(pl.col("p") >= siblings.CONFIDENT).sum().cast(pl.UInt16))
    return per_s1.collect(engine="streaming"), per_s23.collect(engine="streaming")


def write_features(split: str, out_dir: str, base: str) -> None:
    """Write name-pair features in S1 chunks, with labels on train."""
    pairs_all = pl.read_parquet(pairs_path(split))
    per_s1, per_s23 = base_context(split, out_dir, base)
    n_s2 = pl.scan_parquet(f"{records.PARQUET_DIR}{split}_source2.parquet").select(pl.len()).collect().item()
    truth = features.truth_links().with_columns(label=pl.lit(True)) if split == "train" else None
    Path(part_path(split, 0)).parent.mkdir(parents=True, exist_ok=True)
    high = int(pairs_all["s1"].max()) + 1
    for part, low in enumerate(range(0, high, CHUNK_S1)):
        pairs = pairs_all.filter(pl.col("s1").is_between(low, low + CHUNK_S1, closed="left"))
        s1 = features.side_table(split, "s1", pairs.select(idx="s1").unique())
        s23 = features.side_table(split, "s23", pairs.select(idx="s23").unique())
        frame = features.pair_features(pairs, s1, s23, n_s2).join(per_s1, on="s1", how="left").join(per_s23, on="s23", how="left")
        frame = frame.with_columns(s23_taken=(pl.col("base_s23_max") >= siblings.CONFIDENT).fill_null(False))
        if truth is not None:
            frame = frame.join(truth, on=["s1", "s23"], how="left").with_columns(pl.col("label").fill_null(False))
        frame.write_parquet(part_path(split, part))
        print(f"part {part}: pairs={len(frame)}", flush=True)
        del pairs, s1, s23, frame


def model_path_for(held_out: int) -> str:
    """Return the name-pair model that left out fold ``held_out``."""
    return f"{MODEL_DIR}{PREFIX}_lgbm_f{held_out}.txt"


def load(split: str) -> pl.DataFrame:
    """Return every name-pair feature part of ``split``."""
    return pl.read_parquet(f"{NAMEPAIR_DIR}{split}/part_*.parquet")


def fit(held_out: int, rounds: int, threads: int) -> None:
    """Train on every S1 entity outside fold ``held_out`` and early-stop on that fold."""
    frame = load("train").with_columns(fold=train.fold())
    columns = [c for c in frame.columns if c not in (*train.NON_FEATURES, "fold")]
    fit_rows, valid_rows = frame.filter(pl.col("fold") != held_out), frame.filter(pl.col("fold") == held_out)
    del frame
    dtrain = lgb.Dataset(train.matrix(fit_rows, columns), fit_rows["label"].to_numpy(), feature_name=columns, free_raw_data=True)
    dvalid = lgb.Dataset(train.matrix(valid_rows, columns), valid_rows["label"].to_numpy(), reference=dtrain)
    print(f"fold {held_out}: train rows={len(fit_rows)} positives={fit_rows['label'].sum()} valid rows={len(valid_rows)}", flush=True)
    booster = lgb.train(
        PARAMS | {"num_threads": threads},
        dtrain,
        rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path_for(held_out), num_iteration=booster.best_iteration)
    gain = booster.feature_importance("gain")
    top = sorted(zip(columns, gain / gain.sum(), strict=True), key=lambda item: -item[1])[:10]
    print(f"best iteration={booster.best_iteration} valid={dict(booster.best_score['valid_0'])}")
    print("gain share: " + ", ".join(f"{name}={share:.3f}" for name, share in top))


def predict(split: str, out_dir: str, oof: bool, threads: int) -> None:
    """Score every name pair and write ``namepair_000.parquet`` next to the other probability files."""
    frame = load(split)
    folds = range(train.FOLDS) if oof else [train.HOLDOUT]
    boosters = {k: lgb.Booster(model_file=model_path_for(k)) for k in folds}
    columns = boosters[train.HOLDOUT].feature_name()
    fold = frame.select(train.fold()).to_series().to_numpy() if oof else np.full(len(frame), train.HOLDOUT)
    p = np.empty(len(frame), dtype=np.float32)
    x = train.matrix(frame, columns)
    for k, booster in boosters.items():
        rows = fold == k
        p[rows] = booster.predict(x[rows], num_threads=threads)
    frame.select("s1", "s23").with_columns(p=pl.Series(p)).write_parquet(stack.prob_path(split, 0, out_dir, PREFIX))
    print(f"{split}: {len(frame)} name pairs scored, {(p >= siblings.CONFIDENT).sum()} at p >= {siblings.CONFIDENT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["pairs", "features", "fit", "predict"])
    parser.add_argument("split", nargs="?", default="train", choices=["train", "test"])
    parser.add_argument("--base", default="blend+sibling", help="probability prefixes, joined by '+', for the context features")
    parser.add_argument("--fold", type=int, default=train.HOLDOUT)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--oof", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out-dir", default=features.FEATURE_DIR)
    arguments = parser.parse_args()
    if arguments.step == "pairs":
        write_pairs(arguments.split)
    elif arguments.step == "features":
        write_features(arguments.split, arguments.out_dir, arguments.base)
    elif arguments.step == "fit":
        fit(arguments.fold, arguments.rounds, arguments.threads)
    else:
        predict(arguments.split, arguments.out_dir, arguments.oof, arguments.threads)
