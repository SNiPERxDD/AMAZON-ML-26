"""Second-round candidates: S2/S3 records that share a key with a confidently linked record.

On train, about 40% of the true links that blocking misses share an exact key (core name,
house number plus street, or first name token plus house number) with another S2/S3 record
truly linked to the same S1 entity. This module proposes those pairs from confident predicted
links, scores them with their own classifier, and writes probabilities in the same format as
the other stages, so the decision rule can read both sets together (``--pred blend+sibling``).

Steps, each run from the repository root under ``tools/run_guarded.sh``:

- ``python -m pipeline.siblings pairs train`` (and ``test``): for every pair with base
  probability at least ``CONFIDENT``, take the other S2/S3 records in the same key block
  (2 to ``CAP`` records) and drop pairs that are already candidates.
- ``python -m pipeline.siblings features train`` (and ``test``): pair features from
  :func:`features.pair_features`, sibling evidence, and base-probability context of the S1
  entity and of the S2/S3 record.
- ``python -m pipeline.siblings fit --fold k`` (k = 0–4), ``predict train --oof``, ``predict test``:
  writes ``data/features/{split}/sibling_000.parquet``.

Base probabilities for train must be out-of-fold, as for the stacked stages. Per-S1 context
features from :func:`features.pair_features` are computed over sibling pairs only, so they
differ in meaning from the same columns in the main pipeline; this model is trained on them
as they are.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import features, records, stack, train

SIBLING_DIR = "data/siblings/"
MODEL_DIR = "data/models/"
CONFIDENT = 0.5
CAP = 20
CHUNK_S1 = 400_000
KEYS = ("core", "house_street", "first_house")
PARAMS = train.PARAMS | {"num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 100}


def pairs_path(split: str) -> str:
    """Return the sibling pair file of ``split``."""
    return f"{SIBLING_DIR}{split}_pairs.parquet"


def part_path(split: str, part: int) -> str:
    """Return one sibling feature part."""
    return f"{SIBLING_DIR}{split}/part_{part:03d}.parquet"


def code(s1: str, s23: str) -> pl.Expr:
    """Encode a pair of row indices as one ``u64``."""
    return pl.col(s1).cast(pl.UInt64) * 2**32 + pl.col(s23).cast(pl.UInt64)


def s23_keys(split: str) -> pl.DataFrame:
    """Return hashed sibling keys per S2/S3 record; a key is null when its parts are missing."""
    table = pl.read_parquet(records.records_path(split, "s23")).select("idx", "country", "core", "house", "street")
    has_core, has_address = pl.col("core") != "", pl.col("house").is_not_null() & pl.col("street").is_not_null()
    first = pl.col("core").str.split(" ").list.first()
    return table.select(
        "idx",
        core=pl.when(has_core).then(pl.concat_str("country", "core", separator="|").hash(1)),
        house_street=pl.when(has_address).then(pl.concat_str("country", "house", "street", separator="|").hash(2)),
        first_house=pl.when(has_core).then(pl.concat_str(pl.col("country"), first, pl.col("house").fill_null(""), separator="|").hash(3)),
    )


def write_pairs(split: str, out_dir: str, base: str) -> None:
    """Write sibling pairs with the best confident probability behind them, counts and key flags."""
    confident = stack.pred_scan(split, out_dir, base=base).filter(pl.col("p") >= CONFIDENT).select("s1", "s23", "p").collect()
    keys = s23_keys(split)
    found = []
    for bit, key in enumerate(KEYS):
        block = keys.select("idx", k=key).drop_nulls("k")
        block = block.join(block.group_by("k").len().filter(pl.col("len").is_between(2, CAP)).select("k"), on="k", how="semi")
        reached = confident.join(block.rename({"idx": "s23"}), on="s23").join(block.rename({"idx": "b"}), on="k").filter(pl.col("b") != pl.col("s23"))
        found.append(
            reached.group_by(code=code("s1", "b")).agg(sib_p=pl.col("p").max(), sib_n=pl.len().cast(pl.UInt16)).with_columns(mask=pl.lit(1 << bit, pl.UInt8))
        )
        del block, reached
    siblings = pl.concat(found).group_by("code").agg(pl.col("sib_p").max(), pl.col("sib_n").sum(), pl.col("mask").sum())
    del found, keys
    union = pl.read_parquet(f"{records.CACHE_DIR}{split}_pairs.parquet").select(code=code("s1", "s23"))
    siblings = siblings.join(union, on="code", how="anti")
    del union
    siblings = siblings.select(
        s1=(pl.col("code") // 2**32).cast(pl.UInt32),
        s23=(pl.col("code") % 2**32).cast(pl.UInt32),
        sib_p="sib_p",
        sib_n="sib_n",
        n_keys=(pl.col("mask") // 1 % 2 + pl.col("mask") // 2 % 2 + pl.col("mask") // 4 % 2).cast(pl.UInt8),
        **{f"sib_{key}": (pl.col("mask") // (1 << bit) % 2 == 1) for bit, key in enumerate(KEYS)},
    ).sort("s1", "s23")
    Path(SIBLING_DIR).mkdir(parents=True, exist_ok=True)
    siblings.write_parquet(pairs_path(split))
    print(f"{split}: confident links={len(confident)} sibling pairs={len(siblings)}")


def base_context(split: str, out_dir: str, base: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return base-probability summaries per S1 entity and per S2/S3 record over the main candidates."""
    preds = stack.pred_scan(split, out_dir, base=base)
    per_s1 = preds.group_by("s1").agg(
        base_s1_max=pl.col("p").max(), base_s1_n50=(pl.col("p") >= CONFIDENT).sum().cast(pl.UInt16), base_s1_sum=pl.col("p").sum()
    )
    per_s23 = preds.group_by("s23").agg(base_s23_max=pl.col("p").max(), base_s23_n50=(pl.col("p") >= CONFIDENT).sum().cast(pl.UInt16))
    return per_s1.collect(engine="streaming"), per_s23.collect(engine="streaming")


def write_features(split: str, out_dir: str, base: str) -> None:
    """Write sibling-pair features in S1 chunks, with labels on train."""
    siblings = pl.read_parquet(pairs_path(split))
    per_s1, per_s23 = base_context(split, out_dir, base)
    n_s2 = pl.scan_parquet(f"{records.PARQUET_DIR}{split}_source2.parquet").select(pl.len()).collect().item()
    truth = features.truth_links().with_columns(label=pl.lit(True)) if split == "train" else None
    Path(part_path(split, 0)).parent.mkdir(parents=True, exist_ok=True)
    high = int(siblings["s1"].max()) + 1
    for part, low in enumerate(range(0, high, CHUNK_S1)):
        pairs = siblings.filter(pl.col("s1").is_between(low, low + CHUNK_S1, closed="left"))
        s1 = features.side_table(split, "s1", pairs.select(idx="s1").unique())
        s23 = features.side_table(split, "s23", pairs.select(idx="s23").unique())
        frame = features.pair_features(pairs, s1, s23, n_s2).join(per_s1, on="s1", how="left").join(per_s23, on="s23", how="left")
        frame = frame.with_columns(
            sib_ratio=pl.col("sib_p") / pl.col("base_s1_max"),
            s23_taken=(pl.col("base_s23_max") >= CONFIDENT).fill_null(False),
        )
        if truth is not None:
            frame = frame.join(truth, on=["s1", "s23"], how="left").with_columns(pl.col("label").fill_null(False))
        frame.write_parquet(part_path(split, part))
        print(f"part {part}: pairs={len(frame)}", flush=True)
        del pairs, s1, s23, frame


def model_path_for(held_out: int) -> str:
    """Return the sibling model that left out fold ``held_out``."""
    return f"{MODEL_DIR}sibling_lgbm_f{held_out}.txt"


def load(split: str) -> pl.DataFrame:
    """Return every sibling feature part of ``split``."""
    return pl.read_parquet(f"{SIBLING_DIR}{split}/part_*.parquet")


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
    """Score every sibling pair and write ``sibling_000.parquet`` next to the feature parts."""
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
    frame.select("s1", "s23").with_columns(p=pl.Series(p)).write_parquet(stack.prob_path(split, 0, out_dir, "sibling"))
    print(f"{split}: {len(frame)} sibling pairs scored, {(p >= CONFIDENT).sum()} at p >= {CONFIDENT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["pairs", "features", "fit", "predict"])
    parser.add_argument("split", nargs="?", default="train", choices=["train", "test"])
    parser.add_argument("--base", default="blend", help="prefix of the probability files that define confident links")
    parser.add_argument("--fold", type=int, default=train.HOLDOUT)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--oof", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out-dir", default=features.FEATURE_DIR)
    arguments = parser.parse_args()
    if arguments.step == "pairs":
        write_pairs(arguments.split, arguments.out_dir, arguments.base)
    elif arguments.step == "features":
        write_features(arguments.split, arguments.out_dir, arguments.base)
    elif arguments.step == "fit":
        fit(arguments.fold, arguments.rounds, arguments.threads)
    else:
        predict(arguments.split, arguments.out_dir, arguments.oof, arguments.threads)
