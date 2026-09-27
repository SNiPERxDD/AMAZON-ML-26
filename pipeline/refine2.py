"""Second refit of the band pairs over the per-entity context of the refit probabilities.

The refit (``pipeline/refine.py``) sees per-S1 and per-S2/S3 summaries of the base
probabilities, which are less accurate than its own output. This stage gives a second model
the refit features plus, per S1 and per S2/S3, the best, second-best, count at 0.5, rank of
the pair and ratio to the best of the refit probability ``q``.

Training needs ``q`` that no model saw the labels of: the refit is fitted four times on three
of the training folds and scores the fourth (``refine-oof.parquet``). The hold-out and test use
the pipeline refit (``refine_000.parquet``), which was fitted on all four. Test pairs of
countries that train does not have (France) keep their base probability, as in the refit.

Steps, each run from the repository root under ``tools/run_guarded.sh``, after the refit:

- ``python -m pipeline.refine2 fit``
- ``python -m pipeline.refine2 predict train`` (hold-out fold) and ``predict test``: writes
  ``data/features/{split}/refine2_000.parquet``
- ``python -m pipeline.train tune --pred refine2``
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import features, refine, stack, train

MODEL_PATH = "data/models/refine2_lgbm.txt"
PREFIX = "refine2"


def oof_path(out_dir: str) -> str:
    """Return the path of the out-of-fold refit probabilities of training folds other than the hold-out."""
    return f"{out_dir}train/{refine.PREFIX}-oof.parquet"


def context(split: str, out_dir: str) -> pl.DataFrame:
    """Return per band pair the refit probability ``q`` with its S1 and S2/S3 summaries."""
    probs = pl.read_parquet(stack.prob_path(split, 0, out_dir, refine.PREFIX)).select("s1", "s23", q="p")
    if split == "train":
        probs = pl.concat([probs, pl.read_parquet(oof_path(out_dir))])
    for side in ("s1", "s23"):
        probs = probs.with_columns(
            pl.col("q").max().over(side).alias(f"q_{side}_max"),
            pl.col("q").top_k(2).min().over(side).alias(f"q_{side}_second"),
            (pl.col("q") >= 0.5).sum().over(side).cast(pl.UInt16).alias(f"q_{side}_n50"),
            pl.col("q").rank("ordinal", descending=True).over(side).cast(pl.UInt16).alias(f"q_{side}_rank"),
        ).with_columns((pl.col("q") / pl.col(f"q_{side}_max")).alias(f"q_{side}_ratio"))
    return probs


def fit(out_dir: str, rounds: int, threads: int) -> None:
    """Write out-of-fold refit probabilities for the training folds, then train the second model on them."""
    folds = [k for k in range(train.FOLDS) if k != train.HOLDOUT]
    frame = refine.band_features("train", out_dir, folds).with_columns(fold=train.fold())
    columns = [c for c in frame.columns if c not in [*train.NON_FEATURES, "fold"]]
    x, y, fold, keys = train.matrix(frame, columns), frame["label"].to_numpy(), frame["fold"].to_numpy(), frame.select("s1", "s23")
    del frame
    q = np.zeros(len(y), dtype=np.float32)
    for k in folds:
        rest = fold != k
        booster = lgb.train(refine.PARAMS | {"num_threads": threads}, lgb.Dataset(x[rest], y[rest], feature_name=columns, free_raw_data=True), rounds)
        q[~rest] = booster.predict(x[~rest], num_threads=threads)
        print(f"out-of-fold refit for fold {k} done", flush=True)
    keys.with_columns(q=pl.Series(q)).write_parquet(oof_path(out_dir))
    extra = context("train", out_dir)
    names = [c for c in extra.columns if c not in ("s1", "s23")]
    x = np.hstack([x, keys.join(extra, on=["s1", "s23"], how="left", maintain_order="left").select(names).to_numpy().astype(np.float32)])
    booster = lgb.train(refine.PARAMS | {"num_threads": threads}, lgb.Dataset(x, y, feature_name=columns + names, free_raw_data=True), rounds)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(MODEL_PATH)
    gain = booster.feature_importance("gain")
    top = sorted(zip(columns + names, gain / gain.sum(), strict=True), key=lambda item: -item[1])[:12]
    print("gain share: " + ", ".join(f"{name}={share:.3f}" for name, share in top))


def predict(split: str, out_dir: str, threads: int) -> None:
    """Score the band pairs of the hold-out fold (train) or of test and write ``refine2_000.parquet``."""
    frame = refine.band_features(split, out_dir, [train.HOLDOUT] if split == "train" else None)
    frame = frame.join(context(split, out_dir), on=["s1", "s23"], how="left", maintain_order="left")
    booster = lgb.Booster(model_file=MODEL_PATH)
    p = refine.keep_base(split, frame, booster.predict(train.matrix(frame, booster.feature_name()), num_threads=threads).astype(np.float32))
    frame.select("s1", "s23").with_columns(p=pl.Series(p)).write_parquet(stack.prob_path(split, 0, out_dir, PREFIX))
    print(f"{split}: {frame.height} band pairs scored, {(p >= 0.5).sum()} at p >= 0.5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["fit", "predict"])
    parser.add_argument("split", nargs="?", default="train", choices=["train", "test"])
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out-dir", default=features.FEATURE_DIR)
    arguments = parser.parse_args()
    if arguments.step == "fit":
        fit(arguments.out_dir, arguments.rounds, arguments.threads)
    else:
        predict(arguments.split, arguments.out_dir, arguments.threads)
