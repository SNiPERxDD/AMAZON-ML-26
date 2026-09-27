"""Train the pair classifier, score candidate pairs, and tune the per-S1 decision rule.

Run each step from the repository root under ``tools/run_guarded.sh``:

- ``python -m pipeline.train fit --fold k`` trains LightGBM on a hashed sample of S1 entities
  outside fold ``k``. Early stopping uses a sample of fold ``k``. Fold 0 is the held-out fold.
- ``python -m pipeline.train predict train --oof`` scores each S1 entity with the model that
  did not see its fold. ``predict test`` uses the fold-0 model, so test predictions come from
  the same model as the held-out ones. Both write ``pred_{k:03d}.parquet`` next to each part.
- ``python -m pipeline.train tune`` scores decision rules on the held-out fold with macro
  F0.5, counting every true link, including links candidate generation missed.

The held-out fold is ``s1.hash(seed=3) % 5 == 0``, a random 20% of S1 entities.
Without ``--oof``, predictions for S1 entities in folds 1–4 are in-sample; they are used as
competitors in the partition rule, where they are probably overconfident.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import features, metric, records

FOLDS = 5
FOLD_SEED = 3
HOLDOUT = 0
SAMPLE_SEED = 11
MODEL_PATH = "data/models/pair_lgbm.txt"
RULE_PATH = "data/models/rule.json"
NON_FEATURES = ["s1", "s23", "label", "country"]
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "num_threads": 4,
    "metric": ["binary_logloss", "auc"],
    "verbose": -1,
}


def fold() -> pl.Expr:
    """Return the fold of each S1 entity."""
    return pl.col("s1").hash(seed=FOLD_SEED) % FOLDS


def sampled(modulus: int) -> pl.Expr:
    """Keep one in ``modulus`` S1 entities, chosen by hash."""
    return pl.col("s1").hash(seed=SAMPLE_SEED) % modulus == 0


def matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the feature columns as a ``float32`` matrix with nulls as NaN."""
    return frame.select(pl.col(columns).cast(pl.Float32)).to_numpy()


def model_path_for(model_path: str, held_out: int) -> str:
    """Return the model path for the model that left out fold ``held_out`` (fold 0 keeps the base name)."""
    return model_path if held_out == HOLDOUT else model_path.replace(".txt", f"_f{held_out}.txt")


def pred_path(split: str, part: int, out_dir: str) -> str:
    """Return the prediction path of one feature part."""
    return features.part_path(split, part, out_dir).replace("part_", "pred_")


def fit(train_modulus: int, valid_modulus: int, rounds: int, out_dir: str, model_path: str, held_out: int = HOLDOUT, threads: int = 4) -> None:
    """Train on sampled S1 entities outside fold ``held_out``, early-stopping on a sample of that fold."""
    model_path = model_path_for(model_path, held_out)
    base = features.scan("train", out_dir=out_dir)
    columns = [c for c in base.collect_schema().names() if c not in NON_FEATURES]
    train = base.filter((fold() != held_out) & sampled(train_modulus)).select(*columns, "label").collect()
    x_train, y_train = matrix(train, columns), train["label"].to_numpy()
    del train
    valid = base.filter((fold() == held_out) & sampled(valid_modulus)).select(*columns, "label").collect()
    x_valid, y_valid = matrix(valid, columns), valid["label"].to_numpy()
    del valid
    print(f"train rows={len(y_train)} positives={y_train.mean():.3f}; valid rows={len(y_valid)}; features={len(columns)}", flush=True)
    train_set = lgb.Dataset(x_train, y_train, feature_name=columns, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set)
    booster = lgb.train(
        PARAMS | {"num_threads": threads}, train_set, num_boost_round=rounds, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    gain = sorted(zip(columns, booster.feature_importance("gain")), key=lambda item: -item[1])
    total = sum(g for _, g in gain)
    print(f"best iteration={booster.best_iteration} valid={dict(booster.best_score['valid_0'])}")
    print("gain share: " + ", ".join(f"{name}={g / total:.3f}" for name, g in gain[:15]))


def predict(split: str, out_dir: str, model_path: str, resume: bool = False, threads: int = 4, oof: bool = False) -> None:
    """Score every feature part of ``split`` and write the probabilities.

    With ``oof``, each S1 entity is scored by the model that left out its fold. With
    ``resume``, parts that already have a prediction file are skipped; use it only when
    the models have not changed since those files were written.
    """
    folds = range(FOLDS) if oof else [HOLDOUT]
    boosters = {k: lgb.Booster(model_file=model_path_for(model_path, k)) for k in folds}
    columns = boosters[HOLDOUT].feature_name()
    parts = sorted(Path(f"{out_dir}{split}").glob("part_*.parquet"))
    for path in parts:
        part = int(path.stem.split("_")[1])
        if resume and Path(pred_path(split, part, out_dir)).exists():
            continue
        frame = features.scan(split, [part], out_dir).collect()
        keep = ["s1", "s23"] + (["label"] if "label" in frame.columns else [])
        fold_of = frame.select(fold()).to_series().to_numpy() if oof else np.zeros(len(frame), dtype=np.uint64)
        p = np.empty(len(frame), dtype=np.float32)
        for k, booster in boosters.items():
            rows = np.flatnonzero(fold_of == k)
            if len(rows):
                p[rows] = booster.predict(matrix(frame[rows], columns), num_threads=threads)
        frame.select(keep).with_columns(p=pl.Series(p)).write_parquet(pred_path(split, part, out_dir))
        print(f"{path.name}: {len(frame)} pairs scored", flush=True)


def decide(preds: pl.DataFrame, threshold: float, relative: float, partition: bool) -> pl.DataFrame:
    """Return the kept links under one decision rule.

    A pair is kept when its probability reaches ``threshold`` and ``relative`` times the
    best probability of its S1 entity. With ``partition``, it must also be the best S1 of
    its S2/S3 record, because each S2/S3 record belongs to at most one S1 entity.
    """
    kept = preds.filter((pl.col("p") >= threshold) & (pl.col("p") >= relative * pl.col("p_s1_max")))
    if partition:
        kept = kept.filter(pl.col("p") >= pl.col("p_s23_max"))
    return kept.select("s1", "s23")


def prob_scan(split: str, out_dir: str, pred: str) -> pl.LazyFrame:
    """Return (``s1``, ``s23``, ``p``) from the probability files named by ``pred``.

    ``pred`` is one prefix (``pred``, ``stack``, ``blend``, ...) or several joined by ``+`` for
    disjoint pair sets, such as ``blend+sibling``.
    """
    return pl.concat([pl.scan_parquet(f"{out_dir}{split}/{prefix}_*.parquet").select("s1", "s23", "p") for prefix in pred.split("+")])


def with_maxima(preds: pl.LazyFrame) -> pl.LazyFrame:
    """Add each pair's best probability per S1 and per S2/S3 record."""
    return preds.with_columns(p_s1_max=pl.col("p").max().over("s1"), p_s23_max=pl.col("p").max().over("s23"))


def tune(out_dir: str, rule_path: str, pred: str = "pred") -> None:
    """Grid-search the decision rule on the held-out fold and save the best one.

    ``pred`` names the probability files, as in :func:`prob_scan`.
    """
    preds = with_maxima(prob_scan("train", out_dir, pred))
    scored_s1 = pl.scan_parquet(f"{out_dir}train/part_*.parquet").select("s1").unique()
    holdout_s1 = scored_s1.filter(fold() == HOLDOUT).collect()
    preds = preds.join(holdout_s1.lazy(), on="s1", how="semi").collect()
    truth = features.truth_links().join(holdout_s1, on="s1", how="semi")
    s1_ids = pl.scan_parquet(records.records_path("train", "s1")).select(s1="idx").filter(fold() == HOLDOUT)
    s1_ids = s1_ids.join(holdout_s1.lazy(), on="s1", how="semi").collect()
    print(f"holdout S1={len(s1_ids)} pairs={len(preds)} true links={len(truth)}", flush=True)
    results = []
    for partition in (False, True):
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            for relative in (0.0, 0.3, 0.5, 0.7):
                score = metric.macro_f05(decide(preds, threshold, relative, partition), truth, s1_ids)
                results.append({"partition": partition, "threshold": threshold, "relative": relative, "f05": round(score, 4)})
    table = pl.DataFrame(results).sort("f05", descending=True)
    print(table.head(8))
    best = table.row(0, named=True) | {"pred": pred}
    Path(rule_path).parent.mkdir(parents=True, exist_ok=True)
    Path(rule_path).write_text(json.dumps(best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["fit", "predict", "tune"])
    parser.add_argument("split", nargs="?", default="train", choices=["train", "test"])
    parser.add_argument("--train-modulus", type=int, default=4, help="train on one in N training-fold S1 entities")
    parser.add_argument("--valid-modulus", type=int, default=8, help="early-stop on one in N held-out S1 entities")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="predict: skip parts that already have predictions")
    parser.add_argument("--threads", type=int, default=4, help="LightGBM threads (the machine has 4 performance and 4 efficiency cores)")
    parser.add_argument("--fold", type=int, default=HOLDOUT, help="fit: the fold to leave out")
    parser.add_argument("--oof", action="store_true", help="predict: score each S1 with the model that left out its fold")
    parser.add_argument("--out-dir", default=features.FEATURE_DIR)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--rule", default=RULE_PATH)
    parser.add_argument("--pred", default="pred", help="tune: prefix of the probability files (pred, stack, blend+sibling, ...)")
    arguments = parser.parse_args()
    if arguments.step == "fit":
        fit(arguments.train_modulus, arguments.valid_modulus, arguments.rounds, arguments.out_dir, arguments.model, arguments.fold, arguments.threads)
    elif arguments.step == "predict":
        predict(arguments.split, arguments.out_dir, arguments.model, arguments.resume, arguments.threads, arguments.oof)
    else:
        tune(arguments.out_dir, arguments.rule, arguments.pred)
