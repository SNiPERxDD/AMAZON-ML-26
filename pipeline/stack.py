"""Second-stage pair classifier over first-stage probabilities of related pairs.

The first stage scores each pair on its own. This stage adds how the other candidates of
the same S1 entity, and the other S1 candidates of the same S2/S3 record, were scored,
so it can learn that a pair is weaker when its S2/S3 record has a better S1 elsewhere,
or stronger when its S1 has few confident candidates.

It also compares each plausible candidate with its S1's confident candidates ("peers",
first-stage probability at least ``PEER_P``). True links of one S1 are noisy copies of the
same business, so a candidate that shares a house number or street with its peers is more
likely true; on the held-out fold this split pairs scored 0.6–0.8 into 53% and 82% true.

Run from the repository root under ``tools/run_guarded.sh``, after
``pipeline.train predict train --oof`` and ``predict test``:

- ``python -m pipeline.stack peers train`` (and ``test``), which caches ``peer_{k:03d}.parquet``
- ``python -m pipeline.stack fit --fold k`` for each fold ``k`` in 0–4
- ``python -m pipeline.stack predict train --oof`` and ``predict test``, which write
  ``stack_{k:03d}.parquet`` next to each feature part
- ``python -m pipeline.train tune --pred stack``, then ``python -m pipeline.submit --pred stack``

First-stage probabilities must be out-of-fold for train, or this stage learns from
overconfident in-sample scores that test does not have.

``--base`` and ``--name`` repeat the stage on another stage's probabilities: for example
``--base stack --name stage3`` reads ``stack_*`` files, caches ``peer_stack_*`` and writes
``stage3_*`` files and ``data/models/stage3_lgbm_f*.txt``.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import features, records, train

MODEL_DIR = "data/models/"
PEER_P = 0.5
CANDIDATE_P = 0.01
PARAMS = train.PARAMS | {"num_leaves": 63, "learning_rate": 0.05}


def prob_path(split: str, part: int, out_dir: str, prefix: str) -> str:
    """Return the path of one part's probability file with the given prefix (``pred``, ``stack``, ...)."""
    return train.pred_path(split, part, out_dir).replace("pred_", f"{prefix}_")


def pred_scan(split: str, out_dir: str, parts: list[int] | None = None, base: str = "pred") -> pl.LazyFrame:
    """Return the base stage's predictions (``s1``, ``s23``, ``p``) lazily."""
    paths = [prob_path(split, p, out_dir, base) for p in parts] if parts is not None else f"{out_dir}{split}/{base}_*.parquet"
    return pl.scan_parquet(paths).select("s1", "s23", "p")


def s23_table(split: str, out_dir: str, base: str) -> pl.DataFrame:
    """Return, per S2/S3 record, its best-scored S1, the top two probabilities, their sum and count."""
    return (
        pred_scan(split, out_dir, base=base)
        .group_by("s23")
        .agg(
            top_s1=pl.col("s1").sort_by("p", descending=True).first(),
            top_p=pl.col("p").max(),
            second_p=pl.col("p").top_k(2).get(1, null_on_oob=True),
            sum_p23=pl.col("p").sum(),
            n50_23=(pl.col("p") >= 0.5).sum().cast(pl.UInt16),
        )
        .collect(engine="streaming")
    )


def peer_path(split: str, part: int, out_dir: str, base: str) -> str:
    """Return the peer-feature path of one part (``peer_`` for first-stage peers, ``peer_{base}_`` otherwise)."""
    return prob_path(split, part, out_dir, "peer" if base == "pred" else f"peer_{base}")


def write_peers(split: str, out_dir: str, base: str) -> None:
    """Write peer-agreement features for every pair with base probability at least ``CANDIDATE_P``.

    Parts hold whole S1 entities, so each part's peers are complete within it.
    """
    records_s23 = (
        pl.scan_parquet(records.records_path(split, "s23"))
        .select("idx", "house", "street", "numbers", "words")
        .collect()
    )
    for part in parts(split, out_dir):
        preds = pred_scan(split, out_dir, [part], base).filter(pl.col("p") >= CANDIDATE_P).collect()
        peers = preds.filter(pl.col("p") >= PEER_P).select("s1", peer="s23", peer_p="p")
        pairs = (
            preds.select("s1", "s23")
            .join(peers, on="s1")
            .filter(pl.col("s23") != pl.col("peer"))
            .join(records_s23, left_on="s23", right_on="idx")
            .join(records_s23, left_on="peer", right_on="idx", suffix="_p")
            .with_columns(
                house_eq=(pl.col("house") == pl.col("house_p")).fill_null(False),
                street_eq=(pl.col("street") == pl.col("street_p")).fill_null(False),
                number_eq=pl.col("numbers").list.set_intersection("numbers_p").list.len() > 0,
                word_jaccard=pl.col("words").list.set_intersection("words_p").list.len()
                / pl.col("words").list.set_union("words_p").list.len().clip(1, None),
            )
        )
        pairs.group_by("s1", "s23").agg(
            n_peer=pl.len().cast(pl.UInt16),
            peer_house=pl.col("house_eq").sum().cast(pl.UInt16),
            peer_house_p=(pl.col("house_eq") * pl.col("peer_p")).sum(),
            peer_street=pl.col("street_eq").sum().cast(pl.UInt16),
            peer_number=pl.col("number_eq").sum().cast(pl.UInt16),
            peer_words_max=pl.col("word_jaccard").max().cast(pl.Float32),
            peer_words_mean=pl.col("word_jaccard").mean().cast(pl.Float32),
        ).with_columns(
            peer_house_share=pl.col("peer_house") / pl.col("n_peer"),
        ).write_parquet(peer_path(split, part, out_dir, base))
        print(f"part {part:03d}: {len(pairs)} peer comparisons", flush=True)


def frame(split: str, part: int, out_dir: str, s23: pl.DataFrame, base: str) -> pl.LazyFrame:
    """Return one part's first-stage features plus the base stage's probability context features."""
    per_s1 = "s1"
    return (
        features.scan(split, [part], out_dir)
        .join(pred_scan(split, out_dir, [part], base), on=["s1", "s23"], how="left")
        .join(s23.lazy(), on="s23", how="left")
        .join(pl.scan_parquet(peer_path(split, part, out_dir, base)), on=["s1", "s23"], how="left")
        .with_columns(
            p_other_s1=pl.when(pl.col("s1") == pl.col("top_s1")).then(pl.col("second_p")).otherwise(pl.col("top_p")).fill_null(0.0),
            p_other_sum=(pl.col("sum_p23") - pl.col("p")),
            p_s1_max=pl.col("p").max().over(per_s1),
            p_s1_second=pl.col("p").top_k(2).get(1, null_on_oob=True).over(per_s1),
            p_s1_sum=pl.col("p").sum().over(per_s1),
            p_s1_n50=(pl.col("p") >= 0.5).sum().over(per_s1).cast(pl.UInt16),
            p_rank=pl.col("p").rank("ordinal", descending=True).over(per_s1).cast(pl.UInt16),
            p_s2_max=pl.col("p").filter(~pl.col("is_s3")).max().over(per_s1),
            p_s3_max=pl.col("p").filter(pl.col("is_s3")).max().over(per_s1),
        )
        .with_columns(
            p_margin=pl.col("p") - pl.col("p_other_s1"),
            p_ratio=pl.col("p") / pl.col("p_s1_max"),
            p_s1_other=pl.when(pl.col("p_rank") == 1).then(pl.col("p_s1_second")).otherwise(pl.col("p_s1_max")),
        )
        .drop("top_s1", "top_p", "second_p", "sum_p23")
    )


def parts(split: str, out_dir: str) -> list[int]:
    """Return the part numbers of ``split``."""
    return sorted(int(path.stem.split("_")[1]) for path in Path(f"{out_dir}{split}").glob("part_*.parquet"))


def collect(
    split: str, out_dir: str, s23: pl.DataFrame, base: str, keep: pl.Expr, columns: list[str] | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect the rows selected by ``keep`` from every part as a matrix and labels."""
    blocks, labels = [], []
    for part in parts(split, out_dir):
        block = frame(split, part, out_dir, s23, base).filter(keep).collect()
        columns = columns or [c for c in block.columns if c not in train.NON_FEATURES]
        blocks.append(train.matrix(block, columns))
        labels.append(block["label"].to_numpy())
    return np.concatenate(blocks), np.concatenate(labels), columns


def model_path_for(name: str, held_out: int) -> str:
    """Return the path of stage ``name``'s model that left out fold ``held_out``."""
    return f"{MODEL_DIR}{name}_lgbm_f{held_out}.txt"


def fit(held_out: int, train_modulus: int, valid_modulus: int, rounds: int, out_dir: str, threads: int, base: str, name: str) -> None:
    """Train on sampled S1 entities outside fold ``held_out``, early-stopping on a sample of that fold."""
    s23 = s23_table("train", out_dir, base)
    x_train, y_train, columns = collect("train", out_dir, s23, base, (train.fold() != held_out) & train.sampled(train_modulus), None)
    x_valid, y_valid, _ = collect("train", out_dir, s23, base, (train.fold() == held_out) & train.sampled(valid_modulus), columns)
    del s23
    print(f"train rows={len(y_train)} positives={y_train.mean():.3f}; valid rows={len(y_valid)}; features={len(columns)}", flush=True)
    train_set = lgb.Dataset(x_train, y_train, feature_name=columns, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set)
    booster = lgb.train(
        PARAMS | {"num_threads": threads}, train_set, num_boost_round=rounds, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path_for(name, held_out), num_iteration=booster.best_iteration)
    gain = sorted(zip(columns, booster.feature_importance("gain")), key=lambda item: -item[1])
    total = sum(g for _, g in gain)
    print(f"best iteration={booster.best_iteration} valid={dict(booster.best_score['valid_0'])}")
    print("gain share: " + ", ".join(f"{name}={g / total:.3f}" for name, g in gain[:15]))


def predict(split: str, out_dir: str, oof: bool, threads: int, base: str, name: str) -> None:
    """Score every part and write ``{name}_{k:03d}.parquet``; with ``oof``, each S1 by the model that left out its fold."""
    folds = range(train.FOLDS) if oof else [train.HOLDOUT]
    boosters = {k: lgb.Booster(model_file=model_path_for(name, k)) for k in folds}
    columns = boosters[train.HOLDOUT].feature_name()
    s23 = s23_table(split, out_dir, base)
    for part in parts(split, out_dir):
        block = frame(split, part, out_dir, s23, base).collect()
        fold_of = block.select(train.fold()).to_series().to_numpy() if oof else np.zeros(len(block), dtype=np.uint64)
        p = np.empty(len(block), dtype=np.float32)
        for k, booster in boosters.items():
            rows = np.flatnonzero(fold_of == k)
            if len(rows):
                p[rows] = booster.predict(train.matrix(block[rows], columns), num_threads=threads)
        block.select("s1", "s23").with_columns(p=pl.Series(p)).write_parquet(prob_path(split, part, out_dir, name))
        print(f"part {part:03d}: {len(block)} pairs scored", flush=True)


def blend(split: str, out_dir: str, inputs: list[str], name: str) -> None:
    """Write ``{name}_{k:03d}.parquet`` holding the mean probability of the ``inputs`` stages for every pair."""
    for part in parts(split, out_dir):
        frames = [pl.read_parquet(prob_path(split, part, out_dir, prefix)).select("s1", "s23", pl.col("p").alias(f"p{i}")) for i, prefix in enumerate(inputs)]
        joined = frames[0]
        for extra in frames[1:]:
            joined = joined.join(extra, on=["s1", "s23"], how="inner")
        mean = pl.mean_horizontal([f"p{i}" for i in range(len(inputs))]).cast(pl.Float32).alias("p")
        joined.select("s1", "s23", mean).write_parquet(prob_path(split, part, out_dir, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["peers", "fit", "predict", "blend"])
    parser.add_argument("split", nargs="?", default="train", choices=["train", "test"])
    parser.add_argument("--fold", type=int, default=train.HOLDOUT, help="fit: the fold to leave out")
    parser.add_argument("--oof", action="store_true", help="predict: score each S1 with the model that left out its fold")
    parser.add_argument("--train-modulus", type=int, default=8, help="train on one in N S1 entities outside the fold")
    parser.add_argument("--valid-modulus", type=int, default=8, help="early-stop on one in N S1 entities of the fold")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out-dir", default=features.FEATURE_DIR)
    parser.add_argument("--base", default="pred", help="prefix of the probability files this stage builds on")
    parser.add_argument("--name", default="stack", help="prefix of this stage's models and probability files")
    parser.add_argument("--inputs", default="stack,stage3", help="blend: comma-separated probability prefixes to average")
    arguments = parser.parse_args()
    if arguments.step == "peers":
        write_peers(arguments.split, arguments.out_dir, arguments.base)
    elif arguments.step == "fit":
        fit(arguments.fold, arguments.train_modulus, arguments.valid_modulus, arguments.rounds, arguments.out_dir, arguments.threads, arguments.base, arguments.name)
    elif arguments.step == "blend":
        blend(arguments.split, arguments.out_dir, arguments.inputs.split(","), arguments.name)
    else:
        predict(arguments.split, arguments.out_dir, arguments.oof, arguments.threads, arguments.base, arguments.name)
