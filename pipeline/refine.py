"""Final refit of candidate pairs with token-rarity (IDF) overlap of names and addresses.

The pair features compare names and addresses without weighting tokens by rarity, so a pair
that shares only common words scores like one that shares a distinctive word. This stage
takes every pair of ``BASE`` (blocking, second-, third- and fourth-round probabilities) with
``p >= BAND`` and refits it with:

- the probability, its candidate round, and per-S1 and per-S2/S3 summaries (best, second
  best, count at 0.5, count) over all pairs of the split;
- IDF-weighted overlap of name tokens, and of address words and numbers: shared and
  per-side weight sums, weighted Jaccard and coverage, and the rarest shared and unshared
  token. IDF is per country, over the S1 and S2/S3 records of train and test together, so
  both splits use the same values;
- fuzzy name-token matches: each name token one record has and the other lacks is compared
  with the other record's unshared tokens by ``rapidfuzz.fuzz.ratio``, and the IDF weight of
  tokens matched at ``SOFT_CUT`` or more counts towards a second coverage; the same matches over address words, without the
  numbers, whose lookalike decoys differ by one digit;
- the first-round pair features in ``STRUCTURAL`` (house relation, street, name and address
  similarity, token overlaps), null for pairs first proposed by a later round.

Pairs below ``BAND`` cannot pass the rule threshold or change a maximum that does, so the
output holds band pairs only: ``--pred refine``. One model is trained on the out-of-fold
probabilities of every fold except the hold-out and scores the hold-out and test. Test
pairs of countries that train does not have (France) keep their base probability: fitted on
one train country and applied to the other, the refit scores 0.007–0.012 below the base
probabilities.

Steps, each run from the repository root under ``tools/run_guarded.sh``:

- ``python -m pipeline.refine fit``
- ``python -m pipeline.refine predict train`` (hold-out fold) and ``predict test``: writes
  ``data/features/{split}/refine_000.parquet``
- ``python -m pipeline.train tune --pred refine``
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from pipeline import features, records, stack, train

MODEL_PATH = "data/models/refine_lgbm.txt"
PREFIX = "refine"
BASE = "blend+sibling+namepair+reverse"
BAND = 0.02
KINDS = ("name", "address")
SOFT_CUT = 80.0
STRUCTURAL = ("core_eq", "house_eq", "house_in", "house_in_reverse", "house_missing_l", "house_missing_r", "house_diff",
              "street_eq", "name_ratio", "core_tset", "addr_ratio", "addr_tset", "token_overlap", "number_overlap", "word_overlap")
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_bin": 255,
    "verbose": -1,
}


def rounds_scan(split: str, out_dir: str) -> pl.LazyFrame:
    """Return (``s1``, ``s23``, ``p``, ``round``) over the probability files of ``BASE``, one round per prefix."""
    return pl.concat([
        pl.scan_parquet(f"{out_dir}{split}/{prefix}_*.parquet").select("s1", "s23", pl.col("p").cast(pl.Float32), round=pl.lit(i, pl.UInt8))
        for i, prefix in enumerate(BASE.split("+"))
    ])


def side_context(split: str, out_dir: str, side: str) -> pl.DataFrame:
    """Return the best, second-best, count at 0.5 and count of probabilities per ``side`` over every pair of ``split``."""
    return rounds_scan(split, out_dir).group_by(side).agg(
        pl.col("p").max().alias(f"{side}_max"),
        pl.col("p").top_k(2).min().alias(f"{side}_second"),
        (pl.col("p") >= 0.5).sum().cast(pl.UInt16).alias(f"{side}_n50"),
        pl.len().cast(pl.UInt32).alias(f"{side}_n"),
    ).with_columns(pl.when(pl.col(f"{side}_n") > 1).then(pl.col(f"{side}_second")).otherwise(0.0).alias(f"{side}_second")).collect(engine="streaming")


def record_tokens(split: str, side: str, kind: str) -> pl.LazyFrame:
    """Return distinct (``idx``, ``country``, ``token``) of one side: name tokens, or address words and ``#``-prefixed numbers."""
    if kind == "name":
        return pl.scan_parquet(records.tokens_path(split, side)).select("idx", "country", "token").unique(["idx", "token"])
    frame = pl.scan_parquet(records.records_path(split, side)).select(
        "idx", "country", token=pl.concat_list(pl.col("words").fill_null([]), pl.col("numbers").fill_null([]).list.eval("#" + pl.element()))
    )
    return frame.explode("token", empty_as_null=True).drop_nulls("token").unique(["idx", "token"])


def idf(kind: str) -> pl.DataFrame:
    """Return (``country``, ``token``, ``idf``) with ``log((n + 1) / (df + 1))`` over the records of both splits."""
    counts, sizes = [], []
    for split in ("train", "test"):
        for side in ("s1", "s23"):
            counts.append(record_tokens(split, side, kind).group_by("country", "token").agg(df=pl.len()).collect(engine="streaming"))
            sizes.append(pl.scan_parquet(records.records_path(split, side)).group_by("country").agg(n=pl.len()).collect())
    df = pl.concat(counts).group_by("country", "token").agg(pl.col("df").sum())
    n = pl.concat(sizes).group_by("country").agg(pl.col("n").sum())
    return df.join(n, on="country").select("country", "token", idf=((pl.col("n") + 1) / (pl.col("df") + 1)).log().cast(pl.Float32))


def weighted_tokens(split: str, side: str, kind: str, weights: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """Return (``side``, ``token``, ``idf``) for the records of ``side`` in ``ids``."""
    tokens = record_tokens(split, side, kind).join(ids.lazy().select(idx=side), on="idx", how="semi")
    return tokens.join(weights.lazy(), on=["country", "token"]).select(pl.col("idx").alias(side), "token", "idf").collect()


def overlap(split: str, pairs: pl.DataFrame, kind: str, weights: pl.DataFrame) -> pl.DataFrame:
    """Return IDF-weighted overlap features of ``kind`` for (``s1``, ``s23``) pairs."""
    keys = pairs.select("s1", "s23")
    t1 = weighted_tokens(split, "s1", kind, weights, keys.select("s1").unique())
    t23 = weighted_tokens(split, "s23", kind, weights, keys.select("s23").unique())
    left = keys.join(t1, on="s1").join(t23.select("s23", "token", shared=pl.lit(True)), on=["s23", "token"], how="left")
    right = keys.join(t23, on="s23").join(t1.select("s1", "token", shared=pl.lit(True)), on=["s1", "token"], how="left")
    left, right = left.with_columns(pl.col("shared").fill_null(False)), right.with_columns(pl.col("shared").fill_null(False))
    a = left.group_by("s1", "s23").agg(
        sum1=pl.col("idf").sum(),
        shared=pl.col("idf").filter("shared").sum(),
        miss1=pl.col("idf").filter(~pl.col("shared")).max(),
        shared_max=pl.col("idf").filter("shared").max(),
    )
    b = right.group_by("s1", "s23").agg(sum2=pl.col("idf").sum(), miss2=pl.col("idf").filter(~pl.col("shared")).max())
    out = keys.join(a, on=["s1", "s23"], how="left").join(b, on=["s1", "s23"], how="left").with_columns(pl.col("sum1", "sum2", "shared").fill_null(0.0))
    out = out.with_columns(
        jacc=pl.col("shared") / (pl.col("sum1") + pl.col("sum2") - pl.col("shared")),
        cov1=pl.col("shared") / pl.col("sum1"),
        cov2=pl.col("shared") / pl.col("sum2"),
    )
    return out.select("s1", "s23", pl.exclude("s1", "s23").name.prefix(f"{kind}_idf_"))


def unshared(keys: pl.DataFrame, own: pl.DataFrame, other: pl.DataFrame, side: str, other_side: str) -> pl.DataFrame:
    """Return (``s1``, ``s23``, ``token``, ``idf``) for tokens of ``side`` that the pair's other record lacks."""
    return keys.join(own, on=side).join(other.select(other_side, "token", shared=pl.lit(True)), on=[other_side, "token"], how="left").filter(
        pl.col("shared").is_null()
    ).drop("shared")


def soft_side(left: pl.DataFrame, right: pl.DataFrame, tag: str) -> pl.DataFrame:
    """Return per pair the IDF weight of ``left`` tokens by their best ``fuzz.ratio`` against the ``right`` tokens of the same pair."""
    cross = left.join(right.select("s1", "s23", other="token"), on=["s1", "s23"], how="left")
    ratio = cpdist(cross["token"].to_list(), cross["other"].fill_null("").to_list(), scorer=fuzz.ratio, workers=features.WORKERS)
    best = cross.select("s1", "s23", "token", "idf", ratio=pl.Series(ratio, dtype=pl.Float32)).group_by("s1", "s23", "token").agg(
        pl.col("idf").first(), pl.col("ratio").max()
    )
    return best.group_by("s1", "s23").agg(
        (pl.col("idf") * (pl.col("ratio") >= SOFT_CUT)).sum().alias(f"soft{tag}"),
        (pl.col("idf") * pl.col("ratio") / 100).sum().alias(f"softw{tag}"),
        pl.col("idf").filter(pl.col("ratio") < SOFT_CUT).max().alias(f"soft_miss{tag}"),
    )


def soft_overlap(split: str, pairs: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Return fuzzy name-token features for pairs that already carry the exact name IDF features."""
    keys = pairs.select("s1", "s23")
    t1 = weighted_tokens(split, "s1", "name", weights, keys.select("s1").unique())
    t23 = weighted_tokens(split, "s23", "name", weights, keys.select("s23").unique())
    u1, u23 = unshared(keys, t1, t23, "s1", "s23"), unshared(keys, t23, t1, "s23", "s1")
    out = keys.join(soft_side(u1, u23, "1"), on=["s1", "s23"], how="left").join(soft_side(u23, u1, "2"), on=["s1", "s23"], how="left")
    out = out.with_columns(pl.col("soft1", "softw1", "soft2", "softw2").fill_null(0.0))
    out = out.join(pairs.select("s1", "s23", "name_idf_shared", "name_idf_sum1", "name_idf_sum2"), on=["s1", "s23"]).with_columns(
        soft_cov1=(pl.col("name_idf_shared") + pl.col("soft1")) / pl.col("name_idf_sum1"),
        soft_cov2=(pl.col("name_idf_shared") + pl.col("soft2")) / pl.col("name_idf_sum2"),
    ).drop("name_idf_shared", "name_idf_sum1", "name_idf_sum2")
    return out.select("s1", "s23", pl.exclude("s1", "s23").name.prefix("name_"))


def soft_address(split: str, pairs: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Return fuzzy address-word features for the pairs; ``#``-prefixed numbers are left out."""
    keys = pairs.select("s1", "s23")
    words = weights.filter(~pl.col("token").str.starts_with("#"))
    t1 = weighted_tokens(split, "s1", "address", words, keys.select("s1").unique())
    t23 = weighted_tokens(split, "s23", "address", words, keys.select("s23").unique())
    u1, u23 = unshared(keys, t1, t23, "s1", "s23"), unshared(keys, t23, t1, "s23", "s1")
    out = keys.join(soft_side(u1, u23, "1"), on=["s1", "s23"], how="left").join(soft_side(u23, u1, "2"), on=["s1", "s23"], how="left")
    out = out.with_columns(pl.col("soft1", "softw1", "soft2", "softw2").fill_null(0.0))
    return out.select("s1", "s23", pl.exclude("s1", "s23").name.prefix("address_"))


def band_features(split: str, out_dir: str, folds: list[int] | None) -> pl.DataFrame:
    """Return the band pairs of ``split`` (train: of S1 entities in ``folds``) with context, IDF overlap and, on train, labels."""
    per_s1, per_s23 = side_context(split, out_dir, "s1"), side_context(split, out_dir, "s23")
    weights = {kind: idf(kind) for kind in KINDS}
    parts = []
    for fold in folds if folds is not None else [None]:
        pairs = rounds_scan(split, out_dir).filter(pl.col("p") >= BAND)
        if fold is not None:
            pairs = pairs.filter(train.fold() == fold)
        pairs = pairs.collect().join(per_s1, on="s1", how="left").join(per_s23, on="s23", how="left")
        for kind in KINDS:
            pairs = pairs.join(overlap(split, pairs, kind, weights[kind]), on=["s1", "s23"], how="left")
        pairs = pairs.join(soft_overlap(split, pairs, weights["name"]), on=["s1", "s23"], how="left")
        pairs = pairs.join(soft_address(split, pairs, weights["address"]), on=["s1", "s23"], how="left")
        part = (pl.scan_parquet(f"{out_dir}{split}/part_*.parquet").select("s1", "s23", *STRUCTURAL)
                .join(pairs.select("s1", "s23").lazy(), on=["s1", "s23"], how="semi").collect())
        pairs = pairs.join(part, on=["s1", "s23"], how="left")
        if split == "train":
            truth = features.truth_links().with_columns(label=pl.lit(True))
            pairs = pairs.join(truth, on=["s1", "s23"], how="left").with_columns(pl.col("label").fill_null(False))
        print(f"{split} fold {fold}: band pairs={pairs.height}", flush=True)
        parts.append(pairs)
    return pl.concat(parts)


def fit(out_dir: str, rounds: int, threads: int) -> None:
    """Train one model on the band pairs of every fold except the hold-out."""
    frame = band_features("train", out_dir, [k for k in range(train.FOLDS) if k != train.HOLDOUT])
    columns = [c for c in frame.columns if c not in train.NON_FEATURES]
    data = lgb.Dataset(train.matrix(frame, columns), frame["label"].to_numpy(), feature_name=columns, free_raw_data=True)
    print(f"train rows={frame.height} positives={frame['label'].sum()}", flush=True)
    del frame
    booster = lgb.train(PARAMS | {"num_threads": threads}, data, rounds)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(MODEL_PATH)
    gain = booster.feature_importance("gain")
    top = sorted(zip(columns, gain / gain.sum(), strict=True), key=lambda item: -item[1])[:12]
    print("gain share: " + ", ".join(f"{name}={share:.3f}" for name, share in top))


def keep_base(split: str, frame: pl.DataFrame, p: np.ndarray) -> np.ndarray:
    """Return ``p`` with the base probability ``frame["p"]`` restored for S1 entities of countries absent from train."""
    seen = pl.scan_parquet(records.records_path("train", "s1")).select("country").unique().collect()
    unseen = pl.scan_parquet(records.records_path(split, "s1")).select(s1="idx", country="country").join(seen.lazy(), on="country", how="anti").collect()
    keep = frame.select(pl.col("s1").is_in(unseen["s1"].implode())).to_series().to_numpy()
    p[keep] = frame["p"].to_numpy()[keep]
    print(f"{split}: {int(keep.sum())} band pairs of countries absent from train keep their base probability", flush=True)
    return p


def predict(split: str, out_dir: str, threads: int) -> None:
    """Score the band pairs of the hold-out fold (train) or of test and write ``refine_000.parquet``."""
    frame = band_features(split, out_dir, [train.HOLDOUT] if split == "train" else None)
    booster = lgb.Booster(model_file=MODEL_PATH)
    p = keep_base(split, frame, booster.predict(train.matrix(frame, booster.feature_name()), num_threads=threads).astype(np.float32))
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
