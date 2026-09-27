"""Fourth-round candidates: reverse trigram retrieval from unmatched S2/S3 records to S1 entities.

``stages/reverse_frontier.py`` found that, for train S2/S3 records with an address whose
best base probability is below 0.5, the S1 entity with the highest ``name+address`` trigram
cosine (rank 1) is a true link for 0.8–2.3% of records, and that the out-of-fold stage-1 model
separates those pairs with AUC 0.999. This round proposes those rank-1 pairs, scores them with
the stage-1 model (``p1``, out of fold on train), keeps the pairs with ``p1 >= PREFILTER``, and
fits its own classifier on them. Probabilities are written in the same format as the other
rounds (``--pred blend+sibling+namepair+reverse``).

Steps, each run from the repository root under ``tools/run_guarded.sh``:

- ``python -m pipeline.reverse pairs train`` (and ``test``): S2/S3 records of every country with
  an address and a best probability over ``--base`` below ``GATE``; rank-1 S1 entity of the same
  country by the cosine of binary IDF ``name+address`` trigrams (trigrams in more than
  ``DF_CAP`` S1 records are left out); pairs already proposed by blocking, siblings or name
  pairs are dropped. Writes ``cos1`` (the pair's cosine), ``cos2`` (the second-best S1 cosine)
  and ``gap``.
- ``python -m pipeline.reverse features train`` (and ``test``): the first-round pair features.
  Per-S1 counts, ranks and gaps are computed over the S1 entity's blocking pairs together with
  its reverse pairs, and the S2/S3 context counts the reverse pair as one more candidate.
  ``p1`` is the stage-1 probability with no candidate family (``n_families`` 0, all ``f_*``
  false), from the model that left out the S1 entity's fold on train and from the hold-out
  model on test. Writes ``p1`` for every pair and features of the pairs at ``p1 >= PREFILTER``,
  with base-probability context as in :mod:`pipeline.namepairs` and labels on train.
- ``python -m pipeline.reverse fit --fold k`` (k = 0–4), ``predict train --oof``, ``predict test``:
  writes ``data/features/{split}/reverse_000.parquet`` for the countries in ``SCORED``.
  ``f_*`` and ``n_families`` are constant here and are not model features.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp

from pipeline import features, namepairs, records, siblings, stack, train

REVERSE_DIR = "data/reverse/"
MODEL_DIR = "data/models/"
PREFIX = "reverse"
GATE = 0.5
PREFILTER = 0.5
DF_CAP = 5000
CHUNK_Q = 1000
CHUNK_BUILD = 200_000
CHUNK_S1 = 400_000
SCORED = ("US", "India")
TEXT = pl.concat_str([pl.col("core"), pl.col("words").list.join(" "), pl.col("numbers").list.join(" ")], separator=" ")
WINDOW = ("name_ratio", "raw_name_ratio", "addr_tset", "strong", "core_eq")
PARAMS = siblings.PARAMS | {"num_leaves": 15, "min_data_in_leaf": 50}


def pairs_path(split: str) -> str:
    """Return the reverse pair file of ``split``."""
    return f"{REVERSE_DIR}{split}_pairs.parquet"


def p1_path(split: str) -> str:
    """Return the stage-1 probability file of every reverse pair of ``split``."""
    return f"{REVERSE_DIR}{split}_p1.parquet"


def part_path(split: str, part: int) -> str:
    """Return one reverse feature part."""
    return f"{REVERSE_DIR}{split}/part_{part:03d}.parquet"


def trigrams(frame: pl.DataFrame) -> pl.DataFrame:
    """Return the distinct (``idx``, ``tri``) word-bounded character trigrams of ``TEXT``."""
    tokens = frame.select("idx", tok=TEXT.str.split(" ")).explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    tokens = tokens.with_columns(tok=pl.concat_str([pl.lit(" "), pl.col("tok"), pl.lit(" ")]))
    tokens = tokens.with_columns(off=pl.int_ranges(0, pl.col("tok").str.len_chars() - 2)).explode("off").drop_nulls("off")
    return tokens.select("idx", tri=pl.col("tok").str.slice(pl.col("off"), 3)).unique()


def matrix(tri: pl.DataFrame, rows: pl.DataFrame, vocab: pl.DataFrame) -> sp.csr_matrix:
    """Return the L2-normalised binary IDF matrix of ``tri`` with rows ordered as ``rows`` and columns kept by ``vocab``."""
    t = tri.join(rows.with_row_index("r"), on="idx").join(vocab, on="tri")
    t = t.with_columns(norm=(pl.col("idf") ** 2).sum().over("r").sqrt()).filter(pl.col("keep"))
    return sp.csr_matrix(((t["idf"] / t["norm"]).to_numpy(), (t["r"].to_numpy(), t["col"].to_numpy())), shape=(rows.height, vocab.height))


def top2(q: sp.csr_matrix, s1t: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per query row the S1 row position of the highest cosine (-1 when none), that cosine, and the second-highest cosine."""
    prod = (q @ s1t).tocsr()
    rows = np.flatnonzero(np.diff(prod.indptr))
    best, cos1, cos2 = np.full(q.shape[0], -1), np.zeros(q.shape[0]), np.zeros(q.shape[0])
    if len(rows) == 0:
        return best, cos1, cos2
    starts = prod.indptr[rows]
    row_of = np.repeat(np.arange(q.shape[0]), np.diff(prod.indptr))
    cos1[rows] = np.maximum.reduceat(prod.data, starts)
    at_max = np.flatnonzero(prod.data == cos1[row_of])
    first = at_max[np.unique(row_of[at_max], return_index=True)[1]]
    best[row_of[first]] = prod.indices[first]
    prod.data[first] = 0.0
    cos2[rows] = np.maximum.reduceat(prod.data, starts)
    return best, cos1, cos2


def retrieve(c1: pl.DataFrame, queries: pl.DataFrame, threads: int) -> pl.DataFrame:
    """Return (``s23``, ``s1``, ``cos1``, ``cos2``): the rank-1 S1 entity of ``c1`` for each query record."""
    t1 = trigrams(c1)
    vocab = t1.group_by("tri").agg(df=pl.len()).with_columns(idf=(np.log((c1.height + 1) / (pl.col("df") + 1)) + 1), keep=pl.col("df") <= DF_CAP)
    vocab = vocab.sort("tri").with_row_index("col")
    s1t = matrix(t1, c1.select("idx"), vocab).T.tocsr()
    del t1
    out = []
    with ThreadPoolExecutor(threads) as pool:
        for low in range(0, queries.height, CHUNK_BUILD):
            block = queries.slice(low, CHUNK_BUILD)
            q = matrix(trigrams(block), block.select("idx"), vocab)
            out += pool.map(lambda start, q=q: top2(q[start:start + CHUNK_Q], s1t), range(0, q.shape[0], CHUNK_Q))
    best, cos1, cos2 = (np.concatenate([o[i] for o in out]) for i in range(3))
    found = best >= 0
    return pl.DataFrame({
        "s23": queries["idx"].to_numpy()[found],
        "s1": c1["idx"].to_numpy()[best[found]],
        "cos1": cos1[found].astype(np.float32),
        "cos2": cos2[found].astype(np.float32),
    }).with_columns(pl.col("s1", "s23").cast(pl.UInt32))


def write_pairs(split: str, out_dir: str, base: str, threads: int) -> None:
    """Write the new rank-1 reverse pairs of gated S2/S3 records, per country."""
    cols = ["idx", "country", "core", "words", "numbers"]
    s1 = pl.read_parquet(records.records_path(split, "s1"), columns=cols)
    s23 = pl.read_parquet(records.records_path(split, "s23"), columns=cols)
    best = train.prob_scan(split, out_dir, base).group_by("s23").agg(best=pl.col("p").max()).collect(engine="streaming")
    s23 = s23.filter((pl.col("words").list.len() > 0) | (pl.col("numbers").list.len() > 0))
    s23 = s23.join(best, left_on="idx", right_on="s23", how="left").filter(pl.col("best").fill_null(0.0) < GATE).drop("best")
    found = []
    for country in s1["country"].unique().sort():
        queries = s23.filter(pl.col("country") == country)
        found.append(retrieve(s1.filter(pl.col("country") == country), queries, threads).with_columns(country=pl.lit(country)))
        print(f"{split} {country}: gated S2/S3 {queries.height}, retrieved {found[-1].height}", flush=True)
    found = pl.concat(found).with_columns(gap=pl.col("cos1") - pl.col("cos2"))
    scored = pl.concat([
        pl.scan_parquet(f"{records.CACHE_DIR}{split}_pairs.parquet").select("s1", "s23"),
        pl.scan_parquet(siblings.pairs_path(split)).select("s1", "s23"),
        pl.scan_parquet(namepairs.pairs_path(split)).select("s1", "s23"),
    ])
    found = found.lazy().join(scored, on=["s1", "s23"], how="anti").collect().sort("s1", "s23")
    Path(REVERSE_DIR).mkdir(parents=True, exist_ok=True)
    found.write_parquet(pairs_path(split))
    print(f"{split}: reverse pairs={found.height}, per country {sorted(found.group_by('country').len().rows())}", flush=True)


def with_window(frame: pl.DataFrame, split: str, out_dir: str) -> pl.DataFrame:
    """Return ``frame`` with the per-S1 window features of :func:`features.pair_features` over its pairs and the S1 entities' blocking pairs."""
    block = (pl.scan_parquet(f"{out_dir}{split}/part_*.parquet").select("s1", *WINDOW)
             .join(frame.lazy().select("s1").unique(), on="s1", how="semi").collect())
    union = pl.concat([frame.select("s1", *WINDOW).with_row_index("row"), block.with_columns(row=pl.lit(None, pl.UInt32))], how="diagonal_relaxed")
    per_s1 = pl.col("s1")
    union = union.with_columns(
        s1_candidates=pl.len().over(per_s1).cast(pl.UInt16),
        s1_strong=pl.col("strong").sum().over(per_s1).cast(pl.UInt16),
        s1_hard=(pl.col("name_ratio") >= features.STRONG_RATIO).sum().over(per_s1).cast(pl.UInt16),
        s1_core_eq=pl.col("core_eq").sum().over(per_s1).cast(pl.UInt16),
        name_rank=pl.col("name_ratio").rank("min", descending=True).over(per_s1).cast(pl.UInt16),
        name_gap=(pl.col("name_ratio").max().over(per_s1) - pl.col("name_ratio")),
        raw_name_rank=pl.col("raw_name_ratio").rank("min", descending=True).over(per_s1).cast(pl.UInt16),
        addr_rank=pl.col("addr_tset").rank("min", descending=True).over(per_s1).cast(pl.UInt16),
        addr_gap=(pl.col("addr_tset").max().over(per_s1) - pl.col("addr_tset")),
    ).filter(pl.col("row").is_not_null()).sort("row")
    window = [c for c in union.columns if c not in ("row", "s1", *WINDOW)]
    return frame.drop(window).hstack(union.select(window))


def with_context(frame: pl.DataFrame, split: str, out_dir: str) -> pl.DataFrame:
    """Return ``frame`` with the S2/S3 competitor features of :func:`features.scan`, counting each reverse pair as one more candidate."""
    context = pl.read_parquet(features.context_path(split, out_dir)).drop("top_s1", "second_name")
    return frame.join(context, on="s23", how="left").with_columns(
        s23_candidates=(pl.col("s23_candidates").fill_null(0) + 1).cast(pl.UInt16),
        other_best_name=pl.col("top_name"),
        other_strong=pl.col("s23_strong").fill_null(0).cast(pl.UInt16),
    ).with_columns(other_name_gap=pl.col("name_ratio") - pl.col("other_best_name")).drop("top_name", "s23_strong")


def stage1(frame: pl.DataFrame, split: str, threads: int) -> np.ndarray:
    """Return the stage-1 probability of each pair: out of fold by S1 entity on train, from the hold-out model on test."""
    folds = range(train.FOLDS) if split == "train" else [train.HOLDOUT]
    fold = frame.select(train.fold()).to_series().to_numpy() if split == "train" else np.full(frame.height, train.HOLDOUT)
    p = np.zeros(frame.height, dtype=np.float32)
    for k in folds:
        booster = lgb.Booster(model_file=train.model_path_for(train.MODEL_PATH, k))
        rows = np.flatnonzero(fold == k)
        if len(rows):
            p[rows] = booster.predict(train.matrix(frame[rows], booster.feature_name()), num_threads=threads)
    return p


def write_features(split: str, out_dir: str, base: str, threads: int) -> None:
    """Write ``p1`` for every reverse pair and, in S1 chunks, the features of pairs at ``p1 >= PREFILTER``, with labels on train."""
    pairs_all = pl.read_parquet(pairs_path(split))
    flags = [c for c in pl.scan_parquet(f"{out_dir}{split}/part_000.parquet").collect_schema().names() if c.startswith("f_")]
    per_s1, per_s23 = namepairs.base_context(split, out_dir, base)
    n_s2 = pl.scan_parquet(f"{records.PARQUET_DIR}{split}_source2.parquet").select(pl.len()).collect().item()
    truth = features.truth_links().with_columns(label=pl.lit(True)) if split == "train" else None
    Path(part_path(split, 0)).parent.mkdir(parents=True, exist_ok=True)
    p1_all = []
    high = int(pairs_all["s1"].max()) + 1
    for part, low in enumerate(range(0, high, CHUNK_S1)):
        pairs = pairs_all.filter(pl.col("s1").is_between(low, low + CHUNK_S1, closed="left"))
        s1 = features.side_table(split, "s1", pairs.select(idx="s1").unique())
        s23 = features.side_table(split, "s23", pairs.select(idx="s23").unique())
        frame = features.pair_features(pairs.drop("country"), s1, s23, n_s2)
        frame = with_context(with_window(frame, split, out_dir), split, out_dir)
        frame = frame.with_columns(*[pl.lit(False).alias(f) for f in flags], n_families=pl.lit(0, pl.UInt8))
        frame = frame.with_columns(p1=pl.Series(stage1(frame, split, threads)))
        p1_all.append(frame.select("s1", "s23", "country", "is_s3", "cos1", "gap", "p1"))
        frame = frame.filter(pl.col("p1") >= PREFILTER).drop(*flags, "n_families")
        frame = frame.join(per_s1, on="s1", how="left").join(per_s23, on="s23", how="left")
        if truth is not None:
            frame = frame.join(truth, on=["s1", "s23"], how="left").with_columns(pl.col("label").fill_null(False))
        frame.write_parquet(part_path(split, part))
        print(f"part {part}: pairs={pairs.height}, at p1 >= {PREFILTER}: {frame.height}", flush=True)
        del pairs, s1, s23, frame
    pl.concat(p1_all).write_parquet(p1_path(split))


def model_path_for(held_out: int) -> str:
    """Return the reverse model that left out fold ``held_out``."""
    return f"{MODEL_DIR}{PREFIX}_lgbm_f{held_out}.txt"


def load(split: str) -> pl.DataFrame:
    """Return every reverse feature part of ``split``."""
    return pl.read_parquet(f"{REVERSE_DIR}{split}/part_*.parquet")


def fit(held_out: int, rounds: int, threads: int) -> None:
    """Train on the US and India pairs of every S1 entity outside fold ``held_out`` and early-stop on that fold."""
    frame = load("train").filter(pl.col("country").is_in(SCORED)).with_columns(fold=train.fold())
    columns = [c for c in frame.columns if c not in (*train.NON_FEATURES, "fold")]
    fit_rows, valid_rows = frame.filter(pl.col("fold") != held_out), frame.filter(pl.col("fold") == held_out)
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
    """Score the reverse pairs of the countries in ``SCORED`` and write ``reverse_000.parquet``; print densities per country and source."""
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
    frame = frame.with_columns(p=pl.Series(p))
    n = pl.read_parquet(records.records_path(split, "s1"), columns=["country"]).group_by("country").agg(n=pl.len())
    raw = pl.read_parquet(p1_path(split)).group_by("country", "is_s3").agg(raw=pl.len(), passed=(pl.col("p1") >= PREFILTER).sum())
    aggs = [(pl.col("p") >= 0.5).sum().alias("p50"), pl.col("p").quantile(0.1).alias("p_q10"), pl.col("p").median().alias("p_q50")]
    if "label" in frame.columns:
        aggs.append(pl.col("label").mean().alias("true_share"))
    table = raw.join(frame.group_by("country", "is_s3").agg(aggs), on=["country", "is_s3"], how="left").join(n, on="country")
    table = table.with_columns(raw_per_s1=pl.col("raw") / pl.col("n"), passed_share=pl.col("passed") / pl.col("raw"), p50_per_s1=pl.col("p50") / pl.col("n"))
    with pl.Config(tbl_rows=10, tbl_cols=20):
        print(split, table.drop("n").sort("country", "is_s3"), flush=True)
    frame = frame.filter(pl.col("country").is_in(SCORED))
    frame.select("s1", "s23", "p").write_parquet(stack.prob_path(split, 0, out_dir, PREFIX))
    print(f"{split}: {frame.height} reverse pairs of {', '.join(SCORED)} scored, {(frame['p'] >= siblings.CONFIDENT).sum()} at p >= {siblings.CONFIDENT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["pairs", "features", "fit", "predict"])
    parser.add_argument("split", nargs="?", default="train", choices=["train", "test"])
    parser.add_argument("--base", default="blend+sibling+namepair", help="probability prefixes, joined by '+', for the gate and the context features")
    parser.add_argument("--fold", type=int, default=train.HOLDOUT)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--oof", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out-dir", default=features.FEATURE_DIR)
    arguments = parser.parse_args()
    if arguments.step == "pairs":
        write_pairs(arguments.split, arguments.out_dir, arguments.base, arguments.threads)
    elif arguments.step == "features":
        write_features(arguments.split, arguments.out_dir, arguments.base, arguments.threads)
    elif arguments.step == "fit":
        fit(arguments.fold, arguments.rounds, arguments.threads)
    else:
        predict(arguments.split, arguments.out_dir, arguments.oof, arguments.threads)
