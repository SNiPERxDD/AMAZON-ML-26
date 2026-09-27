"""Is there a compact, high-precision region of reverse-retrieved pairs, and does the stage-1 model find it?

Follows ``reverse_trigram_probe.py``. Per country, samples ``GATED`` train S2/S3 records that have
an address and whose best base probability over the current rounds is below 0.5, ranks every S1
record of the country by the ``name+address`` trigram cosine, and keeps the S1 records at ranks 1 to
``RANKS`` when the pair is not already a candidate. Each new pair gets:

- ``rank``, ``cos1``, ``cos2`` and ``gap`` (the pair's cosine, the next rank's cosine, their difference);
- the first-round pair features, computed together with the S1 entity's existing blocking pairs
  so that per-S1 ranks and counts see its other candidates, plus the S2/S3 context table;
- ``p1``: the stage-1 probability from the model that left out the S1 entity's fold. Reverse
  pairs have no candidate family (``n_families`` 0, all ``f_*`` false), a value stage 1 never saw
  in training, so ``p1`` is an extrapolation.

Printed per country and cut, for rank 1 and for ranks 1 to ``RANKS``: new pairs per S1, true share, new true links per S1, the share of
never-scored true links recovered, and a ceiling: the held-out F0.5 at 0.5/0.85 when that share of
the held-out never-scored true links is added with no false pairs. Also the AUC of ``cos1``,
``gap`` and ``p1`` on the new pairs.

With ``--empty-address``, the queries are the gated S2/S3 records with no address words or numbers (excluded above and
by :mod:`pipeline.reverse`), ranked by the ``name`` trigram cosine instead.

Run from the repository root: ``PYTHONPATH=. python stages/reverse_frontier.py [--empty-address]``.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp

from pipeline import features, metric, records, refine, refine2, train
from stages.missed_link_profile import missed
from stages.reverse_trigram_probe import DF_CAP, TEXTS, matrix, trigrams

D = features.FEATURE_DIR
GATED = 40_000
RANKS = 3
FLAGS = [c for c in pl.scan_parquet(f"{D}train/part_000.parquet").collect_schema().names() if c.startswith("f_")]


def top_n(q: sp.csr_matrix, s1t: sp.csr_matrix, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per query row the S1 row positions (-1 when absent) and cosines of the ``n`` highest cosines, best first."""
    best, cos = np.full((q.shape[0], n), -1), np.zeros((q.shape[0], n))
    for start in range(0, q.shape[0], 200):
        prod = (q[start:start + 200] @ s1t).tocsr()
        for i in range(prod.shape[0]):
            lo, hi = prod.indptr[i], prod.indptr[i + 1]
            data = prod.data[lo:hi]
            order = np.argsort(-data)[:n] if hi - lo <= n else np.argpartition(-data, n)[:n]
            order = order[np.argsort(-data[order])]
            best[start + i, :len(order)], cos[start + i, :len(order)] = prod.indices[lo + order], data[order]
    return best, cos


def stage1(new: pl.DataFrame, n_s2: int) -> np.ndarray:
    """Return the out-of-fold stage-1 probability of each pair in ``new`` (s1, s23)."""
    block = pl.scan_parquet(f"{D}train/part_*.parquet").select("s1", "s23", *FLAGS, "n_families").join(new.lazy().select("s1").unique(), on="s1", how="semi").collect()
    pairs = pl.concat([block, new.select("s1", "s23").with_columns(*[pl.lit(False).alias(f) for f in FLAGS], n_families=pl.lit(0, pl.UInt8))], how="vertical_relaxed")
    s1 = features.side_table("train", "s1", pairs.select(idx="s1").unique())
    s23 = features.side_table("train", "s23", pairs.select(idx="s23").unique())
    frame = features.pair_features(pairs, s1, s23, n_s2).join(new.select("s1", "s23"), on=["s1", "s23"], how="semi")
    context = pl.read_parquet(features.context_path("train"))
    frame = frame.join(context, on="s23", how="left").with_columns(
        other_best_name=pl.when(pl.col("s1") == pl.col("top_s1")).then(pl.col("second_name")).otherwise(pl.col("top_name")),
        other_strong=(pl.col("s23_strong") - pl.col("strong").cast(pl.UInt16)),
    ).with_columns(other_name_gap=pl.col("name_ratio") - pl.col("other_best_name"))
    frame = new.select("s1", "s23").join(frame, on=["s1", "s23"], how="left", maintain_order="left")
    fold_of = frame.select(train.fold()).to_series().to_numpy()
    p = np.zeros(frame.height, dtype=np.float32)
    for k in range(train.FOLDS):
        booster = lgb.Booster(model_file=train.model_path_for(train.MODEL_PATH, k))
        rows = np.flatnonzero(fold_of == k)
        if len(rows):
            p[rows] = booster.predict(train.matrix(frame[rows], booster.feature_name()), num_threads=8)
    return p


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Return the rank AUC of ``score`` for ``label``."""
    ranks = pl.Series(score).rank("average").to_numpy()
    pos = label.sum()
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))) if 0 < pos < len(label) else float("nan")


def main() -> None:
    """Print the recovery frontier of gated reverse top-1 pairs per country and cut."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--empty-address", action="store_true")
    empty = parser.parse_args().empty_address
    has_address = (pl.col("words").list.len() > 0) | (pl.col("numbers").list.len() > 0)
    cols = ["idx", "country", "core", "words", "numbers"]
    s1 = pl.read_parquet(records.records_path("train", "s1"), columns=cols)
    s23 = pl.read_parquet(records.records_path("train", "s23"), columns=cols)
    truth = features.truth_links()
    lost = missed()
    n_s2 = pl.scan_parquet(f"{records.PARQUET_DIR}train_source2.parquet").select(pl.len()).collect().item()
    known = pl.concat([pl.scan_parquet("data/candidates/train_pairs.parquet").select("s1", "s23"),
                       *[pl.scan_parquet(str(p)).select("s1", "s23") for n in ("sibling", "namepair") for p in Path(f"{D}train").glob(f"{n}_*.parquet")]])
    best = refine.rounds_scan("train", D).group_by("s23").agg(best=pl.col("p").max()).collect()
    hold = pl.scan_parquet(f"{D}train/part_*.parquet").select("s1").unique().filter(train.fold() == train.HOLDOUT).collect()
    preds = train.with_maxima(train.prob_scan("train", D, refine2.PREFIX).join(hold.lazy(), on="s1", how="semi")).collect()
    kept = train.decide(preds, 0.5, 0.85, False)
    for country in ("US", "India"):
        c1 = s1.filter(pl.col("country") == country)
        c23 = s23.filter(pl.col("country") == country)
        gated_all = c23.filter(~has_address if empty else has_address).join(best, left_on="idx", right_on="s23", how="left")
        gated_all = gated_all.filter(pl.col("best").fill_null(0.0) < 0.5)
        queries = gated_all.sample(min(GATED, gated_all.height), seed=2).drop("best")
        scale = gated_all.height / queries.height / c1.height
        c_lost = lost.join(c23.select(s23="idx"), on="s23", how="semi")
        text = TEXTS["name" if empty else "name+address"]
        t1 = trigrams(c1, text)
        vocab = t1.group_by("tri").agg(df=pl.len()).with_columns(idf=(np.log((c1.height + 1) / (pl.col("df") + 1)) + 1), keep=pl.col("df") <= DF_CAP)
        vocab = vocab.sort("tri").with_row_index("col")
        s1t = matrix(t1, c1.select("idx"), vocab).T.tocsr()
        pos, cos = top_n(matrix(trigrams(queries, text), queries.select("idx"), vocab), s1t, RANKS + 1)
        s1_idx = c1["idx"].to_numpy()
        new = pl.concat([pl.DataFrame({"s23": queries["idx"], "pos": pos[:, r], "rank": r + 1, "cos1": cos[:, r], "cos2": cos[:, r + 1]}) for r in range(RANKS)])
        new = new.filter(pl.col("pos") >= 0)
        new = new.with_columns(s1=pl.Series(s1_idx[new["pos"].to_numpy()], dtype=pl.UInt32), gap=pl.col("cos1") - pl.col("cos2")).drop("pos")
        new = new.join(known.join(new.lazy().select("s23"), on="s23", how="semi").collect(), on=["s1", "s23"], how="anti")
        new = new.join(truth.with_columns(t=pl.lit(True)), on=["s1", "s23"], how="left").with_columns(pl.col("t").fill_null(False))
        new = new.with_columns(p1=pl.Series(stage1(new, n_s2)))
        lost_per_s1 = c_lost.height / c1.height
        label = new["t"].to_numpy()
        print(f"== {country}: S1 {c1.height}, gated S2/S3 {gated_all.height}, sampled {queries.height}, new pairs at ranks 1-{RANKS} {new.height}, true {int(label.sum())}; "
              f"never-scored true per S1 {lost_per_s1:.4f}; AUC cos1 {auc(new['cos1'].to_numpy(), label):.3f}, gap {auc(new['gap'].to_numpy(), label):.3f}, "
              f"p1 {auc(new['p1'].to_numpy(), label):.3f}", flush=True)
        ids = hold.join(c1.select(s1="idx"), on="s1", how="semi")
        c_truth = truth.join(ids, on="s1", how="semi")
        c_kept = kept.join(ids, on="s1", how="semi")
        h_lost = c_lost.join(ids, on="s1", how="semi")
        base_f = metric.macro_f05(c_kept, c_truth, ids)
        cuts = {"all": pl.lit(True)}
        cuts |= {f"cos1>={c}": pl.col("cos1") >= c for c in (0.5, 0.6, 0.7, 0.8)}
        cuts |= {f"gap>={g}": pl.col("gap") >= g for g in (0.05, 0.1, 0.2)}
        cuts |= {f"p1>={p}": pl.col("p1") >= p for p in (0.02, 0.1, 0.3, 0.5)}
        cuts |= {"cos1>=0.6 & gap>=0.1": (pl.col("cos1") >= 0.6) & (pl.col("gap") >= 0.1), "p1>=0.1 & gap>=0.05": (pl.col("p1") >= 0.1) & (pl.col("gap") >= 0.05)}
        cuts = {f"rank1 {k}": v & (pl.col("rank") == 1) for k, v in cuts.items()} | {f"rank<={RANKS} {k}": v for k, v in cuts.items() if k.startswith(("all", "p1"))}
        for name, cut in cuts.items():
            part = new.filter(cut)
            true_per_s1 = part["t"].sum() * scale
            share = true_per_s1 / lost_per_s1
            added = h_lost.sample(fraction=min(share, 1.0), seed=3).select("s1", "s23")
            ceiling = metric.macro_f05(pl.concat([c_kept, added]).unique(), c_truth, ids) - base_f
            print(f"{country} {name}: new pairs per S1 {part.height * scale:.3f}, true share {part['t'].mean() or 0:.4f}, new true per S1 {true_per_s1:.4f}, "
                  f"recovered share {share:.3f}, ceiling F0.5 {ceiling:+.5f}", flush=True)


if __name__ == "__main__":
    main()
