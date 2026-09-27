"""Does a linear model over crossed name words (one-side word pairs) add to the band edit model?

The band edit model (``band_edit_probe.py --hashed --words``) sees hashed words present on one side only, separately for
each side. A substitution such as one abbreviation replacing a full word is then two unrelated columns. This hashes each
pair (word only in the S1 name, word only in the S2/S3 name), plus the one-side words, fits an L2 logistic regression on
them, and adds its out-of-fold logit to the H+words model as one dense feature.

Cross-fitting is nested: for each hold-out S1-hash half ``h``, the logit of half ``h``'s training rows comes from two
inner S1-hash folds of half ``h``, and the logit of the scored half from a regression on all of half ``h``. Printed per
country: F0.5 at 0.5/0.85 and at the best grid rule, for H+words + the name-word logit and H+words + all three logits
(name words, address words, name character 3-grams), and each logit's AUC. H+words alone scored 0.98577 / 0.98266.

Run from the repository root: ``PYTHONPATH=. python stages/crossed_words_probe.py``.
"""

import argparse
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import optimize, sparse

from pipeline import features, records, refine2, train
from stages.band_edit_probe import (
    BASE,
    BITS,
    LOW,
    PARAMS,
    ROUNDS,
    context,
    edit_features,
    evaluate,
    grams,
    hashed,
    raw_side,
    word_features,
)

D = features.FEATURE_DIR
CROSS_BITS = 18
MAX_CROSS = 64
BLOCKS = ("name_words", "addr_words", "name_grams")
L2 = 1.0


def crossed(band: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame, block: str = "name_words") -> sparse.csr_matrix:
    """Return hashed one-side tokens and crossed one-side token pairs of each band pair, in band order.

    ``block`` is ``name_words``, ``addr_words`` or ``name_grams`` (character 3-grams; crosses skipped above ``MAX_CROSS`` pairs).
    """
    field = "addr" if block == "addr_words" else "name"
    frame = band.select("s1", "s23").join(s1.select(s1="idx", ta=field), on="s1", how="left", maintain_order="left").join(
        s23.select(s23="idx", tb=field), on="s23", how="left", maintain_order="left")
    split = grams if block == "name_grams" else (lambda text: set(text.split()))
    size = 1 << CROSS_BITS
    indptr, indices = [0], []
    for ta, tb in frame.select(pl.col("ta").fill_null(""), pl.col("tb").fill_null("")).iter_rows():
        wa, wb = split(ta), split(tb)
        only_a, only_b = wa - wb, wb - wa
        keys = {f"a{w}" for w in only_a} | {f"b{w}" for w in only_b}
        if len(only_a) * len(only_b) <= MAX_CROSS:
            keys |= {f"x{u}|{v}" for u in only_a for v in only_b}
        cols = {zlib.crc32(k.encode()) % size for k in keys}
        indices.extend(sorted(cols))
        indptr.append(len(indices))
    return sparse.csr_matrix((np.ones(len(indices), np.float32), np.array(indices, np.int32), np.array(indptr, np.int64)), shape=(frame.height, size))


def logistic(x: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    """Return the weights (bias last) of an L2 logistic regression fitted with L-BFGS."""
    xb = sparse.hstack([x, np.ones((x.shape[0], 1), np.float32)], format="csr")
    xt = xb.T.tocsr()

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = xb @ w
        value = np.logaddexp(0, z).sum() - y @ z + L2 / 2 * (w[:-1] @ w[:-1])
        grad = xt @ (1 / (1 + np.exp(-z)) - y)
        grad[:-1] += L2 * w[:-1]
        return value, grad

    return optimize.minimize(loss, np.zeros(xb.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


def logit(x: sparse.csr_matrix, w: np.ndarray) -> np.ndarray:
    """Return the regression's logit."""
    return x @ w[:-1] + w[-1]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Return the ROC AUC of scores ``s`` for labels ``y``."""
    ranks = np.argsort(np.argsort(s)) + 1
    pos = y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def main() -> None:
    """Print hold-out F0.5 of H+words with and without the crossed-word logit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", help="write the cross-fitted band scores (s1, s23, one column per model) to this parquet file")
    args = parser.parse_args()
    scored = pl.scan_parquet(f"{D}train/part_*.parquet").select("s1").unique().filter(train.fold() == train.HOLDOUT).collect()
    country = pl.read_parquet(records.records_path("train", "s1"), columns=["idx", "country"]).rename({"idx": "s1"})
    scored = scored.join(country.filter(pl.col("country").is_in(["US", "India"])), on="s1")
    truth_all = features.truth_links().join(scored, on="s1", how="semi").select("s1", "s23")
    preds = context(train.prob_scan("train", D, refine2.PREFIX).join(scored.lazy(), on="s1", how="semi").collect())
    band = (preds.filter(pl.col("p") >= LOW).join(scored, on="s1").join(truth_all.with_columns(t=pl.lit(1, pl.Int8)), on=["s1", "s23"], how="left")
            .with_columns(pl.col("t").fill_null(0), half=(pl.col("s1").hash(7) % 2).cast(pl.Int8), inner=(pl.col("s1").hash(17) % 2).cast(pl.Int8),
                          us=(pl.col("country") == "US").cast(pl.Int8)))
    print(f"band pairs {band.height}, true share {band['t'].mean():.3f}", flush=True)
    s1_raw, s23_raw = raw_side("s1"), raw_side("s23")
    s23_raw = s23_raw.join(band.select(idx="s23").unique(), on="idx", how="semi")
    cross = {block: crossed(band, s1_raw, s23_raw, block) for block in BLOCKS}
    edit = edit_features(band, s1_raw, s23_raw)
    wdense, wmatrix = word_features(band, s1_raw, s23_raw)
    gram_matrix = hashed(band, s1_raw, s23_raw)
    del s1_raw, s23_raw
    dense = np.hstack([band.select(*BASE, "us").cast(pl.Float32).to_numpy(), edit.cast(pl.Float32).to_numpy(), wdense.to_numpy()])
    names = [*BASE, "us", *edit.columns, *wdense.columns] + [f"g{i}" for i in range(1 << BITS)] + [f"w{i}" for i in range(1 << BITS)]
    x = sparse.hstack([sparse.csr_matrix(dense), gram_matrix, wmatrix], format="csr", dtype=np.float32)
    del dense, gram_matrix, wmatrix
    print(f"features done: {x.shape[1]} columns, {x.nnz} non-zeros; crossed {({b: m.nnz for b, m in cross.items()})} non-zeros", flush=True)
    label, halves, inner = band["t"].to_numpy(), band["half"].to_numpy(), band["inner"].to_numpy()
    models = {"H+words + name_words": ["name_words"], "H+words + all crossed": list(BLOCKS)}
    scores = {name: np.zeros(band.height, np.float32) for name in models}
    for h in (0, 1):
        fit, target = halves == h, halves != h
        logits = {}
        for block, matrix in cross.items():
            column = np.zeros(band.height, np.float32)
            for k in (0, 1):
                w = logistic(matrix[fit & (inner != k)], label[fit & (inner != k)].astype(np.float64))
                column[fit & (inner == k)] = logit(matrix[fit & (inner == k)], w)
            w = logistic(matrix[fit], label[fit].astype(np.float64))
            column[target] = logit(matrix[target], w)
            logits[block] = column
            print(f"half {h}: {block} logit AUC on the scored half {auc(label[target], column[target]):.4f}", flush=True)
        for name, blocks in models.items():
            matrix = sparse.hstack([x, sparse.csr_matrix(np.column_stack([logits[b] for b in blocks]))], format="csr")
            booster = lgb.train(PARAMS, lgb.Dataset(matrix[fit], label[fit], feature_name=[*names, *(f"{b}_logit" for b in blocks)]), ROUNDS)
            scores[name][target] = booster.predict(matrix[target])
            print(f"{name}: half {h} done", flush=True)
            del booster, matrix
    if args.save:
        band.select("s1", "s23").with_columns(**{name: pl.Series(score) for name, score in scores.items()}).write_parquet(args.save)
    low = preds.filter(pl.col("p") < LOW).select("s1", "s23", "p")
    for model, score in scores.items():
        for name in ("US", "India"):
            ids = scored.filter(pl.col("country") == name).select("s1")
            mask = band["country"].to_numpy() == name
            candidate = pl.concat([band.filter(pl.Series(mask)).select("s1", "s23", p=pl.Series(score[mask])), low.join(ids, on="s1", how="semi")])
            evaluate(model, candidate, truth_all.join(ids, on="s1", how="semi"), ids, name)


if __name__ == "__main__":
    main()
