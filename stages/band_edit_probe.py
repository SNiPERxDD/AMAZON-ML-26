"""Does edit evidence from the raw text rank the refit band better than its probabilities alone?

An earlier loss breakdown found that the largest recoverable loss on the US and India hold-out is
true links in the ``refine2`` files that the rule (0.5/0.85) rejects (+0.0079 US / +0.0083 India).
Every current feature compares normalised text, from which legal forms, punctuation and the
digit-for-letter swaps are removed. This probe describes how the raw texts of a pair differ:

- name: Levenshtein distance and its normalised form on the folded raw names, equality after
  keeping letters and digits only, equality after mapping digits that stand for letters, the
  legal-form class of each side and their relation (same, different, one side, neither), and the
  length difference;
- address: normalised Levenshtein distance and equality on letters and digits;
- house number: edit type (equal, a digit dropped at an end, offset 1-5, offset 6-30, other of the
  same length, other) and the absolute offset;
- address numbers: the count on each side that the other lacks.

With ``--hashed``, the character 3-grams present on one side only (names and addresses, hashed
with side and field into ``2**BITS`` buckets) are added as a sparse matrix.

Band pairs (``refine2`` hold-out pairs with ``p >= LOW``) are scored twice, cross-fitted over two
hash halves of the S1 entities: ``base`` sees ``p`` and its S1 and S2/S3 context, ``edit`` adds the
features above. Pairs below ``LOW`` keep their ``p``. Printed per country: AUC on the band and
macro F0.5 at 0.5/0.85 and at the best rule of a small grid, for ``refine2`` as is and both models.

With ``--transfer``, each country's band is scored by models fitted on the other country only (leave one
country out), which is how a model would reach France. With ``--base``, the probabilities and
their context come from ``refine.BASE`` instead of ``refine2``, as France has no refit. With
``--disjoint``, the pairs are split into four cells by the S1 half and an S2/S3 hash half, and each
cell is scored by a model fitted on the opposite cell only, so no scored pair shares an S1 or an
S2/S3 record with the pairs its model was fitted on (each model sees about a quarter of the band).
``--shared23`` is its control: the same four cells, but each model is fitted on the cell with the
other S1 half and the same S2/S3 half, so the fit is as small but S2/S3 records are shared.

Run from the repository root: ``PYTHONPATH=. python stages/band_edit_probe.py [--hashed] [--transfer] [--base] [--disjoint | --shared23]``.
"""

import argparse
import re
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse

from pipeline import features, metric, normalize, records, refine, refine2, train

D = features.FEATURE_DIR
LOW = 0.05
BITS = 14
GRID = [(t, r) for t in (0.4, 0.45, 0.5, 0.55, 0.6) for r in (0.75, 0.8, 0.85, 0.9)]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9, "verbose": -1,
          "num_threads": 8}
ROUNDS = 400
LEGAL = {
    "inc": 1, "incorporated": 1, "llc": 2, "ltd": 3, "limited": 3, "pvt": 4, "private": 4, "corp": 5, "corporation": 5, "co": 6, "company": 6,
    "llp": 7, "plc": 8, "pc": 9, "sarl": 10, "sas": 11, "sasu": 11, "sci": 12, "sa": 13, "ets": 14,
}
LEGAL_RE = r"\b(" + "|".join(sorted(LEGAL, key=len, reverse=True)) + r")\b"
BASE = ["p", "ratio", "p_s1_max", "p_s23_max", "rank_s1", "rank_s23", "n50_s1", "n50_s23"]


def raw_side(side: str, split: str = "train") -> pl.DataFrame:
    """Return ``idx`` with folded raw name and address and the house and numbers of one side of ``split``."""
    raw = records.load_side(split, side).select(
        "idx", name=normalize.folded_text("business_name"), addr=normalize.folded_text("business_address"))
    norm = pl.read_parquet(records.records_path(split, side), columns=["idx", "house", "numbers"])
    return raw.join(norm, on="idx", how="left").with_columns(
        legal=pl.col("name").str.extract(LEGAL_RE, 1).replace_strict(LEGAL, default=0, return_dtype=pl.Int8),
        name_an=pl.col("name").str.replace_all(r"[^\p{L}\p{N}]", ""),
        addr_an=pl.col("addr").str.replace_all(r"[^\p{L}\p{N}]", ""))


def house_edit(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Return the house-number edit type: 0 missing, 1 equal, 3 digit dropped at an end, 4 offset 1-5, 5 offset 6-30, 2 other same length, 6 other."""
    ia, ib = a.cast(pl.Int64, strict=False), b.cast(pl.Int64, strict=False)
    offset = (ia - ib).abs()
    same_len = a.str.len_chars() == b.str.len_chars()
    dropped = ((a.str.len_chars() - b.str.len_chars()).abs() == 1) & (
        a.str.starts_with(b) | a.str.ends_with(b) | b.str.starts_with(a) | b.str.ends_with(a))
    return (pl.when(a.is_null() | b.is_null()).then(0).when(a == b).then(1).when(dropped).then(3)
            .when(offset.is_between(1, 5)).then(4).when(offset.is_between(6, 30)).then(5).when(same_len).then(2).otherwise(6).cast(pl.Int8))


def edit_features(band: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """Return the dense edit features of each band pair, in band order."""
    frame = band.select("s1", "s23").join(s1.rename(lambda c: f"{c}_a"), left_on="s1", right_on="idx_a", how="left", maintain_order="left").join(
        s23.rename(lambda c: f"{c}_b"), left_on="s23", right_on="idx_b", how="left", maintain_order="left")
    glyph = {d: letter for d, letter in zip(normalize.HOMOGLYPH_DIGITS, normalize.HOMOGLYPH_LETTERS, strict=True)}
    name_lev = process.cpdist(frame["name_a"].to_list(), frame["name_b"].to_list(), scorer=Levenshtein.distance, workers=8)
    name_norm = process.cpdist(frame["name_a"].to_list(), frame["name_b"].to_list(), scorer=Levenshtein.normalized_distance, workers=8)
    addr_norm = process.cpdist(frame["addr_a"].to_list(), frame["addr_b"].to_list(), scorer=Levenshtein.normalized_distance, workers=8)
    return frame.select(
        name_lev=pl.Series(name_lev).cast(pl.Float32),
        name_norm=pl.Series(name_norm).cast(pl.Float32),
        name_an_eq=(pl.col("name_an_a") == pl.col("name_an_b")).cast(pl.Int8),
        glyph_eq=(pl.col("name_an_a").str.replace_many(glyph) == pl.col("name_an_b").str.replace_many(glyph)).cast(pl.Int8),
        legal_a=pl.col("legal_a"),
        legal_b=pl.col("legal_b"),
        legal_rel=pl.when((pl.col("legal_a") == 0) & (pl.col("legal_b") == 0)).then(0).when(pl.col("legal_a") == pl.col("legal_b")).then(1)
        .when((pl.col("legal_a") == 0) | (pl.col("legal_b") == 0)).then(3).otherwise(2).cast(pl.Int8),
        len_diff=(pl.col("name_a").str.len_chars().cast(pl.Int32) - pl.col("name_b").str.len_chars().cast(pl.Int32)),
        addr_norm=pl.Series(addr_norm).cast(pl.Float32),
        addr_an_eq=(pl.col("addr_an_a") == pl.col("addr_an_b")).cast(pl.Int8),
        house_edit=house_edit(pl.col("house_a"), pl.col("house_b")),
        house_offset=(pl.col("house_a").cast(pl.Int64, strict=False) - pl.col("house_b").cast(pl.Int64, strict=False)).abs().cast(pl.Float32),
        nums_a_only=pl.col("numbers_a").list.set_difference(pl.col("numbers_b")).list.len().cast(pl.Int16),
        nums_b_only=pl.col("numbers_b").list.set_difference(pl.col("numbers_a")).list.len().cast(pl.Int16),
    )


def grams(text: str) -> set[str]:
    """Return the character 3-grams of ``text`` padded with spaces."""
    padded = " " + re.sub(r"\s+", " ", text) + " "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def hashed(band: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> sparse.csr_matrix:
    """Return the hashed one-side character 3-grams of names and addresses, one row per band pair."""
    frame = band.select("s1", "s23").join(s1.select(s1="idx", na="name", aa="addr"), on="s1", how="left", maintain_order="left").join(
        s23.select(s23="idx", nb="name", ab="addr"), on="s23", how="left", maintain_order="left")
    size = 1 << BITS
    indptr, indices = [0], []
    for na, aa, nb, ab in frame.select("na", "aa", "nb", "ab").iter_rows():
        cols = set()
        for field, (x, y) in (("n", (na, nb)), ("a", (aa, ab))):
            gx, gy = grams(x or ""), grams(y or "")
            cols.update(zlib.crc32(f"{field}1{g}".encode()) % size for g in gx - gy)
            cols.update(zlib.crc32(f"{field}2{g}".encode()) % size for g in gy - gx)
        indices.extend(sorted(cols))
        indptr.append(len(indices))
    return sparse.csr_matrix((np.ones(len(indices), np.float32), np.array(indices, np.int32), np.array(indptr, np.int64)), shape=(frame.height, size))


def word_features(band: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> tuple[pl.DataFrame, sparse.csr_matrix]:
    """Return word-level name features and hashed one-side name words of each band pair, in band order.

    Word rarity is ``log(n / df)`` with the document frequency counted over the names of both sides (no labels).
    """
    frame = band.select("s1", "s23").join(s1.select(s1="idx", na="name", aa="addr"), on="s1", how="left", maintain_order="left").join(
        s23.select(s23="idx", nb="name", ab="addr"), on="s23", how="left", maintain_order="left")
    words = pl.concat([s1.select("idx", "name"), s23.select("idx", "name")]).select(w=pl.col("name").fill_null("").str.split(" ").list.unique())
    df = words.explode("w").filter(pl.col("w") != "").group_by("w").len()
    idf = dict(zip(df["w"].to_list(), np.log(words.height / df["len"].to_numpy()).tolist(), strict=True))
    top = float(np.log(words.height))
    na, nb = frame["na"].fill_null("").to_list(), frame["nb"].fill_null("").to_list()
    aa, ab = frame["aa"].fill_null("").to_list(), frame["ab"].fill_null("").to_list()
    dense = {f"name_{k}": process.cpdist(na, nb, scorer=getattr(fuzz, k), workers=8).astype(np.float32)
             for k in ("token_set_ratio", "token_sort_ratio", "partial_ratio")}
    dense["name_jw"] = process.cpdist(na, nb, scorer=JaroWinkler.similarity, workers=8).astype(np.float32)
    dense["addr_token_set_ratio"] = process.cpdist(aa, ab, scorer=fuzz.token_set_ratio, workers=8).astype(np.float32)
    dense["addr_partial_ratio"] = process.cpdist(aa, ab, scorer=fuzz.partial_ratio, workers=8).astype(np.float32)
    rows = {k: np.zeros(frame.height, np.float32) for k in ("w_shared_idf", "w_a_idf", "w_b_idf", "w_a_max", "w_b_max", "w_a_n", "w_b_n")}
    size = 1 << BITS
    indptr, indices = [0], []
    for i, (x, y) in enumerate(zip(na, nb, strict=True)):
        wa, wb = set(x.split()), set(y.split())
        only_a, only_b = wa - wb, wb - wa
        ia, ib = [idf.get(w, top) for w in only_a], [idf.get(w, top) for w in only_b]
        rows["w_shared_idf"][i] = sum(idf.get(w, top) for w in wa & wb)
        rows["w_a_idf"][i], rows["w_b_idf"][i] = sum(ia), sum(ib)
        rows["w_a_max"][i], rows["w_b_max"][i] = max(ia, default=0), max(ib, default=0)
        rows["w_a_n"][i], rows["w_b_n"][i] = len(only_a), len(only_b)
        cols = {zlib.crc32(f"w1{w}".encode()) % size for w in only_a} | {zlib.crc32(f"w2{w}".encode()) % size for w in only_b}
        indices.extend(sorted(cols))
        indptr.append(len(indices))
    matrix = sparse.csr_matrix((np.ones(len(indices), np.float32), np.array(indices, np.int32), np.array(indptr, np.int64)), shape=(frame.height, size))
    return pl.DataFrame(dense | rows), matrix


def context(preds: pl.DataFrame) -> pl.DataFrame:
    """Add the S1 and S2/S3 context of ``p`` used by the base model."""
    out = train.with_maxima(preds.lazy()).collect().with_columns(ratio=pl.col("p") / pl.col("p_s1_max"))
    for side in ("s1", "s23"):
        out = out.with_columns(pl.col("p").rank("ordinal", descending=True).over(side).cast(pl.Float32).alias(f"rank_{side}"),
                               (pl.col("p") >= 0.5).sum().over(side).cast(pl.Float32).alias(f"n50_{side}"))
    return out


def evaluate(name: str, preds: pl.DataFrame, truth: pl.DataFrame, ids: pl.DataFrame, country: str) -> None:
    """Print F0.5 at 0.5/0.85 and at the best grid rule for one probability set."""
    preds = train.with_maxima(preds.lazy()).collect()
    f = {g: metric.macro_f05(train.decide(preds, *g, False), truth, ids) for g in GRID}
    best = max(f, key=f.get)
    print(f"{country} {name}: F0.5 at 0.5/0.85 {f[(0.5, 0.85)]:.5f}, best {f[best]:.5f} at {best[0]}/{best[1]}", flush=True)


def main() -> None:
    """Fit the base and edit models cross-fitted on the hold-out band and print their scores."""
    from stages.reverse_frontier import (
        auc,  # reads feature data on import; only the full probe needs it
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hashed", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--disjoint", action="store_true")
    parser.add_argument("--shared23", action="store_true")
    parser.add_argument("--words", action="store_true")
    arguments = parser.parse_args()
    prefix = refine.BASE if arguments.base else refine2.PREFIX
    scored = pl.scan_parquet(f"{D}train/part_*.parquet").select("s1").unique().filter(train.fold() == train.HOLDOUT).collect()
    country = pl.read_parquet(records.records_path("train", "s1"), columns=["idx", "country"]).rename({"idx": "s1"})
    scored = scored.join(country.filter(pl.col("country").is_in(["US", "India"])), on="s1")
    truth_all = features.truth_links().join(scored, on="s1", how="semi").select("s1", "s23")
    preds = context(train.prob_scan("train", D, prefix).join(scored.lazy(), on="s1", how="semi").collect())
    band = (preds.filter(pl.col("p") >= LOW).join(scored, on="s1").join(truth_all.with_columns(t=pl.lit(1, pl.Int8)), on=["s1", "s23"], how="left")
            .with_columns(pl.col("t").fill_null(0), half=(pl.col("s1").hash(7) % 2).cast(pl.Int8),
                          half23=(pl.col("s23").hash(7) % 2).cast(pl.Int8), us=(pl.col("country") == "US").cast(pl.Int8)))
    print(f"band pairs {band.height}, true share {band['t'].mean():.3f}", flush=True)
    s1_raw, s23_raw = raw_side("s1"), raw_side("s23")
    s23_raw = s23_raw.join(band.select(idx="s23").unique(), on="idx", how="semi")
    edit = edit_features(band, s1_raw, s23_raw)
    print("edit features done", flush=True)
    label, halves, halves23 = band["t"].to_numpy(), band["half"].to_numpy(), band["half23"].to_numpy()
    x_base = band.select(*BASE, "us").cast(pl.Float32).to_numpy()
    x_edit = np.hstack([x_base, edit.cast(pl.Float32).to_numpy()])
    names = {"base": [*BASE, "us"], "edit": [*BASE, "us", *edit.columns]}
    mats = {"base": x_base, "edit": x_edit}
    if arguments.hashed:
        mats["edit + grams"] = sparse.hstack([sparse.csr_matrix(x_edit), hashed(band, s1_raw, s23_raw)], format="csr")
        names["edit + grams"] = names["edit"] + [f"g{i}" for i in range(1 << BITS)]
        print("hashed grams done", flush=True)
    if arguments.words:
        wdense, wmatrix = word_features(band, s1_raw, s23_raw)
        wnames = [*names["edit"], *wdense.columns]
        x_words = np.hstack([x_edit, wdense.to_numpy()])
        mats["edit + words"], names["edit + words"] = x_words, wnames
        if arguments.hashed:
            mats["edit + words + grams"] = sparse.hstack([sparse.csr_matrix(x_words), mats["edit + grams"][:, x_edit.shape[1]:], wmatrix], format="csr")
            names["edit + words + grams"] = wnames + [f"g{i}" for i in range(1 << BITS)] + [f"w{i}" for i in range(1 << BITS)]
            del mats["base"], mats["edit"]
        print("word features done", flush=True)
    del s1_raw, s23_raw
    scores = {}
    for model, x in mats.items():
        score = np.zeros(band.height, np.float32)
        countries = band["country"].to_numpy()
        if arguments.transfer:
            splits = [(countries != c, countries == c) for c in ("US", "India")]
        elif arguments.disjoint or arguments.shared23:
            splits = [((halves != a) & ((halves23 == b) if arguments.shared23 else (halves23 != b)), (halves == a) & (halves23 == b))
                      for a in (0, 1) for b in (0, 1)]
        else:
            splits = [(halves == h, halves != h) for h in (0, 1)]
        for fit_rows, out in splits:
            booster = lgb.train(PARAMS, lgb.Dataset(x[fit_rows], label[fit_rows], feature_name=names[model]), ROUNDS)
            score[out] = booster.predict(x[out])
        gain = booster.feature_importance("gain")
        top = sorted(zip(names[model], gain / gain.sum(), strict=True), key=lambda item: -item[1])[:12]
        print(f"{model} gain share: " + ", ".join(f"{n}={s:.3f}" for n, s in top), flush=True)
        scores[model] = score
    low = preds.filter(pl.col("p") < LOW).select("s1", "s23", "p")
    for name in ("US", "India"):
        ids = scored.filter(pl.col("country") == name).select("s1")
        truth = truth_all.join(ids, on="s1", how="semi")
        mask = band["country"].to_numpy() == name
        evaluate(prefix, preds.join(ids, on="s1", how="semi").select("s1", "s23", "p"), truth, ids, name)
        for model, score in scores.items():
            print(f"{name} {model}: band AUC {auc(score[mask], label[mask].astype(bool)):.5f}", flush=True)
            candidate = pl.concat([band.filter(pl.Series(mask)).select("s1", "s23", p=pl.Series(score[mask])), low.join(ids, on="s1", how="semi")])
            evaluate(model, candidate, truth, ids, name)


if __name__ == "__main__":
    main()
