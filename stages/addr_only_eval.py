"""Hold-out value of adding address-only matches (``addr_only_match.py``) to the current US/India prediction.

The base is the band with the ``xedit`` scores under the 0.5/0.85 rule. Candidates are the address-only pairs whose S1
is in the hold-out and whose S2/S3 record the base leaves unlinked. A LightGBM classifier on the pair's address and
name scores, the record's best and second address scores, the count of S1 within 5 points (shared buildings) and the
base link count of the S1 is cross-fitted over five S1 folds; the out-of-fold probability is cut at a grid of
thresholds and the added links' precision and F0.5 change are printed per country. ``--save`` fits on all candidates
and writes the booster for the test round.

Run from the repository root: ``PYTHONPATH=. python stages/addr_only_eval.py``.
"""

import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import metric, records

ROOT = "data/kaggle_edit_r2/"
SCORE = "H+words + all crossed"
FEATURES = ["shared", "addr", "addr_sort", "name", "n_close", "rank", "n_cand", "top_addr", "second_addr", "nb", "xmax", "india"]
# name and address shape: the generator's decoys (made-up name at an S1's exact address, linked to nothing) copy the
# address verbatim and often carry a legal suffix, while true brand records tend to drop or change the house number
SHAPE = ["n2_tokens", "n2_upper", "n2_legal", "n2_domain", "n2_digits", "n2_ascii", "n1_tokens", "n1_legal",
         "a2_hash", "a2_null", "a2_upper", "num1", "num2", "num_eq", "a_len_ratio", "a2_state_full", "a_order",
         "acro_eq", "acro_core_eq", "dom_eq", "dom_core_eq", "dom_in", "stem_eq"]
LEGAL = r"\b(llc|l l c|inc|corp|corporation|ltd|limited|pvt|private|co|company|fund|trust|holdings|group|partners|lp|llp|associates|pc|pllc|enterprises|services|industries|capital|management|sarl|sas|sasu|sci|eurl|sa|cie|fils)\b"


def shape(pairs: pl.DataFrame, split: str, keep_text: bool = False) -> pl.DataFrame:
    """Add SHAPE features from the raw S1 and S2/S3 text of ``split``."""
    s1, s23 = records.load_split(split)
    pairs = (pairs.join(s1.select(s1="idx", n1="business_name", a1="business_address"), on="s1", how="left")
             .join(s23.select(s23="idx", n2="business_name", a2="business_address"), on="s23", how="left"))
    del s1, s23
    n1, n2 = pl.col("n1").fill_null(""), pl.col("n2").fill_null("")
    a1, a2 = pl.col("a1").fill_null(""), pl.col("a2").fill_null("")
    # the generator's name variants: initials of the S1 words, or a domain made of the concatenated S1 words
    w1 = n1.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
    core = w1.str.replace_all(LEGAL, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    acro = w1.str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    acro_core = core.str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    cat1, cat_core = w1.str.replace_all(" ", ""), core.str.replace_all(" ", "")
    n2k = n2.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", "")
    stem = (n2.str.to_lowercase().str.replace(r"^(https?://)?(www\.)?", "").str.replace(r"\.[a-z.]+$", "")
            .str.replace_all(r"[^\p{L}\p{N}]+", ""))
    num = r"(?:^|[\s,#])(\d+)(?:[\s,-]|$)"
    first1 = a1.str.extract(num, 1).cast(pl.Int64, strict=False)
    first2 = a2.str.extract(num, 1).cast(pl.Int64, strict=False)
    return pairs.with_columns(
        n2_tokens=n2.str.split(" ").list.len().cast(pl.Int16),
        n2_upper=(n2 == n2.str.to_uppercase()).cast(pl.Int8),
        n2_legal=n2.str.to_lowercase().str.contains(LEGAL).cast(pl.Int8),
        n2_domain=n2.str.to_lowercase().str.contains(r"\.(com|net|org|in|co|biz|us)\b|www").cast(pl.Int8),
        n2_digits=n2.str.contains(r"\d").cast(pl.Int8),
        n2_ascii=n2.str.contains(r"^[\x00-\x7f]*$").cast(pl.Int8),
        n1_tokens=n1.str.split(" ").list.len().cast(pl.Int16),
        n1_legal=n1.str.to_lowercase().str.contains(LEGAL).cast(pl.Int8),
        a2_hash=a2.str.contains(r"#").cast(pl.Int8),
        a2_null=a2.str.to_lowercase().str.contains(r"\bnull\b|n/a").cast(pl.Int8),
        a2_upper=(a2 == a2.str.to_uppercase()).cast(pl.Int8),
        num1=first1.is_not_null().cast(pl.Int8),
        num2=first2.is_not_null().cast(pl.Int8),
        num_eq=pl.when(first1.is_null() | first2.is_null()).then(-1).otherwise((first1 == first2).cast(pl.Int8)).cast(pl.Int8),
        a_len_ratio=(a2.str.len_chars() / a1.str.len_chars().clip(1)).cast(pl.Float32),
        a2_state_full=(a2.str.split(",").list.len().cast(pl.Int16) - a1.str.split(",").list.len().cast(pl.Int16)),
        acro_eq=((n2k == acro) & (n2k.str.len_chars() >= 2)).cast(pl.Int8),
        acro_core_eq=((n2k == acro_core) & (n2k.str.len_chars() >= 2)).cast(pl.Int8),
        dom_eq=(stem == cat1).cast(pl.Int8),
        dom_core_eq=(stem == cat_core).cast(pl.Int8),
        dom_in=((stem.str.len_chars() >= 4) & cat1.str.contains(stem, literal=True)).cast(pl.Int8),
        stem_eq=(n2k == cat_core).cast(pl.Int8),
        a_order=(a1.str.to_lowercase().str.split(",").list.first().str.strip_chars()
                 == a2.str.to_lowercase().str.split(",").list.first().str.strip_chars()).cast(pl.Int8),
    ).drop([] if keep_text else ["n1", "n2", "a1", "a2"])
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 4}
ROUNDS = 300


def features(pairs: pl.DataFrame, base: pl.DataFrame, band: pl.DataFrame) -> pl.DataFrame:
    """Join the S1-level context: base link count and the S1's best band score."""
    nb = base.group_by("s1").len().rename({"len": "nb"})
    xmax = band.group_by("s1").agg(xmax=pl.col("x").max())
    return (pairs.join(nb, on="s1", how="left").join(xmax, on="s1", how="left")
            .with_columns(pl.col("nb").fill_null(0), india=(pl.col("country") == "India").cast(pl.Int8)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", default="data/addronly/train_pairs.parquet")
    parser.add_argument("--save", default="")
    parser.add_argument("--no-shape", action="store_true", help="leave out the name and address shape features")
    parser.add_argument("--dump", default="", help="write the out-of-fold best candidate per record to this parquet")
    parser.add_argument("--keep-outside", action="store_true",
                        help="keep records whose owner is outside the hold-out (on test the owner's own links take them)")
    args = parser.parse_args()
    band = pl.read_parquet(f"{ROOT}r2_band.parquet", columns=["s1", "s23"]).join(
        pl.read_parquet(f"{ROOT}xedit_holdout_scores.parquet", columns=["s1", "s23", SCORE]), on=["s1", "s23"]).rename({SCORE: "x"})
    base = band.filter((pl.col("x") >= 0.5) & (pl.col("x") >= 0.85 * pl.col("x").max().over("s1"))).select("s1", "s23")
    ids = pl.read_parquet(f"{ROOT}holdout_ids.parquet").filter(pl.col("country") != "France")
    truth = pl.read_parquet(f"{ROOT}truth.parquet").join(ids, on="s1", how="semi")
    pairs = (pl.read_parquet(args.pairs).join(ids, on="s1")
             .join(base.select("s23").unique(), on="s23", how="anti"))
    if not args.keep_outside:
        owned = records.load_truth().join(ids, on="s1", how="anti").select("s23").unique()
        pairs = pairs.join(owned, on="s23", how="anti")
    pairs = features(pairs, base, band).join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["s1", "s23"], how="left").with_columns(pl.col("y").fill_null(0))
    print(f"candidates {pairs.height}, true {pairs['y'].sum()}, per country {dict(pairs.group_by('country').agg(pl.col('y').sum()).iter_rows())}", flush=True)
    cols = FEATURES if args.no_shape else FEATURES + SHAPE
    if not args.no_shape:
        pairs = shape(pairs, "train")
    fold = (pairs["s1"].hash(7) % 5).to_numpy()
    X, y = pairs.select(cols).to_numpy().astype(np.float32), pairs["y"].to_numpy()
    oof = np.zeros(len(y))
    for k in range(5):
        tr = fold != k
        model = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), ROUNDS)
        oof[~tr] = model.predict(X[~tr])
    pairs = pairs.with_columns(prob=pl.Series(oof))
    if args.save:
        lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS).save_model(args.save)
    # an S2/S3 record goes to at most one S1: keep its most probable candidate
    best = pairs.sort("prob", descending=True).group_by("s23", maintain_order=True).head(1)
    if args.dump:
        best.write_parquet(args.dump)
    for country in ("US", "India"):
        s1_ids = ids.filter(pl.col("country") == country).select("s1")
        t = truth.join(s1_ids, on="s1", how="semi")
        b = base.join(s1_ids, on="s1", how="semi")
        f0 = metric.per_entity(b, t, s1_ids)["f05"].mean()
        c = best.filter(pl.col("country") == country)
        for cut in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            add = c.filter(pl.col("prob") >= cut)
            f1 = metric.per_entity(pl.concat([b, add.select("s1", "s23")]), t, s1_ids)["f05"].mean()
            print(f"{country} prob>={cut}: add {add.height}, precision {add['y'].mean() if add.height else 0:.3f}, "
                  f"F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)


if __name__ == "__main__":
    main()
