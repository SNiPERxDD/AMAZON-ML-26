"""Link no-address S2/S3 records whose name is a typo of an S1 name, with word-deletion blocking and a hold-out-fitted classifier.

A third of the no-address true links the base misses carry a name whose core differs from the owner's by a typo in
one word ("Education Socotey", "Blue Mashse", "TP Lisftyle") or by one word added or dropped ("Internal Clinic LLC
Center" for "Internal Medicine Clinic LLC"); no S1 shares the record's name core, so the exact-core channel
(``noaddr_channel.py``) cannot reach them. Blocking keys are the name core and the core with one word removed; a record
and an S1 are paired when they share a key that at most ``MAX_KEY`` S1 of the country carry. Pairs are scored with
rapidfuzz on the cores and the cleaned names, and each record gets its best and second-best scores (a typo must be
closer to its owner than to the owner's same-name neighbours). A LightGBM classifier is cross-fitted over five S1
folds on the US/India hold-out. Records owned outside the hold-out stay in, sampled at the base's no-address miss rate,
because on test the base links the rest to their owners.

``--eval`` prints the added links' precision and F0.5 change per country over a grid of cuts and saves the booster.
``--probe NAME --cut C`` adds the links to ``--base`` for ``--countries`` and writes ``output/probes/NAME/``.

Run from the repository root: ``PYTHONPATH=. python stages/noaddr_fuzzy.py --eval``.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from pipeline import metric, records
from stages.addr_only_probe import read_lists, write_lists
from stages.france_rescue import clean, name_core
from stages.noaddr_channel import no_address

ROOT = "data/kaggle_edit_r2/"
SCORE = "H+words + all crossed"
MAX_KEY = 30
BOOSTER = "data/addronly/booster_noaddr_fuzzy.txt"
FEATURES = ["core_ratio", "name_ratio", "name_sort", "core_best", "core_second", "n_close", "rank", "n_cand", "w1", "w2",
            "dw", "k1", "k2", "exact_key", "nb", "india"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 4}
ROUNDS = 300


def core_keys(frame: pl.DataFrame, id_col: str, name_col: str) -> pl.DataFrame:
    """(id, country, key, exact): the sorted name core and each core with one word removed (cores of two or more words)."""
    f = frame.select(id_col, "country", core=name_core(pl.col(name_col))).filter(pl.col("core") != "")
    full = f.select(id_col, "country", key="core", exact=pl.lit(True))
    w = (f.with_columns(words=pl.col("core").str.split(" ")).filter(pl.col("words").list.len() >= 2)
         .with_columns(pos=pl.int_ranges(pl.col("words").list.len())).explode("pos"))
    w = w.with_columns(row=pl.int_range(pl.len())).explode("words").with_columns(j=pl.int_range(pl.len()).over("row"))
    drop = (w.filter(pl.col("j") != pl.col("pos")).group_by("row", maintain_order=True)
            .agg(pl.col(id_col).first(), pl.col("country").first(), key=pl.col("words").str.join(" "))
            .select(id_col, "country", "key", exact=pl.lit(False)))
    return pl.concat([full, drop]).unique([id_col, "key"])


def pairs_for(recs: pl.DataFrame, s1: pl.DataFrame, k1: pl.DataFrame) -> pl.DataFrame:
    """Candidate (s23, s1) pairs sharing a blocking key, with name scores and per-record context."""
    k2 = core_keys(recs, "s23", "n2")
    p = (k2.join(k1, on=["country", "key"]).group_by("s23", "s1")
         .agg(pl.col("country").first(), exact_key=(pl.col("exact") & pl.col("exact_right")).any().cast(pl.Int8)))
    p = (p.join(recs.select("s23", "n2"), on="s23").join(s1.select(s1="idx", n1="business_name"), on="s1")
         .with_columns(c1=name_core(pl.col("n1")), c2=name_core(pl.col("n2")), l1=clean(pl.col("n1")), l2=clean(pl.col("n2"))))
    p = p.with_columns(
        core_ratio=pl.Series(cpdist(p["c1"].to_list(), p["c2"].to_list(), scorer=fuzz.ratio, workers=4), dtype=pl.Float32),
        name_ratio=pl.Series(cpdist(p["l1"].to_list(), p["l2"].to_list(), scorer=fuzz.ratio, workers=4), dtype=pl.Float32),
        name_sort=pl.Series(cpdist(p["l1"].to_list(), p["l2"].to_list(), scorer=fuzz.token_sort_ratio, workers=4), dtype=pl.Float32),
        w1=pl.col("c1").str.split(" ").list.len().cast(pl.Int16), w2=pl.col("c2").str.split(" ").list.len().cast(pl.Int16))
    p = p.with_columns(dw=pl.col("w2") - pl.col("w1"), s=pl.col("core_ratio") + 0.5 * pl.col("name_ratio"))
    p = p.with_columns(
        rank=pl.col("s").rank("ordinal", descending=True).over("s23").cast(pl.Int16),
        n_cand=pl.len().over("s23").cast(pl.Int16),
        core_best=pl.col("core_ratio").max().over("s23"),
        n_close=(pl.col("s") >= pl.col("s").max().over("s23") - 5).sum().over("s23").cast(pl.Int16))
    second = p.filter(pl.col("rank") == 2).select("s23", core_second="core_ratio")
    cores = s1.select("country", c1=name_core(pl.col("business_name"))).group_by("country", "c1").len()
    p = (p.join(second, on="s23", how="left").with_columns(pl.col("core_second").fill_null(0))
         .join(cores.rename({"len": "k1"}), on=["country", "c1"], how="left")
         .join(cores.rename({"c1": "c2", "len": "k2"}), on=["country", "c2"], how="left").with_columns(pl.col("k1", "k2").fill_null(0)))
    return p.filter(pl.col("rank") <= 3).drop("n1", "n2", "c1", "c2", "l1", "l2", "s")


def s1_keys(s1: pl.DataFrame) -> pl.DataFrame:
    k1 = core_keys(s1.select(s1="idx", n1="business_name", country="country"), "s1", "n1")
    return k1.join(k1.group_by("country", "key").len().filter(pl.col("len") <= MAX_KEY).select("country", "key"), on=["country", "key"], how="semi")


def evaluate(save: str) -> None:
    s1, s23 = records.load_split("train")
    s1 = s1.filter(pl.col("country").is_in(["US", "India"]))
    band = pl.read_parquet(f"{ROOT}r2_band.parquet", columns=["s1", "s23"]).join(
        pl.read_parquet(f"{ROOT}xedit_holdout_scores.parquet", columns=["s1", "s23", SCORE]), on=["s1", "s23"]).rename({SCORE: "x"})
    base = band.filter((pl.col("x") >= 0.5) & (pl.col("x") >= 0.85 * pl.col("x").max().over("s1"))).select("s1", "s23")
    ids = pl.read_parquet(f"{ROOT}holdout_ids.parquet").filter(pl.col("country") != "France")
    truth = pl.read_parquet(f"{ROOT}truth.parquet").join(ids, on="s1", how="semi").select("s1", "s23")
    na = s23.filter(no_address()).select(s23="idx", n2="business_name", country="country")
    tna = truth.join(na.select("s23", "country"), on="s23")
    miss = {c: 1 - tna.filter(pl.col("country") == c).join(base, on=["s1", "s23"], how="semi").height / tna.filter(pl.col("country") == c).height
            for c in ("US", "India")}
    owned_out = records.load_truth(s1, s23).join(ids, on="s1", how="anti").select("s23").unique()
    recs = na.join(base.select("s23").unique(), on="s23", how="anti")
    # on test the base links an outside-owned record to its owner unless it misses it: keep them at the miss rate
    u = (pl.col("s23").hash(3) % 1000) / 1000
    recs = recs.filter(~pl.col("s23").is_in(owned_out["s23"].implode()) | (u < pl.col("country").replace_strict(miss, default=0.5)))
    p = pairs_for(recs, s1, s1_keys(s1)).join(ids.select("s1"), on="s1", how="semi")
    nb = base.group_by("s1").len().rename({"len": "nb"})
    p = (p.join(nb, on="s1", how="left").with_columns(pl.col("nb").fill_null(0), india=(pl.col("country") == "India").cast(pl.Int8))
         .join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["s1", "s23"], how="left").with_columns(pl.col("y").fill_null(0)))
    print(f"miss rates {miss}; pairs {p.height}, records {p['s23'].n_unique()}, true {p['y'].sum()}", flush=True)
    fold = (p["s1"].hash(7) % 5).to_numpy()
    X, y = p.select(FEATURES).to_numpy().astype(np.float32), p["y"].to_numpy()
    oof = np.zeros(len(y))
    for k in range(5):
        tr = fold != k
        oof[~tr] = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), ROUNDS).predict(X[~tr])
    if save:
        lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS).save_model(save)
    best = p.with_columns(prob=pl.Series(oof)).sort("prob", descending=True).group_by("s23", maintain_order=True).head(1)
    for country in ("US", "India"):
        sid = ids.filter(pl.col("country") == country).select("s1")
        t = truth.join(sid, on="s1", how="semi")
        b = base.join(sid, on="s1", how="semi")
        f0 = metric.per_entity(b, t, sid)["f05"].mean()
        c = best.filter(pl.col("country") == country)
        for cut in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            a = c.filter(pl.col("prob") >= cut)
            f1 = metric.per_entity(pl.concat([b, a.select("s1", "s23")]), t, sid)["f05"].mean()
            print(f"{country} prob>={cut}: add {a.height} (exact core {a['exact_key'].sum()}), precision {a['y'].mean() if a.height else 0:.3f}, "
                  f"F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)


def probe(name: str, base_path: str, cut: float, countries: list[str], model_path: str) -> None:
    s1, s23 = records.load_split("test")
    ids1, ids2 = s1.select(s1="idx", source1_entity_id="entity_id", country="country"), s23.select(s23="idx", e="entity_id")
    base = Path(base_path)
    links = read_lists(base, "matched_entity_ids").join(ids1, on="source1_entity_id").join(ids2, on="e")
    s1 = s1.filter(pl.col("country").is_in(countries))
    recs = (s23.filter(no_address() & pl.col("country").is_in(countries)).select(s23="idx", n2="business_name", country="country")
            .join(links.select("s23").unique(), on="s23", how="anti"))
    del s23
    p = pairs_for(recs, s1, s1_keys(s1))
    nb = links.group_by("s1").len().rename({"len": "nb"})
    p = p.join(nb, on="s1", how="left").with_columns(pl.col("nb").fill_null(0), india=(pl.col("country") == "India").cast(pl.Int8))
    model = lgb.Booster(model_file=model_path)
    p = p.with_columns(prob=pl.Series(model.predict(p.select(FEATURES).to_numpy().astype(np.float32))))
    add = (p.sort("prob", descending=True).group_by("s23", maintain_order=True).head(1).filter(pl.col("prob") >= cut)
           .join(ids1.select("s1", "source1_entity_id"), on="s1").join(ids2, on="s23"))
    print(f"unlinked no-address records {recs.height}; added {add.height} per country {dict(add.group_by('country').len().iter_rows())}, "
          f"S1 empty before {add.join(nb, on='s1', how='anti')['s1'].n_unique()}", flush=True)
    out = Path("output/probes") / name
    out.mkdir(parents=True, exist_ok=True)
    kept = pl.concat([links.select("source1_entity_id", "e"), add.select("source1_entity_id", "e")])
    write_lists(ids1, kept, "matched_entity_ids", out / "matching_results.tsv")
    cands = pl.concat([read_lists(base.parent / "candidate_pairs.tsv", "candidate_entity_ids"), add.select("source1_entity_id", "e")])
    write_lists(ids1, cands, "candidate_entity_ids", out / "candidate_pairs.tsv")
    add.select("s1", "s23", "country", "prob").write_parquet(out.parent / f"{name}_added.parquet")
    print(f"{out}: links {links.height} -> {kept.height}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save", default=BOOSTER)
    parser.add_argument("--probe", default="")
    parser.add_argument("--base", default="output/matching_results.tsv")
    parser.add_argument("--cut", type=float, default=0.8)
    parser.add_argument("--countries", default="US,India")
    args = parser.parse_args()
    if args.eval:
        evaluate(args.save)
    if args.probe:
        probe(args.probe, args.base, args.cut, args.countries.split(","), args.save)


if __name__ == "__main__":
    main()
