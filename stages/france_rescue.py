"""Rescue France band pairs that the strict France rule rejects although name and address agree after generator-aware normalisation.

France has no labels, so its ``refine2`` probabilities come from US/India models and are cut hard (0.7 / relative
0.95). Reading the rejected France pairs shows plain generator variants of the S1: bracketed or dotted legal forms
("Clinique [Dame]", "Tourcoing Amicale E.U.R.L."), a typo in the street ("40 Av Dersines"), French street
abbreviations ("R.", "Av", "Rte", "N°") that the pipeline does not normalise. A pair is rescued when, after
lower-casing, stripping accents, dots and brackets and dropping legal forms and the words the generator adds or drops
(``GEN``), the name cores agree (``name`` score) and the house numbers are equal and the rest of the address agrees
(``addr`` score) or the record has no address and a unique name.

``--eval`` measures the rule on the US/India hold-out band (``refine2`` probabilities, labels ``t``) under the France
rule, i.e. France-like conditions, and prints the precision of the rescued pairs and the F0.5 change per country.
``--probe NAME`` adds the rescued France pairs to ``--base`` and writes ``output/probes/NAME/``.

Run from the repository root: ``PYTHONPATH=. python stages/france_rescue.py --eval``.
"""

import argparse
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from pipeline import features, metric, records, refine2, train
from stages.addr_only_probe import read_lists, write_lists

ROOT = "data/kaggle_edit_r2/"
GEN = ("llc|l l c|inc|incorporated|corp|corporation|ltd|limited|pvt|private|co|company|lp|llp|pc|pllc|sarl|sas|sasu|sa|sci|"
       "eurl|cie|ets|the|and|a|of|center|centre|services|service|partners|dba|smt|sri|shri|mr|dr|formerly|as|aka|doing|"
       "business|fka|nee|known|trading|enterprises|group|groupe|developpement|development|[a-z]")
STREET = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "blvd": "boulevard", "rte": "route", "ch": "chemin",
          "pl": "place", "imp": "impasse", "all": "allee", "sq": "square", "st": "saint", "ste": "sainte", "fg": "faubourg",
          "fbg": "faubourg", "qu": "quai", "crs": "cours", "pass": "passage", "rd": "road", "dr": "drive", "ln": "lane"}


def clean(expr: pl.Expr) -> pl.Expr:
    return (expr.fill_null("").str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}", "")
            .str.replace_all(r"\.", "").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())


def name_core(expr: pl.Expr) -> pl.Expr:
    return (clean(expr).str.replace_all(rf"\b({GEN})\b", " ").str.replace_all(r"\s+", " ").str.strip_chars()
            .str.split(" ").list.unique().list.sort().list.join(" "))


def addr_parts(expr: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Return the house number (leading zeros dropped) and the remaining words with French abbreviations expanded."""
    text = clean(expr.str.replace_all(r"(?i)n\s*[°º]|\bno\b\.?|#", " "))
    words = text.str.split(" ").list.eval(pl.element().replace(STREET))
    house = text.str.extract(r"(?:^|\s)0*(\d+)(?:\s|$)", 1)
    # every number of the address (leading zeros dropped): the generator's decoys keep the first number and change a
    # later one ("718 1 Capitol Road" -> "718 4 Capitol Road")
    nums = text.str.extract_all(r"\d+").list.eval(pl.element().str.replace(r"^0+(\d)", "$1")).list.join(" ")
    rest = words.list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$") & (pl.element() != ""))).list.join(" ")
    return house, rest, nums


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    p = (pairs.join(s1.select(s1="idx", n1="business_name", a1="business_address"), on="s1")
         .join(s23.select(s23="idx", n2="business_name", a2="business_address"), on="s23"))
    h1, r1, m1 = addr_parts(pl.col("a1"))
    h2, r2, m2 = addr_parts(pl.col("a2"))
    p = p.with_columns(c1=name_core(pl.col("n1")), c2=name_core(pl.col("n2")), h1=h1, h2=h2, r1=r1, r2=r2, m1=m1, m2=m2,
                       noaddr=clean(pl.col("a2")).is_in(["", "none", "null", "nan", "n a"]))
    p = p.with_columns(
        name=pl.Series(cpdist(p["c1"].to_list(), p["c2"].to_list(), scorer=fuzz.ratio, workers=4), dtype=pl.Float32),
        addr=pl.Series(cpdist(p["r1"].to_list(), p["r2"].to_list(), scorer=fuzz.token_set_ratio, workers=4), dtype=pl.Float32),
        house_eq=(pl.col("h1") == pl.col("h2")).fill_null(False), nums_eq=(pl.col("m1") == pl.col("m2")).fill_null(False))
    return p.drop("n1", "a1", "n2", "a2", "h1", "h2", "r1", "r2", "m1", "m2")


def rescue(p: pl.DataFrame, name_cut: float, addr_cut: float, noaddr: bool, all_nums: bool = False) -> pl.DataFrame:
    """Rescued pairs: name cores agree and either the address agrees with an equal house number or (optionally) the record has no address and the S1 core is unique."""
    same_nums = pl.col("nums_eq") if all_nums else pl.col("house_eq")
    with_addr = (pl.col("name") >= name_cut) & same_nums & (pl.col("addr") >= addr_cut) & ~pl.col("noaddr")
    without = (pl.col("name") >= 100) & pl.col("noaddr") & (pl.col("core_n") == 1) if noaddr else pl.lit(False)
    r = p.filter(with_addr | without)
    # one owner per record: the most probable S1
    return r.sort("p", descending=True).group_by("s23", maintain_order=True).head(1)


def core_counts(s1: pl.DataFrame) -> pl.DataFrame:
    return (s1.select(s1="idx", country="country", c=name_core(pl.col("business_name")))
            .with_columns(core_n=pl.len().over("country", "c")).select("s1", "core_n"))


def rule_keep(band: pl.DataFrame, thr: float, rel: float) -> pl.DataFrame:
    return band.filter((pl.col("p") >= thr) & (pl.col("p") >= rel * pl.col("p").max().over("s1")))


def evaluate() -> None:
    band = pl.read_parquet(f"{ROOT}r2_band.parquet", columns=["s1", "s23", "p", "t", "country"])
    ids = pl.read_parquet(f"{ROOT}holdout_ids.parquet").filter(pl.col("country") != "France")
    truth = pl.read_parquet(f"{ROOT}truth.parquet").join(ids, on="s1", how="semi").select("s1", "s23")
    s1, s23 = records.load_split("train")
    owned_out = records.load_truth(s1, s23).join(ids, on="s1", how="anti").select("s23").unique()
    cn = core_counts(s1)
    for thr, rel in ((0.7, 0.95), (0.5, 0.85)):
        base = rule_keep(band, thr, rel)
        rej = (band.join(base.select("s23").unique(), on="s23", how="anti").join(owned_out, on="s23", how="anti"))
        feats = pair_features(rej, s1, s23).join(cn, on="s1")
        print(f"=== rule {thr}/{rel}: rejected band pairs {rej.height}", flush=True)
        for country in ("US", "India"):
            s1_ids = ids.filter(pl.col("country") == country).select("s1")
            t = truth.join(s1_ids, on="s1", how="semi")
            b = base.join(s1_ids, on="s1", how="semi").select("s1", "s23")
            f0 = metric.per_entity(b, t, s1_ids)["f05"].mean()
            c = feats.filter(pl.col("country") == country)
            for name_cut, addr_cut, noaddr, all_nums in ((100, 90, False, False), (100, 90, False, True), (100, 80, False, True),
                                                         (90, 80, False, True), (100, 90, True, True), (95, 85, True, True)):
                add = rescue(c, name_cut, addr_cut, noaddr, all_nums)
                f1 = metric.per_entity(pl.concat([b, add.select("s1", "s23")]), t, s1_ids)["f05"].mean()
                print(f"{country} name>={name_cut} addr>={addr_cut} noaddr={noaddr} all_nums={all_nums}: add {add.height}, precision {add['t'].mean() if add.height else 0:.3f}, "
                      f"F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)


def probe(name: str, base_path: str, name_cut: float, addr_cut: float, noaddr: bool, all_nums: bool) -> None:
    s1, s23 = records.load_split("test")
    ids1, ids2 = s1.select(s1="idx", source1_entity_id="entity_id", country="country"), s23.select(s23="idx", e="entity_id")
    fr = ids1.filter(pl.col("country") == "France").select("s1")
    base = Path(base_path)
    links = read_lists(base, "matched_entity_ids").join(ids1, on="source1_entity_id").join(ids2, on="e")
    p = train.prob_scan("test", features.FEATURE_DIR, refine2.PREFIX).join(fr.lazy(), on="s1", how="semi").filter(pl.col("p") >= 0.05).collect()
    rej = p.join(links.select("s23").unique(), on="s23", how="anti")
    feats = pair_features(rej, s1, s23).join(core_counts(s1), on="s1")
    add = rescue(feats, name_cut, addr_cut, noaddr, all_nums).join(ids1.select("s1", "source1_entity_id"), on="s1").join(ids2, on="s23")
    print(f"France rejected pairs {rej.height}; rescued {add.height} (no address {add['noaddr'].sum()}), S1 touched {add['s1'].n_unique()}, "
          f"S1 empty before {add.join(links.select('s1').unique(), on='s1', how='anti')['s1'].n_unique()}; mean p {add['p'].mean():.3f}", flush=True)
    out = Path("output/probes") / name
    out.mkdir(parents=True, exist_ok=True)
    kept = pl.concat([links.select("source1_entity_id", "e"), add.select("source1_entity_id", "e")])
    write_lists(ids1, kept, "matched_entity_ids", out / "matching_results.tsv")
    cands = pl.concat([read_lists(base.parent / "candidate_pairs.tsv", "candidate_entity_ids"), add.select("source1_entity_id", "e")])
    write_lists(ids1, cands, "candidate_entity_ids", out / "candidate_pairs.tsv")
    add.select("s1", "s23", "p", "name", "addr", "noaddr").write_parquet(out.parent / f"{name}_added.parquet")
    print(f"{out}: links {links.height} -> {kept.height}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--probe", default="")
    parser.add_argument("--base", default="output/matching_results.tsv")
    parser.add_argument("--name-cut", type=float, default=100)
    parser.add_argument("--addr-cut", type=float, default=90)
    parser.add_argument("--noaddr", action="store_true")
    parser.add_argument("--all-nums", action="store_true", help="require every address number to agree, not only the first")
    args = parser.parse_args()
    if args.eval:
        evaluate()
    if args.probe:
        probe(args.probe, args.base, args.name_cut, args.addr_cut, args.noaddr, args.all_nums)


if __name__ == "__main__":
    main()
