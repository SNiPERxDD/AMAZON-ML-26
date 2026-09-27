"""Assign no-address S2/S3 records to an S1 with a learned model of the generator's word edits (a noisy channel).

Most no-address records are not exact copies of an S1 name but variants: the generator adds legal forms and filler
words ("Ltd", "Corp", "The", "Center") and drops others ("Limited", "LLC", "Private"). Many S1 share a name core, so a
name score cannot tell which of them owns the record, but the words that differ can: "Surgical Associates Corp" is a
likelier edit of "Surgical Associates" than of "Surgical Associates LLC" when "Corp" is often added and "LLC" often
dropped. Word add rates and per-word drop rates are learned from no-address train records whose name core is unique
to their owner; every candidate S1 sharing the record's name core (in the country, groups of at most ``MAX_K``) is
scored by the log-likelihood of the added and kept or dropped words, and a record goes to its best S1 when the margin
over the second reaches ``--margin`` (a unique core needs no margin).

``--eval`` measures the added links on the US/India hold-out on top of the 0.5/0.85 ``xedit`` rule.
``--probe NAME`` adds the links to ``--base`` for ``--countries`` and writes ``output/probes/NAME/``.

Run from the repository root: ``PYTHONPATH=. python stages/noaddr_channel.py --eval``.
"""

import argparse
import math
from pathlib import Path

import polars as pl

from pipeline import metric, records
from stages.addr_only_probe import read_lists, write_lists
from stages.france_rescue import clean, name_core

ROOT = "data/kaggle_edit_r2/"
SCORE = "H+words + all crossed"
MAX_K = 50


def no_address() -> pl.Expr:
    a = pl.col("business_address")
    return a.is_null() | a.str.strip_chars().str.to_lowercase().is_in(["", "none", "null", "nan", "<null>", "n/a"])


def words(expr: pl.Expr) -> pl.Expr:
    return clean(expr).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def groups(recs: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Pair each no-address record (s23, n2, country) with every S1 of the country sharing its name core."""
    c = (s1.select(s1="idx", n1="business_name", country="country").with_columns(core=name_core(pl.col("n1")))
         .with_columns(k=pl.len().over("country", "core")).filter((pl.col("k") <= MAX_K) & (pl.col("core") != "")))
    g = recs.with_columns(core=name_core(pl.col("n2"))).join(c, on=["country", "core"])
    return g.with_columns(t1=words(pl.col("n1")), t2=words(pl.col("n2")))


def learn(split_s1: pl.DataFrame, split_s23: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Word add log-rates and per-word drop rates from train no-address records whose name core is unique to the owner."""
    truth = records.load_truth(split_s1, split_s23)
    recs = split_s23.filter(no_address()).select(s23="idx", n2="business_name", country="country").join(truth.select("s23", owner="s1"), on="s23")
    g = groups(recs, split_s1).filter((pl.col("k") == 1) & (pl.col("s1") == pl.col("owner")))
    n = g.height
    add = (g.select(w=pl.col("t2").list.set_difference("t1")).explode("w").drop_nulls().group_by("w").len()
           .select("w", la=((pl.col("len") + 0.1) / n).log()))
    dropped = g.select(w=pl.col("t1").list.set_difference("t2")).explode("w").drop_nulls().group_by("w").len()
    occ = g.select(w=pl.col("t1")).explode("w").group_by("w").len().rename({"len": "occ"})
    drop = (dropped.join(occ, on="w", how="right").fill_null(0)
            .select("w", pdrop=((pl.col("len") + 0.1) / (pl.col("occ") + 1)).clip(1e-4, 0.999)))
    return add, drop, n


def score(g: pl.DataFrame, add: pl.DataFrame, drop: pl.DataFrame, n: int) -> pl.DataFrame:
    """Channel log-likelihood per (record, S1) and, per record, the best S1 with its margin over the second."""
    g = g.with_row_index("pid")
    a = (g.select("pid", w=pl.col("t2").list.set_difference("t1")).explode("w").drop_nulls().join(add, on="w", how="left")
         .with_columns(pl.col("la").fill_null(math.log(0.1 / n))).group_by("pid").agg(sa=pl.col("la").sum()))
    d = (g.select("pid", "t1", "t2").explode("t1").rename({"t1": "w"}).drop_nulls("w").join(drop, on="w", how="left")
         .with_columns(pl.col("pdrop").fill_null(0.05), kept=pl.col("t2").list.contains(pl.col("w")))
         .group_by("pid").agg(sd=pl.when(pl.col("kept")).then((1 - pl.col("pdrop")).log()).otherwise(pl.col("pdrop").log()).sum()))
    g = (g.join(a, on="pid", how="left").join(d, on="pid", how="left").with_columns(pl.col("sa", "sd").fill_null(0.0))
         .with_columns(ll=pl.col("sa") + pl.col("sd")))
    return (g.sort("ll", descending=True).group_by("s23", maintain_order=True)
            .agg(pl.col("s1").first(), pl.col("country").first(), pl.col("k").first(), ll=pl.col("ll").first(), ll2=pl.col("ll").get(1, null_on_oob=True))
            .with_columns(margin=pl.when(pl.col("k") == 1).then(99.0).otherwise(pl.col("ll") - pl.col("ll2"))))


def evaluate() -> None:
    s1, s23 = records.load_split("train")
    add, drop, n = learn(s1, s23)
    band = pl.read_parquet(f"{ROOT}r2_band.parquet", columns=["s1", "s23"]).join(
        pl.read_parquet(f"{ROOT}xedit_holdout_scores.parquet", columns=["s1", "s23", SCORE]), on=["s1", "s23"]).rename({SCORE: "x"})
    base = band.filter((pl.col("x") >= 0.5) & (pl.col("x") >= 0.85 * pl.col("x").max().over("s1"))).select("s1", "s23")
    ids = pl.read_parquet(f"{ROOT}holdout_ids.parquet").filter(pl.col("country") != "France")
    truth = pl.read_parquet(f"{ROOT}truth.parquet").join(ids, on="s1", how="semi").select("s1", "s23")
    owned_out = records.load_truth(s1, s23).join(ids, on="s1", how="anti").select("s23").unique()
    recs = (s23.filter(no_address()).select(s23="idx", n2="business_name", country="country")
            .join(base.select("s23").unique(), on="s23", how="anti"))
    best = score(groups(recs, s1), add, drop, n).join(ids.select("s1"), on="s1", how="semi")
    best = best.join(truth.with_columns(t=pl.lit(True)), on=["s1", "s23"], how="left").with_columns(pl.col("t").fill_null(False),
                                                                                                   out=pl.col("s23").is_in(owned_out["s23"].implode()))
    print(f"learned from {n} unique-core pairs; hold-out candidates {best.height}, of them owned outside the hold-out {best['out'].sum()}", flush=True)
    for country in ("US", "India"):
        sid = ids.filter(pl.col("country") == country).select("s1")
        t = truth.join(sid, on="s1", how="semi")
        b = base.join(sid, on="s1", how="semi")
        f0 = metric.per_entity(b, t, sid)["f05"].mean()
        c = best.filter(pl.col("country") == country)
        for keep_out in (False, True):
            cc = c if keep_out else c.filter(~pl.col("out"))
            for m in (1, 2, 3, 5, 99):
                a = cc.filter(pl.col("margin") >= m)
                f1 = metric.per_entity(pl.concat([b, a.select("s1", "s23")]), t, sid)["f05"].mean()
                print(f"{country} {'all' if keep_out else 'in-holdout-owned'} margin>={m}: add {a.height} (unique core {a.filter(pl.col('k') == 1).height}), "
                      f"precision {a['t'].mean() if a.height else 0:.3f}, F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)


def probe(name: str, base_path: str, margin: float, countries: list[str]) -> None:
    tr1, tr23 = records.load_split("train")
    add, drop, n = learn(tr1, tr23)
    del tr1, tr23
    s1, s23 = records.load_split("test")
    ids1, ids2 = s1.select(s1="idx", source1_entity_id="entity_id", country="country"), s23.select(s23="idx", e="entity_id")
    base = Path(base_path)
    links = read_lists(base, "matched_entity_ids").join(ids1, on="source1_entity_id").join(ids2, on="e")
    recs = (s23.filter(no_address() & pl.col("country").is_in(countries)).select(s23="idx", n2="business_name", country="country")
            .join(links.select("s23").unique(), on="s23", how="anti"))
    best = score(groups(recs, s1.filter(pl.col("country").is_in(countries))), add, drop, n).filter(pl.col("margin") >= margin)
    best = best.join(ids1.select("s1", "source1_entity_id"), on="s1").join(ids2, on="s23")
    print(f"unlinked no-address records {recs.height}; added {best.height} per country {dict(best.group_by('country').len().iter_rows())}, "
          f"unique core {best.filter(pl.col('k') == 1).height}", flush=True)
    out = Path("output/probes") / name
    out.mkdir(parents=True, exist_ok=True)
    kept = pl.concat([links.select("source1_entity_id", "e"), best.select("source1_entity_id", "e")])
    write_lists(ids1, kept, "matched_entity_ids", out / "matching_results.tsv")
    cands = pl.concat([read_lists(base.parent / "candidate_pairs.tsv", "candidate_entity_ids"), best.select("source1_entity_id", "e")])
    write_lists(ids1, cands, "candidate_entity_ids", out / "candidate_pairs.tsv")
    best.select("s1", "s23", "country", "k", "margin").write_parquet(out.parent / f"{name}_added.parquet")
    print(f"{out}: links {links.height} -> {kept.height}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--probe", default="")
    parser.add_argument("--base", default="output/matching_results.tsv")
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--countries", default="US,India")
    args = parser.parse_args()
    if args.eval:
        evaluate()
    if args.probe:
        probe(args.probe, args.base, args.margin, args.countries.split(","))


if __name__ == "__main__":
    main()
