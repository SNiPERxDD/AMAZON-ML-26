"""Write a France-only rule variant of a probe (``BASE``): US and India rows are copied unchanged, France is re-decided.

France keeps ``refine2`` probabilities in the xedit probes, so its rows are rebuilt from ``train.prob_scan("test", ...)``
restricted to France S1 (lazily, before collecting). The script first rebuilds France at the current ``TEST_RULES`` and
checks it matches ``BASE`` exactly, then writes ``output/probes/<--name>{relative*1000}/`` with the new
France cut, a hard link to the base probe's candidate pairs, and the links removed.

Run from the repository root: ``PYTHONPATH=. python stages/france_rule_submit.py --relative 0.95``.
"""

import argparse
import json
import os
from pathlib import Path

import polars as pl

from pipeline import features, records, refine2, submit, train

BASE = "output/probes/probe_learned/"


def france_rows(preds: pl.DataFrame, rule: dict, relative: float, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """Return the France matching rows under France at (current threshold, ``relative``)."""
    rules = submit.TEST_RULES | {"France": (submit.TEST_RULES["France"][0], relative)}
    kept = submit.decide_by_country(preds, rule, rules)
    return submit.id_lists(kept, s1.select(s1="idx", source1_entity_id="entity_id"), s23.select(s23="idx", entity_id="entity_id"), "matched_entity_ids")


def main() -> None:
    """Rebuild France, check it against ``BASE``, and write the variant."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relative", type=float, default=0.95)
    parser.add_argument("--base", default=BASE, help="probe directory whose US/India rows and candidate pairs are kept")
    parser.add_argument("--name", default="probe_learned_fr", help="output name prefix; the relative cut x1000 is appended")
    args = parser.parse_args()
    base_dir = args.base.rstrip("/") + "/"
    s1, s23 = records.load_split("test")
    france = s1.filter(pl.col("country") == "France").select(s1="idx")
    s1_fr = s1.join(france.rename({"s1": "idx"}), on="idx", how="semi")
    preds = train.with_maxima(train.prob_scan("test", features.FEATURE_DIR, refine2.PREFIX).join(france.lazy(), on="s1", how="semi")).collect()
    rule = json.loads(Path(train.RULE_PATH).read_text())
    base = pl.read_csv(f"{base_dir}matching_results.tsv", separator="\t", infer_schema=False)
    fr_ids = s1_fr.select(source1_entity_id=pl.col("entity_id").cast(pl.String))
    base_fr = base.join(fr_ids, on="source1_entity_id", how="semi")
    current = france_rows(preds, rule, submit.TEST_RULES["France"][1], s1_fr, s23).with_columns(pl.all().cast(pl.String))
    same = base_fr.sort("source1_entity_id").fill_null("").equals(current.sort("source1_entity_id").fill_null(""))
    print(f"France S1 {s1_fr.height}; rebuilt France at {submit.TEST_RULES['France']} matches {base_dir}: {same}", flush=True)
    if not same:
        raise SystemExit("France rebuild differs from BASE; not writing the variant")
    new = france_rows(preds, rule, args.relative, s1_fr, s23).with_columns(pl.all().cast(pl.String))
    out = f"output/probes/{args.name}{round(args.relative * 1000)}/"
    Path(out).mkdir(parents=True, exist_ok=True)
    rows = pl.concat([base.join(fr_ids, on="source1_entity_id", how="anti"), new])
    order = base.select("source1_entity_id").with_row_index("i")
    rows.join(order, on="source1_entity_id", how="left").sort("i").drop("i").write_csv(f"{out}matching_results.tsv", separator="\t", quote_style="never")
    if not Path(f"{out}candidate_pairs.tsv").exists():
        os.link(f"{base_dir}candidate_pairs.tsv", f"{out}candidate_pairs.tsv")

    def count(frame: pl.DataFrame) -> int:
        return frame.select(pl.col("matched_entity_ids").str.split(",").explode().drop_nulls().ne("").sum()).item()

    print(f"{out}: France links {count(base_fr)} -> {count(new)}; France S1 changed "
          f"{new.join(base_fr, on=['source1_entity_id', 'matched_entity_ids'], how='anti', join_nulls=True).height}; rows {rows.height} vs {base.height}", flush=True)


if __name__ == "__main__":
    main()
