"""Write a probe that adds address-only links (``addr_only_match.py`` pairs, ``addr_only_eval.py --save`` booster) to an uploaded submission.

Reads ``--base`` (default ``output/matching_results.tsv``), keeps the address-only test pairs whose S2/S3 record the
base leaves unlinked, scores them with the booster fitted on the US/India hold-out (features as in
``addr_only_eval.py``: the base link count of the S1 and its best ``xedit`` band score in
``data/kaggle_edit_r2/test_scores_r.parquet``), keeps each record's most probable S1 and adds the links at or above
``--cut`` for the countries in ``--countries``. The added pairs are also appended to the candidate lists.

Run from the repository root: ``PYTHONPATH=. python stages/addr_only_probe.py --countries US,India --cut 0.8 --name probe_addronly``.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import records
from stages.addr_only_eval import FEATURES, SHAPE, shape


def read_lists(path: Path, column: str) -> pl.DataFrame:
    return (pl.read_csv(path, separator="\t", infer_schema=False).select("source1_entity_id", e=pl.col(column).str.split(","))
            .explode("e").drop_nulls().filter(pl.col("e") != ""))


def write_lists(ids1: pl.DataFrame, pairs: pl.DataFrame, column: str, path: Path) -> None:
    rows = ids1.select("source1_entity_id").join(pairs.group_by("source1_entity_id").agg(pl.col("e").unique().sort().str.join(",").alias(column)),
                                                 on="source1_entity_id", how="left").with_columns(pl.col(column).fill_null(""))
    rows.write_csv(path, separator="\t", quote_style="never")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="output/matching_results.tsv")
    parser.add_argument("--pairs", default="data/addronly/test_pairs_k300.parquet")
    parser.add_argument("--model", default="data/addronly/booster_shape.txt")
    parser.add_argument("--countries", default="India")
    parser.add_argument("--cut", type=float, default=0.8)
    parser.add_argument("--name", required=True, help="probe directory name under output/probes/")
    args = parser.parse_args()
    s1, s23 = records.load_split("test")
    ids1, ids2 = s1.select(s1="idx", source1_entity_id="entity_id", country="country"), s23.select(s23="idx", e="entity_id")
    del s1, s23
    base = Path(args.base)
    links = read_lists(base, "matched_entity_ids").join(ids1, on="source1_entity_id").join(ids2, on="e")
    nb = links.group_by("s1").len().rename({"len": "nb"})
    xmax = pl.read_parquet("data/kaggle_edit_r2/test_scores_r.parquet").group_by("s1").agg(xmax=pl.col("xedit").max())
    pairs = (pl.read_parquet(args.pairs).join(ids1.select("s1", "country"), on="s1").filter(pl.col("country").is_in(args.countries.split(",")))
             .join(links.select("s23").unique(), on="s23", how="anti")
             .join(nb, on="s1", how="left").join(xmax, on="s1", how="left")
             .with_columns(pl.col("nb").fill_null(0), india=(pl.col("country") == "India").cast(pl.Int8)))
    pairs = shape(pairs, "test")
    model = lgb.Booster(model_file=args.model)
    pairs = pairs.with_columns(prob=pl.Series(model.predict(pairs.select(FEATURES + SHAPE).to_numpy().astype(np.float32))))
    add = (pairs.sort("prob", descending=True).group_by("s23", maintain_order=True).head(1).filter(pl.col("prob") >= args.cut)
           .join(ids1.select("s1", "source1_entity_id"), on="s1").join(ids2, on="s23"))
    pairs.select("s1", "s23", "country", "prob").write_parquet(Path(args.pairs).with_name(f"scored_{args.name}.parquet"))
    print(f"added per country: {dict(add.group_by('country').len().iter_rows())}; S1 touched {add['s1'].n_unique()}; "
          f"S1 empty before {add.join(nb, on='s1', how='anti')['s1'].n_unique()}", flush=True)
    out = Path("output/probes") / args.name
    out.mkdir(parents=True, exist_ok=True)
    kept = pl.concat([links.select("source1_entity_id", "e"), add.select("source1_entity_id", "e")])
    write_lists(ids1, kept, "matched_entity_ids", out / "matching_results.tsv")
    cands = pl.concat([read_lists(base.parent / "candidate_pairs.tsv", "candidate_entity_ids"), add.select("source1_entity_id", "e")])
    write_lists(ids1, cands, "candidate_entity_ids", out / "candidate_pairs.tsv")
    print(f"{out}: links {links.height} -> {kept.height}", flush=True)


if __name__ == "__main__":
    main()
