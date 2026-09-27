"""Profile train true links that are still never scored after blocking, second-round and third-round candidates.

Splits the misses by what the two normalised records share (house number, street, name
tokens, compact-name prefix, empty S2/S3 address) and prints sampled raw records.

Run from the repository root: ``PYTHONPATH=. python stages/missed_link_profile.py [EXAMPLES]``.
"""

import sys
from pathlib import Path

import polars as pl

from pipeline import features, records


def missed() -> pl.DataFrame:
    """Return the true train links absent from blocking, second-round and (when present) third-round pairs."""
    scored = pl.concat([
        pl.scan_parquet("data/candidates/train_pairs.parquet").select("s1", "s23"),
        pl.scan_parquet(f"{features.FEATURE_DIR}train/sibling_*.parquet").select("s1", "s23"),
        *[pl.scan_parquet(str(path)).select("s1", "s23") for path in Path(f"{features.FEATURE_DIR}train").glob("namepair_*.parquet")],
    ])
    return features.truth_links().lazy().join(scored, on=["s1", "s23"], how="anti").collect()


def main(examples: int) -> None:
    """Print the share of each overlap group among missed links and sampled raw pairs."""
    links = missed()
    total = features.truth_links().height
    print(f"missed links={links.height} ({links.height / total:.4f} of {total})", flush=True)
    cols = ["idx", "country", "core", "compact", "numbers", "words", "house", "street"]
    s1 = pl.scan_parquet(records.records_path("train", "s1")).select(cols)
    s23 = pl.scan_parquet(records.records_path("train", "s23")).select(cols)
    pairs = (
        links.lazy().join(s1, left_on="s1", right_on="idx").join(s23, left_on="s23", right_on="idx", suffix="_r").collect()
    )
    tok1 = pl.read_parquet(records.tokens_path("train", "s1")).rename({"idx": "s1"}).join(links.select("s1").unique(), on="s1", how="semi")
    tok23 = pl.read_parquet(records.tokens_path("train", "s23")).rename({"idx": "s23"}).join(links.select("s23").unique(), on="s23", how="semi")
    shared = links.join(tok1, on="s1").join(tok23, on=["s23", "token"]).group_by("s1", "s23").agg(shared_tokens=pl.len())
    pairs = pairs.join(shared, on=["s1", "s23"], how="left").with_columns(
        house_in=pl.col("numbers_r").list.contains(pl.col("house")).fill_null(False),
        street_eq=(pl.col("street") == pl.col("street_r")).fill_null(False),
        word_shared=pl.col("words").list.set_intersection(pl.col("words_r")).list.len().fill_null(0) > 0,
        token=pl.col("shared_tokens").fill_null(0) > 0,
        prefix4=pl.col("compact").str.slice(0, 4) == pl.col("compact_r").str.slice(0, 4),
        empty_addr=pl.col("numbers_r").list.len().fill_null(0) + pl.col("words_r").list.len().fill_null(0) == 0,
        same_country=pl.col("country") == pl.col("country_r"),
    )
    flags = ["house_in", "street_eq", "word_shared", "token", "prefix4", "empty_addr", "same_country"]
    print(pairs.select([pl.col(f).mean().round(3) for f in flags]))
    with pl.Config(tbl_rows=30):
        print(pairs.group_by("country", "token", "house_in", "empty_addr").agg(n=pl.len()).sort("n", descending=True).head(20))
    raw1 = features.raw_scan("train", "s1").rename({"idx": "s1", "business_name": "n1", "business_address": "a1"})
    raw23 = features.raw_scan("train", "s23").rename({"idx": "s23", "business_name": "n2", "business_address": "a2"})
    sample = pairs.sample(examples, seed=5).lazy().join(raw1, on="s1").join(raw23, on="s23").collect()
    with pl.Config(tbl_rows=examples, fmt_str_lengths=50, tbl_width_chars=250):
        print(sample.select("country", "token", "house_in", pl.col("n1").str.slice(0, 40), pl.col("n2").str.slice(0, 40),
                            pl.col("a1").str.slice(0, 45), pl.col("a2").str.slice(0, 45)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
