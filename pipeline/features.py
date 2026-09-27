"""Compute comparison features for every candidate pair.

Two steps, each run from the repository root:

- ``python -m pipeline.features pairs train`` processes S1 in blocks of ``--chunk-s1`` rows.
  Each block's pairs come from the per-family pair files, so every pair carries a flag for
  each family that produced it. The pairs are joined with both record tables and the
  folded raw text, then name, address and per-S1 context features are computed. The result
  is written to ``data/features/{split}/part_{k:03d}.parquet``. On train the true link is
  added as ``label``.
- ``python -m pipeline.features context train`` aggregates all parts per S2/S3 record: the
  number of candidate S1, the best and second-best name ratio with the best S1, and the
  number of strong candidates. It writes ``s23_context.parquet``. :func:`load` joins this
  table and turns it into competitor features for each pair.

Country is kept for analysis only and is not meant as a model input, because France has
no labels.
"""

import argparse
import os
import time
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from pipeline import candidates, normalize, records

FEATURE_DIR = "data/features/"
STRONG_RATIO = 85
WORKERS = int(os.environ.get("RAPIDFUZZ_WORKERS", "4"))
RECORD_COLUMNS = ["idx", "country", "core", "compact", "numbers", "words", "house", "street"]


def part_path(split: str, part: int, out_dir: str = FEATURE_DIR) -> str:
    """Return the path of one feature part."""
    return f"{out_dir}{split}/part_{part:03d}.parquet"


def context_path(split: str, out_dir: str = FEATURE_DIR) -> str:
    """Return the path of the per-S2/S3 context table."""
    return f"{out_dir}{split}/s23_context.parquet"


def truth_links() -> pl.DataFrame:
    """Return train links as (``s1``, ``s23``), cached; reads only the entity-id columns."""
    path = Path(f"{records.CACHE_DIR}train_truth.parquet")
    if not path.exists():
        s1 = pl.scan_parquet(f"{records.PARQUET_DIR}train_source1.parquet").select("entity_id").with_row_index("s1")
        s23 = pl.concat(
            [pl.scan_parquet(f"{records.PARQUET_DIR}train_source{i}.parquet").select("entity_id") for i in (2, 3)]
        ).with_row_index("s23")
        (
            pl.scan_parquet(f"{records.PARQUET_DIR}train_ground_truth.parquet")
            .with_columns(mid=pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("mid", empty_as_null=True)
            .filter(pl.col("mid") != "")
            .join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
            .join(s23.rename({"entity_id": "mid"}), on="mid")
            .select("s1", "s23")
            .collect()
            .write_parquet(path)
        )
    return pl.read_parquet(path)


def raw_scan(split: str, side: str) -> pl.LazyFrame:
    """Return raw name and address with the same ``idx`` as :func:`records.load_split`."""
    paths = [f"{records.PARQUET_DIR}{split}_source1.parquet"] if side == "s1" else [
        f"{records.PARQUET_DIR}{split}_source{i}.parquet" for i in (2, 3)
    ]
    return pl.concat([pl.scan_parquet(p).select("business_name", "business_address") for p in paths]).with_row_index("idx")


def side_table(split: str, side: str, ids: pl.DataFrame) -> pl.DataFrame:
    """Return records and folded raw text for the given ``idx`` values of one side.

    Rows are filtered before folding, and the streaming engine keeps the full-file scans
    from being materialised.
    """
    lazy_ids = ids.lazy()
    table = pl.scan_parquet(records.records_path(split, side)).select(RECORD_COLUMNS).join(lazy_ids, on="idx", how="semi")
    raw = raw_scan(split, side).join(lazy_ids, on="idx", how="semi")
    return (
        table.join(raw, on="idx")
        .with_columns(raw_name=normalize.folded_text("business_name"), raw_address=normalize.folded_text("business_address"))
        .drop("business_name", "business_address")
        .collect(engine="streaming")
    )


def block_pairs(split: str, low: int, high: int) -> pl.DataFrame:
    """Return the pairs with ``low <= s1 < high`` and one boolean flag per key family."""
    frames = [
        pl.scan_parquet(candidates.pairs_path(split, family))
        .filter(pl.col("s1").is_between(low, high, closed="left"))
        .with_columns(pl.lit(1 << bit, pl.UInt16).alias("mask"))
        for bit, family in enumerate(candidates.FAMILIES)
    ]
    pairs = pl.concat(frames).group_by("s1", "s23").agg(pl.col("mask").sum()).collect()
    return pairs.with_columns(
        [((pl.col("mask") // (1 << bit)) % 2 == 1).alias(f"f_{family}") for bit, family in enumerate(candidates.FAMILIES)]
    ).with_columns(n_families=pl.sum_horizontal([pl.col(f"f_{family}") for family in candidates.FAMILIES]).cast(pl.UInt8)).drop("mask")


def scores(left: pl.Series, right: pl.Series, scorer) -> pl.Series:
    """Score aligned string columns pairwise with rapidfuzz; nulls compare as empty strings."""
    return pl.Series(cpdist(left.fill_null("").to_list(), right.fill_null("").to_list(), scorer=scorer, workers=WORKERS), dtype=pl.Float32)


def set_overlap(left: str, right: str) -> pl.Expr:
    """Return the number of shared elements of two list columns."""
    return pl.col(left).list.set_intersection(pl.col(right)).list.len().fill_null(0).cast(pl.UInt8)


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame, n_s2: int) -> pl.DataFrame:
    """Join both record sides onto ``pairs`` and compute name, address and per-S1 context features."""
    frame = pairs.join(s1.rename({c: f"{c}_l" for c in s1.columns if c != "idx"}), left_on="s1", right_on="idx").join(
        s23.drop("country").rename({c: f"{c}_r" for c in s23.columns if c not in ("idx", "country")}), left_on="s23", right_on="idx"
    ).rename({"country_l": "country"})
    frame = frame.with_columns(
        name_ratio=scores(frame["compact_l"], frame["compact_r"], fuzz.ratio),
        name_partial=scores(frame["compact_l"], frame["compact_r"], fuzz.partial_ratio),
        name_jw=scores(frame["compact_l"], frame["compact_r"], JaroWinkler.normalized_similarity),
        core_tset=scores(frame["core_l"], frame["core_r"], fuzz.token_set_ratio),
        raw_name_ratio=scores(frame["raw_name_l"], frame["raw_name_r"], fuzz.ratio),
        addr_ratio=scores(frame["raw_address_l"], frame["raw_address_r"], fuzz.ratio),
        addr_tset=scores(frame["raw_address_l"], frame["raw_address_r"], fuzz.token_set_ratio),
        house_edit=scores(frame["house_l"], frame["house_r"], Levenshtein.distance),
    )
    tokens_l, tokens_r = pl.col("core_l").str.split(" "), pl.col("core_r").str.split(" ")
    house_l, house_r = pl.col("house_l").cast(pl.Int64, strict=False), pl.col("house_r").cast(pl.Int64, strict=False)
    missing_l, missing_r = pl.col("house_l").is_null(), pl.col("house_r").is_null()
    frame = frame.with_columns(
        is_s3=pl.col("s23") >= n_s2,
        core_eq=(pl.col("core_l") == pl.col("core_r")).fill_null(False),
        compact_eq=(pl.col("compact_l") == pl.col("compact_r")).fill_null(False),
        raw_name_eq=pl.col("raw_name_l") == pl.col("raw_name_r"),
        len_l=pl.col("compact_l").str.len_chars().fill_null(0).cast(pl.UInt16),
        len_r=pl.col("compact_r").str.len_chars().fill_null(0).cast(pl.UInt16),
        tokens_l=tokens_l.list.len().fill_null(0).cast(pl.UInt8),
        tokens_r=tokens_r.list.len().fill_null(0).cast(pl.UInt8),
        token_overlap=tokens_l.list.set_intersection(tokens_r).list.len().fill_null(0).cast(pl.UInt8),
        house_missing_l=missing_l,
        house_missing_r=missing_r,
        house_eq=(pl.col("house_l") == pl.col("house_r")).fill_null(False),
        house_in=pl.col("numbers_r").list.contains(pl.col("house_l")).fill_null(False),
        house_in_reverse=pl.col("numbers_l").list.contains(pl.col("house_r")).fill_null(False),
        house_edit=pl.when(missing_l | missing_r).then(None).otherwise(pl.col("house_edit")),
        house_diff=(house_l - house_r).abs().cast(pl.Float32),
        numbers_l=pl.col("numbers_l").list.len().fill_null(0).cast(pl.UInt8),
        numbers_r=pl.col("numbers_r").list.len().fill_null(0).cast(pl.UInt8),
        number_overlap=set_overlap("numbers_l", "numbers_r"),
        words_l=pl.col("words_l").list.len().fill_null(0).cast(pl.UInt8),
        words_r=pl.col("words_r").list.len().fill_null(0).cast(pl.UInt8),
        word_overlap=set_overlap("words_l", "words_r"),
        street_eq=(pl.col("street_l") == pl.col("street_r")).fill_null(False),
        address_empty_r=pl.col("raw_address_r").str.strip_chars().is_in(["", "null"]),
    ).with_columns(
        strong=(pl.col("name_ratio") >= STRONG_RATIO) & pl.col("house_in"),
    )
    per_s1 = pl.col("s1")
    return frame.with_columns(
        s1_candidates=pl.len().over(per_s1).cast(pl.UInt16),
        s1_strong=pl.col("strong").sum().over(per_s1).cast(pl.UInt16),
        # Hard-name and same-core counts per S1: the US singleton rate varies with the hard-decoy count.
        s1_hard=(pl.col("name_ratio") >= STRONG_RATIO).sum().over(per_s1).cast(pl.UInt16),
        s1_core_eq=pl.col("core_eq").sum().over(per_s1).cast(pl.UInt16),
        name_rank=pl.col("name_ratio").rank("min", descending=True).over(per_s1).cast(pl.UInt16),
        name_gap=(pl.col("name_ratio").max().over(per_s1) - pl.col("name_ratio")),
        raw_name_rank=pl.col("raw_name_ratio").rank("min", descending=True).over(per_s1).cast(pl.UInt16),
        addr_rank=pl.col("addr_tset").rank("min", descending=True).over(per_s1).cast(pl.UInt16),
        addr_gap=(pl.col("addr_tset").max().over(per_s1) - pl.col("addr_tset")),
    ).drop(
        [f"{c}_{s}" for c in ("core", "compact", "words", "house", "street", "raw_name", "raw_address") for s in ("l", "r")]
    )


def write_pairs(split: str, chunk_s1: int, limit: int | None, out_dir: str, shard: str = "0/1") -> None:
    """Write one feature part per block of ``chunk_s1`` S1 rows; ``shard`` ``k/n`` writes only the parts whose number is k modulo n."""
    n_s1 = pl.scan_parquet(records.records_path(split, "s1")).select(pl.len()).collect().item()
    n_s2 = pl.scan_parquet(f"{records.PARQUET_DIR}{split}_source2.parquet").select(pl.len()).collect().item()
    truth = truth_links() if split == "train" else None
    Path(part_path(split, 0, out_dir)).parent.mkdir(parents=True, exist_ok=True)
    blocks = range(0, n_s1, chunk_s1)
    started, total = time.time(), 0
    k, n = (int(x) for x in shard.split("/"))
    for part, low in enumerate(blocks[:limit] if limit else blocks):
        if part % n != k:
            continue
        pairs = block_pairs(split, low, low + chunk_s1)
        s1 = side_table(split, "s1", pairs.select(idx="s1").unique())
        s23 = side_table(split, "s23", pairs.select(idx="s23").unique())
        frame = pair_features(pairs, s1, s23, n_s2)
        if truth is not None:
            links = truth.filter(pl.col("s1").is_between(low, low + chunk_s1, closed="left")).with_columns(label=pl.lit(True))
            frame = frame.join(links, on=["s1", "s23"], how="left").with_columns(pl.col("label").fill_null(False))
        frame.write_parquet(part_path(split, part, out_dir))
        total += len(frame)
        print(f"part {part}: s1 {low}-{low + chunk_s1} pairs={len(frame)} total={total} t={time.time() - started:.0f}s", flush=True)
        del pairs, s1, s23, frame


def write_context(split: str, out_dir: str) -> None:
    """Aggregate every part per S2/S3 record and write the context table."""
    parts = pl.scan_parquet(f"{out_dir}{split}/part_*.parquet").select("s1", "s23", "name_ratio", "strong")
    context = parts.group_by("s23").agg(
        s23_candidates=pl.len().cast(pl.UInt16),
        s23_strong=pl.col("strong").sum().cast(pl.UInt16),
        top_s1=pl.col("s1").sort_by("name_ratio", descending=True).first(),
        top_name=pl.col("name_ratio").max(),
        second_name=pl.col("name_ratio").top_k(2).get(1, null_on_oob=True),
    )
    context.collect(engine="streaming").write_parquet(context_path(split, out_dir))


def scan(split: str, parts: list[int] | None = None, out_dir: str = FEATURE_DIR) -> pl.LazyFrame:
    """Return feature parts lazily, joined with competitor features from the S2/S3 context table.

    ``other_best_name`` is the best name ratio of any other S1 candidate of the same S2/S3
    record (null when there is none), and ``other_strong`` counts other S1 candidates with
    a strong name and house-number match.
    """
    paths = [part_path(split, p, out_dir) for p in parts] if parts is not None else f"{out_dir}{split}/part_*.parquet"
    context = pl.scan_parquet(context_path(split, out_dir))
    return (
        pl.scan_parquet(paths)
        .join(context, on="s23", how="left")
        .with_columns(
            other_best_name=pl.when(pl.col("s1") == pl.col("top_s1")).then(pl.col("second_name")).otherwise(pl.col("top_name")),
            other_strong=(pl.col("s23_strong") - pl.col("strong").cast(pl.UInt16)),
        )
        .with_columns(other_name_gap=pl.col("name_ratio") - pl.col("other_best_name"))
        .drop("top_s1", "top_name", "second_name", "s23_strong")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["pairs", "context"])
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--chunk-s1", type=int, default=100_000)
    parser.add_argument("--limit", type=int, help="process only the first N S1 blocks")
    parser.add_argument("--shard", default="0/1", help="k/n: write only the parts whose number is k modulo n, for n parallel processes")
    parser.add_argument("--out-dir", default=FEATURE_DIR)
    arguments = parser.parse_args()
    if arguments.step == "pairs":
        write_pairs(arguments.split, arguments.chunk_s1, arguments.limit, arguments.out_dir, arguments.shard)
    else:
        write_context(arguments.split, arguments.out_dir)
