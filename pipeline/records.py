"""Load the raw splits and cache per-record features as parquet.

The cache holds, per split and side (``s1`` or ``s23``), one row per record
(``idx``, ``country``, ``core``, ``compact``, ``house``, ``numbers``, ``street``,
``words``) and one row per name token (``idx``, ``country``, ``token``). The
Indic-to-Latin table is learned once from train links and applied to both splits.

Build with ``python -m pipeline.records train`` (or ``test``) from the repository root.
"""

import sys
from pathlib import Path

import polars as pl

from pipeline import normalize

PARQUET_DIR = "data/parquet/"
CACHE_DIR = "data/candidates/"
TRANSLITERATION_PATH = f"{CACHE_DIR}transliteration.parquet"
# Address words in at least this share of a country's records (S1 and S2/S3 together) are
# states, regions, and large cities (``texas``, ``maharashtra``, ``aquitaine``). They never
# serve as the street, because blocks keyed on them exceed the caps.
FREQUENT_WORD_SHARE = 0.002


def load_split(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return S1 and concatenated S2/S3 with a ``u32`` row index ``idx``."""
    s1 = pl.read_parquet(f"{PARQUET_DIR}{split}_source1.parquet").with_row_index("idx")
    s23 = pl.concat(
        [pl.read_parquet(f"{PARQUET_DIR}{split}_source{i}.parquet") for i in (2, 3)]
    ).with_row_index("idx")
    return s1, s23


def load_truth(s1: pl.DataFrame | None = None, s23: pl.DataFrame | None = None) -> pl.DataFrame:
    """Return train links as row indices (``s1``, ``s23``)."""
    if s1 is None or s23 is None:
        s1, s23 = load_split("train")
    ground_truth = pl.read_parquet(f"{PARQUET_DIR}train_ground_truth.parquet")
    return (
        ground_truth.with_columns(mid=pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("mid", empty_as_null=True)
        .filter(pl.col("mid") != "")
        .join(s1.select(s1="idx", source1_entity_id="entity_id"), on="source1_entity_id")
        .join(s23.select(s23="idx", mid="entity_id"), on="mid")
        .select("s1", "s23")
    )


def records_path(split: str, side: str) -> str:
    """Return the cache path of the per-record table."""
    return f"{CACHE_DIR}{split}_{side}_records.parquet"


def tokens_path(split: str, side: str) -> str:
    """Return the cache path of the per-token table."""
    return f"{CACHE_DIR}{split}_{side}_tokens.parquet"


def transliteration_table() -> pl.DataFrame:
    """Return the cached Indic-to-Latin token and phrase table, learning it from train links on first use."""
    if not Path(TRANSLITERATION_PATH).exists():
        s1, s23 = load_split("train")
        links = load_truth(s1, s23)
        table = normalize.learn_transliteration(s1, s23, links)
        pl.concat([table, normalize.learn_phrases(s1, s23, links, table)]).write_parquet(TRANSLITERATION_PATH)
    return pl.read_parquet(TRANSLITERATION_PATH)


def load_side(split: str, side: str) -> pl.DataFrame:
    """Return one side of ``split`` with the same ``idx`` numbering as :func:`load_split`."""
    if side == "s1":
        return pl.read_parquet(f"{PARQUET_DIR}{split}_source1.parquet").with_row_index("idx")
    return pl.concat([pl.read_parquet(f"{PARQUET_DIR}{split}_source{i}.parquet") for i in (2, 3)]).with_row_index("idx")


def build(split: str, chunk_rows: int = 2_000_000) -> None:
    """Write the per-record and per-token caches of ``split``, in row chunks to bound memory."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    transliteration = transliteration_table()
    for side in ("s1", "s23"):
        frame = load_side(split, side)
        parts = []
        for number, start in enumerate(range(0, len(frame), chunk_rows)):
            chunk = frame.slice(start, chunk_rows)
            tokens = normalize.name_tokens(chunk, transliteration)
            record_part, token_part = f"{CACHE_DIR}part_r{number}.parquet", f"{CACHE_DIR}part_t{number}.parquet"
            (
                chunk.select("idx", "country")
                .join(normalize.name_features(tokens), on="idx", how="left")
                .join(normalize.address_features(chunk), on="idx", how="left")
                .sort("idx")
                .write_parquet(record_part)
            )
            tokens.join(chunk.select("idx", "country"), on="idx").select("idx", "country", "token").write_parquet(token_part)
            parts.append((record_part, token_part))
            del chunk, tokens
        del frame
        pl.scan_parquet([r for r, _ in parts]).sink_parquet(records_path(split, side))
        pl.scan_parquet([t for _, t in parts]).sink_parquet(tokens_path(split, side))
        for record_part, token_part in parts:
            Path(record_part).unlink()
            Path(token_part).unlink()
        print(f"cached {split} {side}", flush=True)
    refine_street(split)


def frequent_words_path(split: str) -> str:
    """Return the cache path of the frequent address words of ``split``."""
    return f"{CACHE_DIR}{split}_frequent_words.parquet"


def refine_street(split: str) -> None:
    """Recompute ``street`` as the longest address word that is not frequent in its country.

    Frequency is the share of records, both sides together, whose ``words`` hold the word
    (``words`` is unique per record). Only the first ``max_words`` words are considered.
    The frequent words are cached for inspection.
    """
    scans = [pl.scan_parquet(records_path(split, side)).select("country", "words") for side in ("s1", "s23")]
    both = pl.concat(scans)
    totals = both.group_by("country").agg(records=pl.len())
    frequent = (
        both.explode("words", empty_as_null=True).drop_nulls("words").group_by("country", "words").agg(count=pl.len())
    ).join(totals, on="country").filter(pl.col("count") >= FREQUENT_WORD_SHARE * pl.col("records")).collect()
    frequent.select("country", word="words", share=pl.col("count") / pl.col("records")).sort("country", "share").write_parquet(
        frequent_words_path(split)
    )
    for side in ("s1", "s23"):
        frame = pl.read_parquet(records_path(split, side))
        street = (
            frame.select("idx", "country", "words").explode("words", empty_as_null=True).drop_nulls("words")
            .join(frequent.select("country", "words"), on=["country", "words"], how="anti")
            .sort(pl.col("words").str.len_chars(), descending=True, maintain_order=True)
            .group_by("idx", maintain_order=True).agg(street=pl.col("words").first())
        )
        frame.drop("street").join(street, on="idx", how="left").select(frame.columns).sort("idx").write_parquet(records_path(split, side))
        del frame, street
    print(f"refined street for {split}: {len(frequent)} frequent words", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--street-only":
        refine_street(sys.argv[1])
    else:
        build(sys.argv[1])
