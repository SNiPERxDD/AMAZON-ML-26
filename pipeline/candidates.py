"""Candidate generation: pair each S1 entity with plausible S2/S3 records through several blocking keys.

Keys are built from the per-record cache (``pipeline.records``) as 64-bit hashes of
their parts; country is always one of the parts. A block contributes pairs only
when it holds at most ``cap_s1`` S1 records and ``cap_s23`` S2/S3 records.

S1 uses its first house number; S2/S3 uses every leading number, because S2/S3
addresses often gain an inserted number in front of the real one.

Each step runs in its own process so that memory is returned to the system between
key families (polars keeps freed memory otherwise). From the repository root:

    python -m pipeline.candidates run train      # parts, every family, then evaluation
    python -m pipeline.candidates run test       # parts, every family, then the union of pairs

Per-family pairs go to ``data/candidates/{split}_pairs/{family}.parquet``.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from pipeline import records
from pipeline.metric import macro_f05

SEED = 17
MIN_KEY_TOKEN_CHARS = 3
MIN_COMPACT_CHARS = 6
COMPACT_PREFIX_CHARS = 8
RARE_TOKENS_PER_RECORD = 3
REACH_SAMPLE_LINKS = 300_000
FAMILIES = [
    "core", "core_house", "token_number", "number_street", "token_street", "prefix_number",
    "compact", "core_word", "rare_token_word", "prefix_word", "number_word",
]
# Families whose exploded key tables are large are blocked in this many passes.
FAMILY_PARTITIONS = {"token_number": 2, "token_street": 2, "core_word": 2, "rare_token_word": 6, "prefix_word": 3, "number_word": 4}
# Family -> (table holding the parts, parts). "base" has one row per record.
FAMILY_PARTS = {
    "core": ("base", ["core"]),
    "core_house": ("base", ["core", "house"]),
    "compact": ("base", ["compact"]),
    "token_number": ("tokens*numbers", ["token", "number"]),
    "number_street": ("numbers", ["number", "street"]),
    "token_street": ("tokens", ["token", "street"]),
    "prefix_number": ("numbers", ["prefix", "number"]),
    "core_word": ("words", ["core", "word"]),
    "rare_token_word": ("rare*words", ["token", "word"]),
    "prefix_word": ("words", ["prefix", "word"]),
    "number_word": ("numbers*words", ["number", "word"]),
}
# Address-only keys ignore the name, so their blocks are held to tighter caps (S1, S2/S3).
FAMILY_CAPS = {"number_word": (3, 8)}


def hashed(expr: pl.Expr) -> pl.Expr:
    """Hash one string expression to ``u64``; null and empty strings stay null."""
    return pl.when(expr.is_not_null() & (expr != "")).then(expr.hash(seed=SEED))


def token_rarity(split: str) -> pl.DataFrame:
    """Return ``country``, ``token``, ``count`` over both sides, for picking each record's rarest tokens."""
    return (
        pl.scan_parquet([records.tokens_path(split, side) for side in ("s1", "s23")])
        .group_by("country", "token").len("count")
        .collect()
    )


def side_parts(split: str, side: str, rarity: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return hashed key parts of one side: per record, numbers, address words, tokens, and rare tokens."""
    frame = pl.read_parquet(records.records_path(split, side))
    long_compact = pl.when(pl.col("compact").str.len_chars() >= MIN_COMPACT_CHARS)
    base = frame.select(
        "idx",
        c=pl.col("country").hash(seed=SEED),
        core=hashed(pl.col("core")),
        compact=hashed(long_compact.then(pl.col("compact"))),
        prefix=hashed(long_compact.then(pl.col("compact").str.slice(0, COMPACT_PREFIX_CHARS))),
        house=hashed(pl.col("house")),
        street=hashed(pl.col("street")),
    )
    numbers = frame.select("idx", number=pl.col("house")) if side == "s1" else \
        frame.select("idx", number=pl.col("numbers")).explode("number", empty_as_null=True)
    words = frame.select("idx", word=pl.col("words")).explode("word", empty_as_null=True)
    del frame
    tokens = (
        pl.read_parquet(records.tokens_path(split, side))
        .filter(pl.col("token").str.len_chars() >= MIN_KEY_TOKEN_CHARS)
        .unique(["idx", "token"])
        .join(rarity, on=["country", "token"], how="left")
    )
    rare = tokens.sort("count").group_by("idx").head(RARE_TOKENS_PER_RECORD)
    return {
        "base": base,
        "numbers": numbers.select("idx", number=hashed(pl.col("number"))).drop_nulls(),
        "words": words.select("idx", word=hashed(pl.col("word"))).drop_nulls(),
        "tokens": tokens.select("idx", token=hashed(pl.col("token"))).drop_nulls(),
        "rare": rare.select("idx", token=hashed(pl.col("token"))).drop_nulls(),
    }


def family_keys(family: str, parts: dict[str, pl.DataFrame], partition: int = 0, partitions: int = 1) -> pl.DataFrame:
    """Return unique (``idx``, ``key``) rows of one key family for one side, skipping rows with a null part.

    With ``partitions > 1`` only rows whose first key part hashes to ``partition`` are
    kept; every record of a block shares that part, so partitions never split a block.
    """
    source, columns = FAMILY_PARTS[family]
    base = parts["base"]
    tables = source.split("*")
    first = columns[0]
    holder = "base" if first in base.columns else next(t for t in tables if first in parts[t].columns)
    selected = {name: parts[name] for name in {*tables, "base"}}
    if partitions > 1:
        selected[holder] = selected[holder].filter(pl.col(first) % partitions == partition)
    frame = selected[tables[0]]
    for extra in tables[1:]:
        frame = frame.join(selected[extra], on="idx")
    if tables[0] != "base":
        base_columns = [c for c in columns if c in base.columns and c not in frame.columns]
        frame = frame.join(selected["base"].select("idx", "c", *base_columns), on="idx")
    columns = ["c", *columns]
    return frame.drop_nulls(columns).select("idx", key=pl.struct(columns).hash(seed=SEED)).unique()


def block_pairs(keys_s1: pl.DataFrame, keys_s23: pl.DataFrame, cap_s1: int, cap_s23: int) -> pl.DataFrame:
    """Join two key tables on blocks within the size caps and return unique (``s1``, ``s23``) pairs."""
    sizes_s1 = keys_s1.group_by("key").len().filter(pl.col("len") <= cap_s1).select("key")
    sizes_s23 = keys_s23.group_by("key").len().filter(pl.col("len") <= cap_s23).select("key")
    allowed = sizes_s1.join(sizes_s23, on="key")
    left = keys_s1.join(allowed, on="key", how="semi").rename({"idx": "s1"})
    right = keys_s23.join(allowed, on="key", how="semi").rename({"idx": "s23"})
    return left.join(right, on="key").select("s1", "s23").unique()


def parts_path(split: str, side: str, name: str) -> str:
    """Return the cache path of one hashed parts table."""
    return f"{records.CACHE_DIR}{split}_parts/{side}_{name}.parquet"


def pairs_path(split: str, family: str) -> str:
    """Return the path of one family's pair table."""
    return f"{records.CACHE_DIR}{split}_pairs/{family}.parquet"


def write_parts(split: str) -> None:
    """Hash and cache the key parts of both sides of ``split``."""
    Path(parts_path(split, "s1", "x")).parent.mkdir(parents=True, exist_ok=True)
    rarity = token_rarity(split)
    for side in ("s1", "s23"):
        for name, table in side_parts(split, side, rarity).items():
            table.write_parquet(parts_path(split, side, name))


def write_family(split: str, family: str, cap_s1: int, cap_s23: int) -> None:
    """Block one family and write its pairs; on train also write the uncapped reachable links of a link sample."""
    names = {"base", *FAMILY_PARTS[family][0].split("*")}
    parts = {side: {name: pl.read_parquet(parts_path(split, side, name)) for name in names} for side in ("s1", "s23")}
    partitions = FAMILY_PARTITIONS.get(family, 1)
    cap_s1, cap_s23 = FAMILY_CAPS.get(family, (cap_s1, cap_s23))
    sample = records.load_truth().sample(REACH_SAMPLE_LINKS, seed=0) if split == "train" else None
    pairs, reachable = [], []
    for partition in range(partitions):
        left = family_keys(family, parts["s1"], partition, partitions)
        right = family_keys(family, parts["s23"], partition, partitions)
        if sample is not None:
            reachable.append(
                sample.join(left.rename({"idx": "s1"}), on="s1")
                .join(right.rename({"idx": "s23"}), on=["s23", "key"])
                .select("s1", "s23")
            )
        pairs.append(block_pairs(left, right, cap_s1, cap_s23))
        del left, right
    Path(pairs_path(split, family)).parent.mkdir(parents=True, exist_ok=True)
    pl.concat(pairs).unique().write_parquet(pairs_path(split, family))
    if sample is not None:
        pl.concat(reachable).unique().write_parquet(pairs_path(split, f"reach_{family}"))


def pair_code(frame: pl.DataFrame) -> pl.Series:
    """Encode (``s1``, ``s23``) row-index pairs as one ``u64`` per pair."""
    return frame.select(pl.col("s1").cast(pl.UInt64) * 2**32 + pl.col("s23").cast(pl.UInt64)).to_series()


def decode(codes: pl.Series) -> pl.DataFrame:
    """Invert :func:`pair_code`."""
    return pl.DataFrame({"code": codes}).select(
        s1=(pl.col("code") // 2**32).cast(pl.UInt32), s23=(pl.col("code") % 2**32).cast(pl.UInt32)
    )


def write_union(split: str, families: list[str]) -> None:
    """Write the union of the family pair files as ``{split}_pairs.parquet``, sorted by (``s1``, ``s23``)."""
    union = pl.concat([pair_code(pl.read_parquet(pairs_path(split, family))) for family in families]).unique()
    decode(union).sort("s1", "s23").write_parquet(f"{records.CACHE_DIR}{split}_pairs.parquet")
    print(f"{split} union: {len(union)} pairs")


def evaluate_train(families: list[str], save_path: str | None) -> None:
    """Print capped recall, uncapped reachability (on a link sample) and pair counts per family, then the oracle."""
    truth = records.load_truth()
    truth_codes = pair_code(truth).sort()
    country = pl.read_parquet(records.records_path("train", "s1"), columns=["idx", "country"]).rename({"idx": "s1"})
    n_true, n_s1, n_sample = len(truth), len(country), REACH_SAMPLE_LINKS
    union, union_reach = pl.Series("code", [], pl.UInt64), pl.Series("code", [], pl.UInt64)
    for family in families:
        codes = pair_code(pl.read_parquet(pairs_path("train", family)))
        reach = pair_code(pl.read_parquet(pairs_path("train", f"reach_{family}")))
        union = pl.concat([union, codes]).unique()
        union_reach = pl.concat([union_reach, reach]).unique()
        hit = codes.is_in(truth_codes).sum()
        cum = union.is_in(truth_codes).sum()
        print(f"{family:15s} pairs={len(codes) / 1e6:6.2f}M recall={hit / n_true:.4f} reach={len(reach) / n_sample:.4f} "
              f"| union {len(union) / 1e6:6.2f}M per_s1={len(union) / n_s1:5.1f} recall={cum / n_true:.4f} "
              f"reach={len(union_reach) / n_sample:.4f}", flush=True)
        del codes, reach
    pairs = decode(union)
    del union
    per_s1 = pairs.group_by("s1").len()["len"]
    print(f"pairs per S1: p50={per_s1.quantile(0.5):.0f} p90={per_s1.quantile(0.9):.0f} p99={per_s1.quantile(0.99):.0f} "
          f"max={per_s1.max()} s1_without_pairs={1 - len(per_s1) / n_s1:.4f}")
    oracle = pairs.join(truth, on=["s1", "s23"], how="semi")
    for name in ("US", "India"):
        ids = country.filter(pl.col("country") == name).select("s1")
        truth_part = truth.join(ids, on="s1", how="semi")
        oracle_part = oracle.join(ids, on="s1", how="semi")
        print(f"{name}: recall={len(oracle_part) / len(truth_part):.4f} oracle macro F0.5={macro_f05(oracle_part, truth_part, ids):.4f}")
    print(f"all: oracle macro F0.5={macro_f05(oracle, truth, country.select('s1')):.4f}")
    if save_path:
        pairs.sort("s1", "s23").write_parquet(save_path)
        truth.join(oracle, on=["s1", "s23"], how="anti").write_parquet(save_path.replace(".parquet", "_missed.parquet"))
        print(f"saved {save_path}")


def run(split: str, arguments: argparse.Namespace) -> None:
    """Run parts, each family, and (on train) evaluation, each in a fresh subprocess."""
    started = time.time()
    common = ["--cap-s1", str(arguments.cap_s1), "--cap-s23", str(arguments.cap_s23)]
    steps = [["parts", split]] + [["family", split, family, *common] for family in arguments.families.split(",")]
    if split == "train":
        steps.append(["evaluate", split, "--families", arguments.families] + (["--save", arguments.save] if arguments.save else []))
    else:
        steps.append(["union", split, "--families", arguments.families])
    for step in steps:
        subprocess.run([sys.executable, "-m", "pipeline.candidates", *step], check=True)
        print(f"[{' '.join(step[:3])}] done t={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["run", "parts", "family", "evaluate", "union"])
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("family", nargs="?")
    parser.add_argument("--cap-s1", type=int, default=10)
    parser.add_argument("--cap-s23", type=int, default=30)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--save", help="write the train union pairs and missed links to this parquet path")
    arguments = parser.parse_args()
    if arguments.step == "run":
        run(arguments.split, arguments)
    elif arguments.step == "parts":
        write_parts(arguments.split)
    elif arguments.step == "family":
        write_family(arguments.split, arguments.family, arguments.cap_s1, arguments.cap_s23)
    elif arguments.step == "union":
        write_union(arguments.split, arguments.families.split(","))
    else:
        evaluate_train(arguments.families.split(","), arguments.save)
