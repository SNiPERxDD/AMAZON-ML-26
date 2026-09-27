"""Address-only retrieval: match every S2/S3 record with an address to the S1 entities whose address it shares, ignoring the name.

The missed true links of the hold-out that carry an address are mostly never candidates, and reading them shows why:
the S2/S3 name is a made-up brand ("JAXFLUXSYN", "Zephiri", "Pyrakor"), a domain made from the S1 name
("vcpacific.com", "pcreations.com"), or "X doing business as Y", while the address is the owner's with the usual
noise (reordered components, abbreviations, a dropped or mistyped house number). Retrieval that leans on the name
cannot find them. This blocks on address token bigrams (ordinal suffixes, street types and state names normalised)
that at most ``MAX_KEY`` S1 of the country share, keeps the ``TOP`` S1 per record by shared bigrams, and scores
address and name with rapidfuzz. Per record it keeps the candidates with the record's best and second address
scores and the number of S1 within 5 points of the best, which measure how ambiguous the address is (shared buildings
have several S1). Output: ``data/addronly/{split}_pairs.parquet``.

Run from the repository root: ``PYTHONPATH=. python stages/addr_only_match.py --split train`` (or ``test``).
"""

import argparse
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from pipeline import records

OUT = Path("data/addronly")
MAX_KEY = 30
TOP = 5
CHUNK = 1_000_000
WORDS = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway", "hwy": "highway", "cir": "circle", "ter": "terrace",
    "trl": "trail", "sq": "square", "mt": "mount", "ft": "fort", "n": "north", "s": "south", "e": "east", "w": "west",
    "apt": "unit", "ste": "unit", "suite": "unit", "bldg": "building", "fl": "floor", "no": "", "nr": "near", "opp": "opposite",
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    "maharashtra": "mh", "karnataka": "ka", "telangana": "tg", "delhi": "dl", "gujarat": "gj", "kerala": "kl",
    "rajasthan": "rj", "haryana": "hr", "punjab": "pb", "bihar": "br", "odisha": "od", "orissa": "od", "goa": "ga",
}


def no_address(column: str) -> pl.Expr:
    value = pl.col(column).str.strip_chars().str.to_lowercase()
    return pl.col(column).is_null() | value.is_in(["", "none", "null", "nan", "<null>", "n/a"])


def tokens(frame: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Return (id, country, pos, tok) rows of the normalised address."""
    text = (pl.col("business_address").str.to_lowercase().str.replace_all(r"(\d+)(st|nd|rd|th)\b", "$1")
            .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" "))
    t = frame.select(id_col, "country", tok=text).explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    t = t.with_columns(pl.col("tok").replace(WORDS)).filter(pl.col("tok") != "")
    return t.with_columns(pos=pl.int_range(pl.len()).over(id_col))


def bigrams(tok: pl.DataFrame, id_col: str) -> pl.DataFrame:
    nxt = tok.select(id_col, pos=pl.col("pos") - 1, tok2="tok")
    return tok.join(nxt, on=[id_col, "pos"]).select(id_col, "country", key=pl.col("tok") + " " + pl.col("tok2")).unique()


def norm_text(expr: pl.Expr) -> pl.Expr:
    return expr.fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-key", type=int, default=MAX_KEY)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    s1, s23 = records.load_split(args.split)
    s1 = s1.filter(~no_address("business_address")).select("country", "business_address", "business_name", s1="idx")
    s23 = s23.filter(~no_address("business_address")).select("country", "business_address", "business_name", s23="idx")
    k1 = bigrams(tokens(s1, "s1"), "s1")
    k1 = k1.join(k1.group_by("country", "key").len().filter(pl.col("len") <= args.max_key).select("country", "key"), on=["country", "key"], how="semi")
    print(f"{args.split}: S1 {s1.height}, S2/S3 with address {s23.height}, S1 keys {k1.height}", flush=True)
    addr1 = s1.select("s1", a1=norm_text(pl.col("business_address")), n1=norm_text(pl.col("business_name")))
    parts = []
    for start in range(0, s23.height, args.chunk):
        chunk = s23.slice(start, args.chunk)
        kq = bigrams(tokens(chunk, "s23"), "s23")
        pairs = (kq.join(k1, on=["country", "key"]).group_by("s23", "s1").agg(shared=pl.len()).filter(pl.col("shared") >= 2)
                 .sort("shared", descending=True).group_by("s23", maintain_order=True).head(TOP))
        pairs = (pairs.join(chunk.select("s23", a2=norm_text(pl.col("business_address")), n2=norm_text(pl.col("business_name"))), on="s23")
                 .join(addr1, on="s1"))
        pairs = pairs.with_columns(
            addr=pl.Series(cpdist(pairs["a1"].to_list(), pairs["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=4), dtype=pl.Float32),
            addr_sort=pl.Series(cpdist(pairs["a1"].to_list(), pairs["a2"].to_list(), scorer=fuzz.token_sort_ratio, workers=4), dtype=pl.Float32),
            name=pl.Series(cpdist(pairs["n1"].to_list(), pairs["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=4), dtype=pl.Float32),
        ).drop("a1", "a2", "n1", "n2")
        pairs = pairs.with_columns(
            n_close=(pl.col("addr") >= pl.col("addr").max().over("s23") - 5).sum().over("s23").cast(pl.UInt8),
            rank=pl.col("addr").rank("ordinal", descending=True).over("s23").cast(pl.UInt8),
            n_cand=pl.len().over("s23").cast(pl.UInt8),
        )
        top = pairs.filter(pl.col("rank") == 1).select("s23", top_addr="addr")
        second = pairs.filter(pl.col("rank") == 2).select("s23", second_addr="addr")
        pairs = pairs.join(top, on="s23").join(second, on="s23", how="left").filter(pl.col("addr") >= 60)
        parts.append(pairs)
        print(f"  chunk {start // args.chunk}: {pairs.height} pairs", flush=True)
    out = pl.concat(parts)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / f"{args.split}_pairs{args.tag}.parquet")
    print(f"wrote {out.height} pairs for {out['s23'].n_unique()} records", flush=True)


if __name__ == "__main__":
    main()
