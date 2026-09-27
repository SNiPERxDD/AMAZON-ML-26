"""Record normalisation: name tokens, compact names, and address numbers and street tokens.

Every record is reduced to a row index (``idx``) plus derived columns, so later
stages join on integers instead of entity-id strings.
"""

from collections import Counter, defaultdict

import polars as pl

INDIC = r"[ऀ-෿]"
TOKEN = r"[\p{L}\p{M}\p{N}]+"
# Token delimiters used while replacing multi-token phrases in a joined name.
OPEN, CLOSE = "\x02", "\x03"
# Legal forms, honorifics, web fragments, and link words that carry no identity.
STOP_TOKENS = [
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
    "llp", "plc", "pc", "the", "and", "of", "dr", "mr", "mrs", "ms", "smt", "shri", "group", "services",
    "holdings", "sarl", "sas", "sasu", "sci", "sa", "ets", "cie", "com", "net", "org", "www", "dba", "fka",
    "aka", "formerly", "nee", "née",
]
# Digits that stand in for letters inside words (C0uncil, Roya1, 5chadt, 6reat, 8uildscape).
HOMOGLYPH_DIGITS = ["0", "1", "3", "4", "5", "6", "7", "8"]
HOMOGLYPH_LETTERS = ["o", "l", "e", "a", "s", "g", "t", "b"]
GENERIC_STREET = [
    "road", "street", "null", "none", "city", "near", "floor", "door", "drive", "avenue", "lane", "nagar",
    "unit", "suite", "plot", "shop", "office", "flat", "building", "sector", "north", "south", "east", "west",
]


def folded_text(column: str) -> pl.Expr:
    """Lowercase and strip Latin diacritics, keeping Indic combining marks."""
    return (
        pl.col(column).fill_null("").str.to_lowercase().str.normalize("NFKD").str.replace_all(r"[̀-ͯ]", "")
    )


def name_tokens(records: pl.DataFrame, transliteration: pl.DataFrame | None) -> pl.DataFrame:
    """Return one row per (``idx``, ``token``, ``position``) after folding, transliteration, and stop-token removal.

    ``transliteration`` maps Indic tokens (column ``token``) to Latin tokens (column ``latin``).
    Rows whose ``token`` holds several space-separated tokens are phrases (:func:`learn_phrases`):
    each run of those tokens in a name is replaced by the one Latin token before the lookup.
    """
    token = pl.col("token")
    mixed = token.str.contains(r"[a-z]") & token.str.contains(r"\d") & ~token.str.contains(r"^\d+(st|nd|rd|th)$")
    words = folded_text("business_name").str.extract_all(TOKEN)
    if transliteration is not None:
        phrase = pl.col("token").str.contains(" ")
        phrases = transliteration.filter(phrase).sort(pl.col("token").str.len_chars(), descending=True)
        transliteration = transliteration.filter(~phrase)
        if phrases.height:
            marked = pl.concat_str(pl.lit(OPEN), words.list.join(CLOSE + OPEN), pl.lit(CLOSE))
            words = marked.str.replace_many(
                [OPEN + run.replace(" ", CLOSE + OPEN) + CLOSE for run in phrases["token"]],
                [OPEN + latin + CLOSE for latin in phrases["latin"]],
                leftmost=True,
            ).str.extract_all(f"[^{OPEN}{CLOSE}]+")
    tokens = (
        records.select("idx", token=words)
        .explode("token")
        .drop_nulls("token")
        .with_columns(position=pl.int_range(pl.len()).over("idx").cast(pl.UInt16))
        .with_columns(pl.when(mixed).then(token.str.replace_many(HOMOGLYPH_DIGITS, HOMOGLYPH_LETTERS)).otherwise(token))
    )
    if transliteration is not None:
        tokens = tokens.join(transliteration, on="token", how="left").with_columns(
            token=pl.coalesce("latin", "token")
        ).drop("latin")
    return tokens.filter(~token.is_in(STOP_TOKENS) & (token.str.len_chars() > 1))


def name_features(tokens: pl.DataFrame) -> pl.DataFrame:
    """Aggregate tokens into ``core`` (sorted unique tokens) and ``compact`` (tokens joined in order)."""
    return tokens.sort("idx", "position").group_by("idx").agg(
        core=pl.col("token").unique().sort().str.join(" "),
        compact=pl.col("token").str.join(""),
    )


def address_features(records: pl.DataFrame, max_numbers: int = 4, max_words: int = 6) -> pl.DataFrame:
    """Return ``idx``, ``house`` (first number), ``numbers`` (first unique numbers), ``words`` (distinctive words),
    and a provisional ``street`` (the longest distinctive word), which ``records.build`` replaces once word
    frequencies of the whole split are known."""
    text = folded_text("business_address")
    numbers = (
        text.str.extract_all(r"\d+")
        .list.eval(pl.element().str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.unique(maintain_order=True)
        .list.head(max_numbers)
    )
    words = (
        text.str.extract_all(r"[a-z]{4,}")
        .list.eval(pl.element().filter(~pl.element().is_in(GENERIC_STREET)))
        .list.unique(maintain_order=True)
    )
    return records.select("idx", numbers=numbers, words=words).with_columns(
        house=pl.col("numbers").list.first(),
        street=pl.col("words").list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True)).list.first(),
        words=pl.col("words").list.head(max_words),
    )


def learn_transliteration(s1: pl.DataFrame, s23: pl.DataFrame, links: pl.DataFrame) -> pl.DataFrame:
    """Learn an Indic-token to Latin-token table from position-aligned true pairs.

    ``links`` holds ``s1`` and ``s23`` row indices. Tokens are folded with :func:`folded_text`,
    the same form :func:`name_tokens` looks them up in. Only pairs whose S2/S3 name is
    entirely non-Latin and whose token counts match are used; each Indic token maps
    to its most frequent Latin counterpart.
    """
    pairs = (
        links.join(raw_tokens(s1, "s1"), on="s1")
        .join(raw_tokens(s23.filter(pl.col("business_name").str.contains(INDIC) & ~pl.col("business_name").str.contains(r"[A-Za-z]")), "s23"), on="s23")
        .filter(pl.col("t_s1").list.len() == pl.col("t_s23").list.len())
        .select(latin="t_s1", token="t_s23")
        .explode(["latin", "token"])
        .filter(pl.col("token").str.contains(INDIC) & pl.col("latin").str.contains(r"^[a-z]+$"))
    )
    return (
        pairs.group_by("token", "latin").len()
        .sort("len", descending=True)
        .group_by("token", maintain_order=True)
        .agg(pl.col("latin").first())
    )


def raw_tokens(frame: pl.DataFrame, side: str) -> pl.DataFrame:
    """Return the record index as ``side`` and the folded name tokens as ``t_{side}``, before any mapping."""
    return frame.select(pl.col("idx").alias(side), folded_text("business_name").str.extract_all(TOKEN).alias(f"t_{side}"))


def learn_phrases(s1: pl.DataFrame, s23: pl.DataFrame, links: pl.DataFrame, table: pl.DataFrame, min_support: int = 3) -> pl.DataFrame:
    """Learn runs of consecutive Indic tokens, missing from ``table``, that stand for one Latin token.

    English words are often written as two or three Indic tokens ("soft ware", "manage ment",
    "L L P"), which the position-aligned learner cannot pair with the one Latin token of the S1
    name. Each link whose S2/S3 name has exactly one such run, and whose S1 name has exactly one
    token that the mapped S2/S3 tokens do not contain, is a vote for that token. A run is kept when
    its most voted token has at least ``min_support`` votes and a majority.
    Returns ``token`` (the run joined by spaces) and
    ``latin``, the columns of :func:`learn_transliteration`.
    """
    mapping = dict(table.select("token", "latin").iter_rows())
    pairs = links.join(raw_tokens(s1, "s1"), on="s1").join(raw_tokens(s23.filter(pl.col("business_name").str.contains(INDIC)), "s23"), on="s23")
    votes: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for s1_tokens, s23_tokens in pairs.select("t_s1", "t_s23").iter_rows():
        mapped, runs, run = set(), [], []
        for word in s23_tokens:
            if word not in mapping and any(0x900 <= ord(ch) < 0xE00 for ch in word):
                run.append(word)
                continue
            if run:
                runs.append(tuple(run))
                run = []
            mapped.add(mapping.get(word, word))
        if run:
            runs.append(tuple(run))
        left = [word for word in s1_tokens if word not in mapped]
        if len(runs) == 1 and len(left) == 1:
            votes[runs[0]][left[0]] += 1
    rows = []
    for run, counter in votes.items():
        latin, count = counter.most_common(1)[0]
        if count >= min_support and count / sum(counter.values()) >= 0.5:
            rows.append((" ".join(run), latin))
    return pl.DataFrame(rows, schema={"token": pl.String, "latin": pl.String}, orient="row")
