"""Does S2/S3 -> S1 character-trigram retrieval find the true S1 of train links that no candidate round proposes?

219,511 train true links are in no candidate set (``missed_link_profile.py``). This samples some
of them per country and, for each S2/S3 record, ranks every S1 record of the country by the
cosine of binary, IDF-weighted, word-bounded character trigrams (``char_wb``). Two texts:
``name`` (the normalised core name) and ``name+address`` (core name, address words and numbers).
Trigrams in more than ``DF_CAP`` S1 records are left out of the product for tractability (they
still count in the norms), so the ranking is approximate.

Printed per country and text, for K in ``KS``:

- missed sample: the share whose true S1 is in the top K, split by empty S2/S3 address and, for
  India, by non-ASCII core name;
- random S2/S3 sample: top-K pairs per record not already a candidate, their true share, and the
  implied new pairs per S1 for a full run; also restricted to S2/S3 records whose best base
  probability over the current rounds is below 0.5 (records without a confident S1).

Run from the repository root: ``PYTHONPATH=. python stages/reverse_trigram_probe.py``.
"""

from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from pipeline import features, records, refine
from stages.missed_link_profile import missed

SAMPLE = 6000
KS = (1, 2, 3, 5, 10)
DF_CAP = 5000
TEXTS = {
    "name": pl.col("core"),
    "name+address": pl.concat_str([pl.col("core"), pl.col("words").list.join(" "), pl.col("numbers").list.join(" ")], separator=" "),
}


def trigrams(frame: pl.DataFrame, text: pl.Expr) -> pl.DataFrame:
    """Return the distinct (``idx``, ``tri``) word-bounded character trigrams of ``text``."""
    tokens = frame.select("idx", tok=text.str.split(" ")).explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
    tokens = tokens.with_columns(tok=pl.concat_str([pl.lit(" "), pl.col("tok"), pl.lit(" ")]))
    tokens = tokens.with_columns(off=pl.int_ranges(0, pl.col("tok").str.len_chars() - 2)).explode("off").drop_nulls("off")
    return tokens.select("idx", tri=pl.col("tok").str.slice(pl.col("off"), 3)).unique()


def matrix(tri: pl.DataFrame, rows: pl.DataFrame, vocab: pl.DataFrame) -> sp.csr_matrix:
    """Return the L2-normalised binary IDF matrix of ``tri`` with rows ordered as ``rows`` and columns kept by ``vocab``."""
    t = tri.join(rows.with_row_index("r"), on="idx").join(vocab, on="tri")
    t = t.with_columns(norm=(pl.col("idf") ** 2).sum().over("r").sqrt()).filter(pl.col("keep"))
    return sp.csr_matrix(((t["idf"] / t["norm"]).to_numpy(), (t["r"].to_numpy(), t["col"].to_numpy())), shape=(rows.height, vocab.height))


def top_k(q: sp.csr_matrix, s1t: sp.csr_matrix, k: int) -> list[np.ndarray]:
    """Return, per query row, the S1 row positions of the ``k`` highest cosines (best first)."""
    out = []
    for start in range(0, q.shape[0], 200):
        prod = (q[start:start + 200] @ s1t).tocsr()
        for i in range(prod.shape[0]):
            lo, hi = prod.indptr[i], prod.indptr[i + 1]
            data, cols = prod.data[lo:hi], prod.indices[lo:hi]
            order = np.argsort(-data, kind="stable")[:k] if hi - lo <= k else np.argpartition(-data, k)[:k]
            order = order[np.argsort(-data[order], kind="stable")]
            out.append(cols[order])
    return out


def main() -> None:
    """Print the top-K recovery of missed true links and the new-pair load of a full reverse run, per country and text."""
    cols = ["idx", "country", "core", "words", "numbers"]
    s1 = pl.read_parquet(records.records_path("train", "s1"), columns=cols)
    s23 = pl.read_parquet(records.records_path("train", "s23"), columns=cols)
    truth = features.truth_links()
    lost = missed()
    scored = pl.scan_parquet("data/candidates/train_pairs.parquet").select("s1", "s23")
    extra = [pl.scan_parquet(str(path)).select("s1", "s23") for name in ("sibling", "namepair") for path in Path(f"{features.FEATURE_DIR}train").glob(f"{name}_*.parquet")]
    best = refine.rounds_scan("train", features.FEATURE_DIR).group_by("s23").agg(best=pl.col("p").max()).collect()
    kmax = max(KS)
    for country in ("US", "India"):
        c1 = s1.filter(pl.col("country") == country)
        c23 = s23.filter(pl.col("country") == country)
        miss = lost.join(c23.select(s23="idx"), on="s23", how="semi").sample(SAMPLE, seed=0)
        rand = c23.select(s23="idx").sample(SAMPLE, seed=1)
        queries = c23.filter(pl.col("idx").is_in(pl.concat([miss["s23"], rand["s23"]]).unique().implode()))
        known = pl.concat([scored, *extra]).join(rand.lazy(), on="s23", how="semi").unique().collect()
        print(f"== {country}: S1 {c1.height}, S2/S3 {c23.height}, missed links {lost.join(c23.select(s23='idx'), on='s23', how='semi').height}", flush=True)
        empty = c23.select(s23="idx", empty=(pl.col("words").list.len() == 0) & (pl.col("numbers").list.len() == 0), nonascii=pl.col("core").str.contains(r"[^\x00-\x7f]"))
        for name, text in TEXTS.items():
            t1 = trigrams(c1, text)
            vocab = t1.group_by("tri").agg(df=pl.len()).with_columns(idf=(np.log((c1.height + 1) / (pl.col("df") + 1)) + 1), keep=pl.col("df") <= DF_CAP)
            vocab = vocab.sort("tri").with_row_index("col")
            s1_rows = c1.select("idx")
            s1t = matrix(t1, s1_rows, vocab).T.tocsr()
            q_rows = queries.select("idx")
            tops = top_k(matrix(trigrams(queries, text), q_rows, vocab), s1t, kmax)
            s1_idx = s1_rows["idx"].to_numpy()
            ranked = pl.DataFrame({"s23": q_rows["idx"], "s1": [s1_idx[t].tolist() for t in tops]}, schema={"s23": pl.UInt32, "s1": pl.List(pl.UInt32)})
            ranked = ranked.explode("s1").drop_nulls("s1").with_columns(rank=pl.int_range(1, pl.len() + 1).over("s23"))
            hit = miss.join(ranked, on=["s1", "s23"], how="left").join(empty, on="s23")
            parts = {"all": hit, "empty address": hit.filter(pl.col("empty")), "with address": hit.filter(~pl.col("empty"))}
            if country == "India":
                parts |= {"non-ASCII name": hit.filter(pl.col("nonascii")), "ASCII name": hit.filter(~pl.col("nonascii"))}
            for label, part in parts.items():
                print(f"{country} {name} missed {label} (n={part.height}): " + ", ".join(f"top{k} {(part['rank'] <= k).sum() / max(part.height, 1):.3f}" for k in KS), flush=True)
            load = ranked.join(rand, on="s23", how="semi").join(known, on=["s1", "s23"], how="anti")
            load = load.join(truth.with_columns(t=pl.lit(True)), on=["s1", "s23"], how="left").with_columns(pl.col("t").fill_null(False))
            load = load.join(best, on="s23", how="left").with_columns(unmatched=pl.col("best").fill_null(0.0) < 0.5)
            gated = hit.join(best, on="s23", how="left").filter(pl.col("best").fill_null(0.0) < 0.5)
            print(f"{country} {name} missed with best base p < 0.5: {gated.height / hit.height:.3f} of the sample; "
                  + ", ".join(f"top{k} {(gated['rank'] <= k).sum() / hit.height:.3f}" for k in KS) + " (share of all missed)", flush=True)
            for gate, frame in (("all records", load), ("best base p < 0.5", load.filter(pl.col("unmatched")))):
                for k in KS:
                    new = frame.filter(pl.col("rank") <= k)
                    per_s1 = new.height / SAMPLE * c23.height / c1.height
                    print(f"{country} {name} random {gate} top{k}: new pairs per S2/S3 {new.height / SAMPLE:.3f}, true share {new['t'].mean() or 0:.4f}, "
                          f"implied new pairs per S1 {per_s1:.2f}, implied new true per S1 {new['t'].sum() / SAMPLE * c23.height / c1.height:.4f}", flush=True)


if __name__ == "__main__":
    main()
