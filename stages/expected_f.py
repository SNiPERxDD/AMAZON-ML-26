"""Choose each S1's links by maximising its expected F0.5 under the calibrated ``xedit`` probabilities, instead of the fixed 0.5/0.85 rule.

The ``xedit`` probabilities are calibrated (a pair at 0.25 is true 26% of the time, at 0.75 73%), so the link count
that maximises the S1's expected score can be computed rather than guessed: candidates are sorted by probability,
``SAMPLES`` Monte Carlo worlds draw each candidate's truth independently plus a Poisson count of true links outside
the band (the country's hold-out mean), and the top-k with the best mean F0.5 is kept. F0.5 is 1 for an empty
prediction with empty truth and 0 when no link is right, so predicting nothing is weighed against the chance the S1
has no links at all.

``--eval`` compares the rule with the expected-F choice on the US/India hold-out. ``--probe NAME`` replaces the US/India
rule links of ``--base`` (the 0.5/0.85 rule on ``data/kaggle_edit_r2/test_scores_r.parquet``) with the expected-F
choice, keeps every other link of the base (address-only, no-address and France rounds) and writes
``output/probes/NAME/``.

Run from the repository root: ``PYTHONPATH=. python stages/expected_f.py --eval``.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from pipeline import metric, records
from stages.addr_only_probe import read_lists, write_lists

ROOT = "data/kaggle_edit_r2/"
SCORE = "H+words + all crossed"
TOP = 16
SAMPLES = 256
CHUNK = 10_000
# an S1 whose candidates all sit outside (LOW, HIGH) keeps x >= 0.5 without sampling
LOW, HIGH = 0.02, 0.98
# mean number of true links per S1 outside the band, hold-out
OUTSIDE = {"US": 0.0811, "India": 0.1195, "France": 0.1}


def choose(band: pl.DataFrame, lam: float, seed: int = 0, beta2: float = 0.25) -> pl.DataFrame:
    """Return the (s1, s23) links that maximise each S1's Monte Carlo expected F-beta; ``band`` has s1, s23, x."""
    unsure = pl.col("x").is_between(LOW, HIGH).any().over("s1")
    sure = band.filter(~unsure & (pl.col("x") >= 0.5)).select("s1", "s23")
    b = (band.filter(unsure).sort(["s1", "x"], descending=[False, True]).with_columns(r=pl.int_range(pl.len()).over("s1"))
         .filter(pl.col("r") < TOP))
    row = b.select(pl.col("s1").rank("dense") - 1)["s1"].to_numpy()
    s1s = b["s1"].unique(maintain_order=True)
    P = np.zeros((len(s1s), TOP), dtype=np.float32)
    P[row, b["r"].to_numpy()] = b["x"].to_numpy()
    n_cand = np.bincount(row, minlength=len(s1s))
    rng = np.random.default_rng(seed)
    best_k = np.zeros(len(s1s), dtype=np.int64)
    k = np.arange(TOP + 1, dtype=np.float32)
    for start in range(0, len(s1s), CHUNK):
        p = P[start:start + CHUNK]
        draws = rng.random((SAMPLES, *p.shape), dtype=np.float32) < p
        tp = np.concatenate([np.zeros((SAMPLES, p.shape[0], 1), np.float32), np.cumsum(draws, axis=2, dtype=np.float32)], axis=2)
        total = tp[:, :, -1:] + rng.poisson(lam, (SAMPLES, p.shape[0], 1)).astype(np.float32)
        denom = (1 + beta2) * tp + beta2 * (total - tp) + (k - tp)
        f = np.where(tp > 0, (1 + beta2) * tp / np.maximum(denom, 1e-9), 0.0)
        f[:, :, 0] = (total[:, :, 0] == 0)
        ef = f.mean(axis=0)
        ef[np.arange(TOP + 1)[None, :] > n_cand[start:start + CHUNK, None]] = -1
        best_k[start:start + CHUNK] = ef.argmax(axis=1)
    kk = pl.DataFrame({"s1": s1s, "k": best_k})
    return pl.concat([sure, b.join(kk, on="s1").filter(pl.col("r") < pl.col("k")).select("s1", "s23")])


def evaluate() -> None:
    band = pl.read_parquet(f"{ROOT}r2_band.parquet", columns=["s1", "s23"]).join(
        pl.read_parquet(f"{ROOT}xedit_holdout_scores.parquet", columns=["s1", "s23", SCORE]), on=["s1", "s23"]).rename({SCORE: "x"})
    ids = pl.read_parquet(f"{ROOT}holdout_ids.parquet").filter(pl.col("country") != "France")
    truth = pl.read_parquet(f"{ROOT}truth.parquet").join(ids, on="s1", how="semi").select("s1", "s23")
    rule = band.filter((pl.col("x") >= 0.5) & (pl.col("x") >= 0.85 * pl.col("x").max().over("s1"))).select("s1", "s23")
    for country in ("US", "India"):
        sid = ids.filter(pl.col("country") == country).select("s1")
        t = truth.join(sid, on="s1", how="semi")
        b = band.join(sid, on="s1", how="semi")
        f0 = metric.per_entity(rule.join(sid, on="s1", how="semi"), t, sid)["f05"].mean()
        for lam in (0.0, OUTSIDE[country], 2 * OUTSIDE[country]):
            c = choose(b, lam)
            f1 = metric.per_entity(c, t, sid)["f05"].mean()
            print(f"{country} lam={lam:.3f}: rule links {rule.join(sid, on='s1', how='semi').height} F0.5 {f0:.5f}; "
                  f"expected-F links {c.height} F0.5 {f1:.5f} ({f1 - f0:+.5f})", flush=True)


def probe(name: str, base_path: str, odds: float) -> None:
    s1, s23 = records.load_split("test")
    ids1, ids2 = s1.select(s1="idx", source1_entity_id="entity_id", country="country"), s23.select(s23="idx", e="entity_id")
    del s1, s23
    base = Path(base_path)
    links = read_lists(base, "matched_entity_ids").join(ids1, on="source1_entity_id").join(ids2, on="e").select("s1", "s23")
    x = pl.read_parquet(f"{ROOT}test_scores_r.parquet").rename({"xedit": "x"}).join(ids1.select("s1", "country"), on="s1")
    x = x.filter(pl.col("country") != "France")
    rule = x.filter((pl.col("x") >= 0.5) & (pl.col("x") >= 0.85 * pl.col("x").max().over("s1"))).select("s1", "s23")
    # leaderboard steps put test odds at ``odds`` times the hold-out odds
    xk = x.with_columns(x=odds * pl.col("x") / (odds * pl.col("x") + 1 - pl.col("x")))
    chosen = pl.concat([choose(xk.filter(pl.col("country") == c), OUTSIDE[c]) for c in ("US", "India")])
    kept = pl.concat([links.join(rule, on=["s1", "s23"], how="anti"), chosen]).unique()
    print(f"base {links.height}, rule {rule.height} -> expected-F {chosen.height}; links {kept.height} "
          f"(added {chosen.join(rule, on=['s1', 's23'], how='anti').height}, dropped {rule.join(chosen, on=['s1', 's23'], how='anti').height})", flush=True)
    kept = kept.join(ids1.select("s1", "source1_entity_id"), on="s1").join(ids2, on="s23")
    out = Path("output/probes") / name
    out.mkdir(parents=True, exist_ok=True)
    write_lists(ids1, kept, "matched_entity_ids", out / "matching_results.tsv")
    cands = pl.concat([read_lists(base.parent / "candidate_pairs.tsv", "candidate_entity_ids"), kept.select("source1_entity_id", "e")])
    write_lists(ids1, cands, "candidate_entity_ids", out / "candidate_pairs.tsv")
    print(f"{out}: links {links.height} -> {kept.height}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--probe", default="")
    parser.add_argument("--base", default="output/matching_results.tsv")
    parser.add_argument("--odds", type=float, default=1.0, help="test/hold-out odds ratio applied to the test probabilities")
    args = parser.parse_args()
    if args.eval:
        evaluate()
    if args.probe:
        probe(args.probe, args.base, args.odds)

if __name__ == "__main__":
    main()
