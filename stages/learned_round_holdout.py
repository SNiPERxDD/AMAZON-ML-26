"""Hold-out learned retrieval round.

The byte-CNN tower of ``learned_retrieval_probe.py`` is trained on train links whose S1 is outside the hold-out fold and
saved. Every train S2/S3 record with an address whose best probability over ``BASE`` is below 0.5 (the gate of
``pipeline.reverse``) queries all S1 records of its country; the top ``K`` neighbours whose S1 is in the hold-out fold and
that no earlier round proposed become new pairs. Ranks are over all S1, as they would be in production.

The pairs get the first-round pair features, per-S1 window and S2/S3 context as in ``pipeline.reverse``, the stage-1
probability ``p1`` (hold-out model, no candidate family), base-probability context, and retrieval features (rank, cosine,
distance to rank 1 and to the next rank, rank-1 minus rank-2 cosine). A LightGBM model is cross-fitted over two S1-hash
halves of the hold-out proposals.

Evaluation per country, US and India, on the hold-out:

- AUC of ``p1``, cosine, gap and the cross-fitted score ``q`` on the new pairs;
- per score cut: accepted pairs per S1, true share, recovered links, and the macro F0.5 change over the ``probe_xedit``
  decision (xedit scores on the band at 0.5/0.85). Two policies: ``add`` keeps accepted pairs on top of the band decision;
  ``rule`` pools them with the band scores under the relative rule. Ranks 1 only and ranks 1-3.

The cut is chosen on the same hold-out, so the best row is optimistic. Stop criterion: weighted US/India gain < +0.0007
(weights: test S1 shares 0.38 / 0.47), or either country < -0.0002. Strong pass: >= +0.0010.

``--queries empty`` runs the same round for gated S2/S3 records without an address (name-only text), with its own
cached files (``empty_*``); the ``add`` rows then measure it on top of the band decision alone, not on top of the address round.

Steps, cached under ``data/learned/``: ``tower``, ``pairs``, ``features``, ``eval``; ``--step all`` runs the missing ones.
Run from the repository root under the guard, for example:
``tools/run_guarded.sh --max-gb 12 --max-swap-gb 3 --log logs/learned_round.log -- env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 PYTHONPATH=. python stages/learned_round_holdout.py``.
"""

import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from pipeline import features, namepairs, records, reverse, siblings, train
from stages.char_pair_probe import Tower
from stages.learned_retrieval_probe import encode, to_device, train_tower

OUT = Path("data/learned")
TOWER = OUT / "tower_holdout.pt"
PAIRS = OUT / "holdout_pairs.parquet"
FEATS = OUT / "holdout_features.parquet"
R = Path("data/kaggle_edit_r2")
SCORE = "H+words + all crossed"
BASE = "blend+sibling+namepair+reverse"
K = 3
THRESHOLD, RELATIVE = 0.5, 0.85
WEIGHTS = {"US": 0.38, "India": 0.47}
CUTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
COLS = ["idx", "country", "core", "words", "numbers"]
HAS_ADDRESS = (pl.col("words").list.len() > 0) | (pl.col("numbers").list.len() > 0)


def byte_table(side: str, ids: pl.LazyFrame | None = None, split: str = "train") -> pl.DataFrame:
    """Return the ``split`` records of ``side`` (optionally restricted to ``ids``), sorted by ``idx``."""
    scan = pl.scan_parquet(records.records_path(split, side)).select(COLS)
    if ids is not None:
        scan = scan.join(ids.select("idx").unique(), on="idx", how="semi")
    return scan.collect(engine="streaming").sort("idx")


def embed(tower: Tower, rows: np.ndarray, device: str, batch: int = 8192) -> torch.Tensor:
    """Unit-norm embeddings of byte rows, kept on ``device``."""
    tower.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            out.append(F.normalize(tower(to_device(rows[i:i + batch], device)), dim=1))
    return torch.cat(out)


def chunk_topk(s: torch.Tensor, k: int, group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact row-wise top-k via group maxima: the k largest values lie in the k groups with the largest maxima.

    MPS ``topk`` over wide rows is about 90x slower than a grouped max, so the full-width ``topk`` runs only on k groups.
    """
    pad = -s.shape[1] % group
    if pad:
        s = F.pad(s, (0, pad), value=-2.0)
    g = s.view(len(s), -1, group)
    top_g = g.amax(2).topk(k, dim=1).indices
    v, i = g.gather(1, top_g[:, :, None].expand(-1, -1, group)).reshape(len(s), -1).topk(k, dim=1)
    return v, top_g.gather(1, i // group) * group + i % group


def search(queries: torch.Tensor, pool: torch.Tensor, k: int, block: int = 2048, chunk: int = 65_536) -> tuple[np.ndarray, np.ndarray]:
    """Pool positions and cosines of the ``k`` nearest pool rows per query, on the tensors' device."""
    idx_all, sim_all = [], []
    for q in range(0, len(queries), block):
        qb = queries[q:q + block]
        best_s = torch.full((len(qb), k), -2.0, device=qb.device)
        best_i = torch.zeros((len(qb), k), dtype=torch.long, device=qb.device)
        for c in range(0, len(pool), chunk):
            s, i = chunk_topk(qb @ pool[c:c + chunk].T, k)
            best_s, pick = torch.cat([best_s, s], 1).topk(k, dim=1)
            best_i = torch.cat([best_i, i + c], 1).gather(1, pick)
            if qb.device.type == "mps":
                torch.mps.synchronize()  # MPS queues the (block x chunk) score matrices asynchronously; unsynced they pile up past RAM
        idx_all.append(best_i.cpu().numpy())
        sim_all.append(best_s.cpu().numpy())
    return np.concatenate(idx_all), np.concatenate(sim_all)


def step_tower(args: argparse.Namespace, started: float) -> None:
    """Train the tower on 1.5M links whose S1 is outside the hold-out fold and save it."""
    s1 = byte_table("s1")
    fit = features.truth_links().filter(train.fold() != train.HOLDOUT).sample(args.links, seed=0)
    s23 = byte_table("s23", fit.lazy().select(idx="s23"))
    idx1, idx23 = s1["idx"].to_numpy(), s23["idx"].to_numpy()
    prefix = s1.select(pl.col("core").fill_null("").str.slice(0, 4))["core"].to_numpy()
    r1 = np.searchsorted(idx1, fit["s1"].to_numpy())
    fit = fit.with_columns(r1=pl.Series(r1), r23=pl.Series(np.searchsorted(idx23, fit["s23"].to_numpy())), prefix=pl.Series(prefix[r1]))
    s1_bytes, s23_bytes = encode(s1), encode(s23)
    del s1, s23
    gc.collect()
    print(f"phase: tower fit links {fit.height}, {(time.time() - started) / 60:.1f} min", flush=True)
    tower = train_tower(fit, s1_bytes, s23_bytes, args, started)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(tower.state_dict(), TOWER)
    print(f"phase: tower saved, {(time.time() - started) / 60:.1f} min", flush=True)


def step_pairs(args: argparse.Namespace, started: float) -> None:
    """Retrieve the top ``K`` S1 of every gated train S2/S3 record and keep new pairs whose S1 is in the hold-out fold."""
    tower = Tower(width=128).to(args.device)
    tower.load_state_dict(torch.load(TOWER, map_location=args.device))
    best = train.prob_scan("train", features.FEATURE_DIR, BASE).group_by("s23").agg(best=pl.col("p").max())
    has = HAS_ADDRESS if args.queries == "address" else ~HAS_ADDRESS
    gated = (pl.scan_parquet(records.records_path("train", "s23")).select(COLS).filter(has)
             .join(best, left_on="idx", right_on="s23", how="left").filter(pl.col("best").fill_null(0.0) < reverse.GATE)
             .drop("best").collect(engine="streaming"))
    print(f"phase: gated S2/S3 {gated.height}, {(time.time() - started) / 60:.1f} min", flush=True)
    holdout = pl.read_parquet(R / "holdout_ids.parquet", columns=["s1"])
    found = []
    for country in ("US", "India"):
        c1 = byte_table("s1").filter(pl.col("country") == country)
        q = gated.filter(pl.col("country") == country)
        pool = embed(tower, encode(c1), args.device)
        c1_idx = c1["idx"].to_numpy()
        for a in range(0, q.height, args.query_chunk):  # chunked: the full gated set does not fit next to the pool on a 24 GB machine
            qc = q.slice(a, args.query_chunk)
            idx, sim = search(embed(tower, encode(qc), args.device), pool, K + 1)
            for r in range(K):
                found.append(pl.DataFrame({
                    "s23": qc["idx"].to_numpy(), "s1": c1_idx[idx[:, r]], "rank": np.full(qc.height, r + 1),
                    "cos": sim[:, r], "d_top": sim[:, 0] - sim[:, r], "d_next": sim[:, r] - sim[:, r + 1], "gap12": sim[:, 0] - sim[:, 1],
                }).with_columns(pl.col("s1", "s23").cast(pl.UInt32), country=pl.lit(country), rank=pl.col("rank").cast(pl.UInt8))
                    .with_columns(pl.col("cos", "d_top", "d_next", "gap12").cast(pl.Float32)).join(holdout, on="s1", how="semi"))
            del idx, sim
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"phase: {country} queries {a + qc.height}/{q.height}, {(time.time() - started) / 60:.1f} min", flush=True)
        del pool
        print(f"phase: {country} gated S2/S3 {q.height}, S1 pool {c1.height}, {(time.time() - started) / 60:.1f} min", flush=True)
        del c1, q
        gc.collect()
    known = pl.concat([
        pl.scan_parquet(f"{records.CACHE_DIR}train_pairs.parquet").select("s1", "s23"),
        pl.scan_parquet(siblings.pairs_path("train")).select("s1", "s23"),
        pl.scan_parquet(namepairs.pairs_path("train")).select("s1", "s23"),
        pl.scan_parquet(reverse.pairs_path("train")).select("s1", "s23"),
    ])
    pairs = pl.concat(found).lazy().join(known, on=["s1", "s23"], how="anti").collect(engine="streaming").sort("s1", "s23")
    pairs.write_parquet(PAIRS)
    print(f"phase: new hold-out pairs {pairs.height}, per country and rank {sorted(pairs.group_by('country', 'rank').len().rows())}", flush=True)


def step_features(args: argparse.Namespace, started: float) -> None:
    """Pair features, window, context, ``p1`` and base context of the new pairs, with labels (as ``pipeline.reverse``)."""
    pairs_all = pl.read_parquet(PAIRS)
    flags = [c for c in pl.scan_parquet(f"{features.FEATURE_DIR}train/part_000.parquet").collect_schema().names() if c.startswith("f_")]
    per_s1, per_s23 = namepairs.base_context("train", features.FEATURE_DIR, BASE)
    n_s2 = pl.scan_parquet(f"{records.PARQUET_DIR}train_source2.parquet").select(pl.len()).collect().item()
    truth = features.truth_links().with_columns(label=pl.lit(True))
    parts = []
    high = int(pairs_all["s1"].max()) + 1
    for low in range(0, high, reverse.CHUNK_S1):
        pairs = pairs_all.filter(pl.col("s1").is_between(low, low + reverse.CHUNK_S1, closed="left"))
        if pairs.is_empty():
            continue
        s1 = features.side_table("train", "s1", pairs.select(idx="s1").unique())
        s23 = features.side_table("train", "s23", pairs.select(idx="s23").unique())
        frame = features.pair_features(pairs.select("s1", "s23"), s1, s23, n_s2)
        frame = reverse.with_context(reverse.with_window(frame, "train", features.FEATURE_DIR), "train", features.FEATURE_DIR)
        frame = frame.with_columns(*[pl.lit(False).alias(f) for f in flags], n_families=pl.lit(0, pl.UInt8))
        frame = frame.with_columns(p1=pl.Series(reverse.stage1(frame, "train", args.threads))).drop(*flags, "n_families")
        frame = (frame.join(pairs.drop("country"), on=["s1", "s23"], how="left").join(per_s1, on="s1", how="left").join(per_s23, on="s23", how="left")
                 .join(truth, on=["s1", "s23"], how="left").with_columns(pl.col("label").fill_null(False)))
        parts.append(frame)
        print(f"phase: features S1 from {low}: {pairs.height} pairs, {(time.time() - started) / 60:.1f} min", flush=True)
        del pairs, s1, s23, frame
        gc.collect()
    pl.concat(parts, how="diagonal_relaxed").write_parquet(FEATS)


def cross_fit(frame: pl.DataFrame, threads: int) -> np.ndarray:
    """Return scores cross-fitted over two S1-hash halves; each fit early-stops on a fifth of its own half's S1."""
    columns = [c for c in frame.columns if c not in (*train.NON_FEATURES, "half", "inner")]
    frame = frame.with_columns(half=pl.col("s1").hash(11) % 2, inner=pl.col("s1").hash(13) % 5)
    q = np.zeros(frame.height, np.float32)
    for h in (0, 1):
        own = frame.filter(pl.col("half") != h)
        fit_rows, stop_rows = own.filter(pl.col("inner") != 0), own.filter(pl.col("inner") == 0)
        dtrain = lgb.Dataset(train.matrix(fit_rows, columns), fit_rows["label"].to_numpy(), feature_name=columns)
        dvalid = lgb.Dataset(train.matrix(stop_rows, columns), stop_rows["label"].to_numpy(), reference=dtrain)
        booster = lgb.train(reverse.PARAMS | {"num_threads": threads}, dtrain, 2000, valid_sets=[dvalid], callbacks=[lgb.early_stopping(50, verbose=False)])
        rows = (frame["half"] == h).to_numpy()
        q[rows] = booster.predict(train.matrix(frame.filter(pl.col("half") == h), columns), num_iteration=booster.best_iteration, num_threads=threads)
        gain = booster.feature_importance("gain")
        top = sorted(zip(columns, gain / gain.sum(), strict=True), key=lambda item: -item[1])[:8]
        print(f"half {h}: rows {fit_rows.height}, positives {fit_rows['label'].sum()}, best iteration {booster.best_iteration}; "
              + ", ".join(f"{n}={s:.3f}" for n, s in top), flush=True)
    return q


def f05(tp: pl.Expr, kept: pl.Expr, n_true: pl.Expr) -> pl.Expr:
    """Per-S1 F0.5 (1 when nothing is kept and nothing is true)."""
    fp, fn = kept - tp, n_true - tp
    return pl.when(kept + n_true == 0).then(1.0).otherwise(1.25 * tp / (1.25 * tp + 0.25 * fn + fp))


def rule_counts(rows: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 kept and true-kept counts under the test rule over ``rows`` (``s1``, ``x``, ``t``)."""
    kept = (pl.col("x") >= THRESHOLD) & (pl.col("x") >= RELATIVE * pl.col("x").max().over("s1"))
    return rows.group_by("s1").agg(kept=kept.sum(), tp=(kept & pl.col("t")).sum())


def step_eval(args: argparse.Namespace) -> None:
    """Cross-fit the round model and print AUCs and the F0.5 frontier per country, policy and rank set."""
    frame = pl.read_parquet(FEATS).filter(pl.col("country").is_in(list(WEIGHTS)))
    frame = frame.with_columns(q=pl.Series(cross_fit(frame.drop("country"), args.threads)))
    band = (pl.read_parquet(R / "r2_band.parquet", columns=["s1", "s23", "country", "t"])
            .join(pl.read_parquet(R / "xedit_holdout_scores.parquet", columns=["s1", "s23", SCORE]).rename({SCORE: "x"}), on=["s1", "s23"])
            .with_columns(pl.col("x").cast(pl.Float64), pl.col("t").cast(pl.Boolean)))
    n_true = pl.read_parquet(R / "truth.parquet").group_by("s1").agg(n_true=pl.len())
    ids = pl.read_parquet(R / "holdout_ids.parquet")
    total = {}
    for country, weight in WEIGHTS.items():
        c = frame.filter(pl.col("country") == country)
        n_s1 = ids.filter(pl.col("country") == country).height
        y = c["label"].to_numpy()
        print(f"{country}: new pairs {c.height} ({c.height / n_s1:.3f} per S1), true {int(y.sum())} (share {y.mean():.4f}); AUC "
              + ", ".join(f"{name} {roc_auc_score(y, c[name].to_numpy()):.4f}" for name in ("p1", "cos", "gap12", "q")), flush=True)
        cb = band.filter(pl.col("country") == country)
        before = rule_counts(cb.select("s1", "x", "t"))
        for ranks in (1, K):
            for cut in CUTS:
                acc = c.filter((pl.col("rank") <= ranks) & (pl.col("q") >= cut)).select("s1", x=pl.col("q").cast(pl.Float64), t="label")
                if acc.is_empty():
                    continue
                affected = acc.select("s1").unique().join(n_true, on="s1", how="left").with_columns(pl.col("n_true").fill_null(0))
                base = affected.join(before, on="s1", how="left").with_columns(pl.col("kept", "tp").fill_null(0))
                f_before = base.select(f05(pl.col("tp"), pl.col("kept"), pl.col("n_true")).sum()).item()
                added = acc.group_by("s1").agg(a=pl.len(), at=pl.col("t").sum())
                add = base.join(added, on="s1").select(f05(pl.col("tp") + pl.col("at"), pl.col("kept") + pl.col("a"), pl.col("n_true")).sum()).item()
                pooled = rule_counts(pl.concat([cb.join(affected, on="s1", how="semi").select("s1", "x", "t"), acc]))
                rule = affected.join(pooled, on="s1", how="left").with_columns(pl.col("kept", "tp").fill_null(0))
                rule = rule.select(f05(pl.col("tp"), pl.col("kept"), pl.col("n_true")).sum()).item()
                d_add, d_rule = (add - f_before) / n_s1, (rule - f_before) / n_s1
                total[(ranks, cut, "add")] = total.get((ranks, cut, "add"), 0.0) + weight * d_add
                total[(ranks, cut, "rule")] = total.get((ranks, cut, "rule"), 0.0) + weight * d_rule
                print(f"{country} ranks 1-{ranks} q >= {cut}: accepted {acc.height} ({acc.height / n_s1:.4f}/S1), true share {acc['t'].mean():.3f}, "
                      f"recovered {int(acc['t'].sum())}; F0.5 add {d_add:+.5f}, rule {d_rule:+.5f}", flush=True)
    best = max(total.items(), key=lambda item: item[1])
    print(f"weighted (0.38 US + 0.47 India) best: ranks 1-{best[0][0]} q >= {best[0][1]} {best[0][2]} {best[1]:+.5f}; kill < +0.0007, strong >= +0.0010", flush=True)


def main() -> None:
    """Run the requested step, or every step whose output is missing."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", default="all", choices=["all", "tower", "pairs", "features", "eval"])
    parser.add_argument("--links", type=int, default=1_500_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--query_chunk", type=int, default=400_000)
    parser.add_argument("--queries", default="address", choices=["address", "empty"])
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    global PAIRS, FEATS
    if args.queries == "empty":
        PAIRS, FEATS = OUT / "empty_holdout_pairs.parquet", OUT / "empty_holdout_features.parquet"
    started = time.time()
    torch.manual_seed(42)
    steps = [("tower", TOWER, step_tower), ("pairs", PAIRS, step_pairs), ("features", FEATS, step_features)]
    for name, path, run in steps:
        if args.step == name or (args.step == "all" and not path.exists()):
            run(args, started)
    if args.step in ("all", "eval"):
        step_eval(args)
    print(f"phase: done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
