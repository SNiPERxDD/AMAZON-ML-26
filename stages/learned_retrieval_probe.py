"""Frontier test for learned S2/S3 -> S1 retrieval: can a byte-CNN two-tower encoder find never-proposed links?

A shared byte-CNN tower (``char_pair_probe.Tower``) embeds ``core | address words
numbers`` of each record (UTF-8 bytes, at most ``LENGTH``) into a unit vector. It is trained on true train links whose S1 is
outside the hold-out fold, with in-batch negatives (symmetric InfoNCE, temperature ``TAU``). Every other batch is drawn from
links sorted by the S1 core-name prefix, so its negatives share a prefix; other links of the same S1 in a batch are masked.

Evaluation on the hold-out fold, per country, ranking every S1 record of the country by cosine:

- never-proposed hold-out links (in no blocking, sibling, name-pair or reverse pair set): top-K recall, split by empty
  S2/S3 address. The ceiling is the recovered share times the hold-out "never proposed" increment of the loss budget
  (US +0.00753, India +0.01091, ``logs/loss_budget_reverse.log``), a linear approximation;
- a random sample of S2/S3 records whose best round probability is below 0.5: top-1 pairs that are not already candidates,
  their true share by cosine gap to rank 2, and the implied new pairs per S1.

Stop criterion: frontier ceiling < +0.0015, or projected gain < +0.0007.

Run from the repository root: ``PYTHONPATH=. python stages/learned_retrieval_probe.py [--links 1500000] [--epochs 2]``.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F

from pipeline import features, records, refine, train
from stages.char_pair_probe import Tower

LENGTH = 96
TAU = 0.05
KS = (1, 3, 10)
SAMPLE = 20000
NEVER = {"US": 0.00753, "India": 0.01091}
TEXT = pl.concat_str([pl.col("core").fill_null(""), pl.lit(" | "), pl.col("words").list.join(" "), pl.lit(" "), pl.col("numbers").list.join(" ")])


def encode(frame: pl.DataFrame) -> np.ndarray:
    """Return the byte matrix (0 is padding) of ``TEXT`` per row of ``frame``."""
    out = np.zeros((frame.height, LENGTH), np.uint8)
    for i, text in enumerate(frame.select(t=TEXT)["t"].to_list()):
        data = np.frombuffer(text.encode()[:LENGTH], np.uint8)
        out[i, :data.size] = data
    return out


def to_device(rows: np.ndarray, device: str) -> torch.Tensor:
    """Shift bytes by one so 0 stays padding and move them to the device."""
    return torch.from_numpy(rows.astype(np.int64) + (rows > 0)).to(device)


def embed(tower: Tower, rows: np.ndarray, device: str, batch: int = 8192) -> torch.Tensor:
    """Unit-norm embeddings of byte rows, moved to the CPU (the pool scan runs there to keep MPS memory small)."""
    tower.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            out.append(F.normalize(tower(to_device(rows[i:i + batch], device)), dim=1).cpu())
    return torch.cat(out)


def top_k(queries: torch.Tensor, pool: torch.Tensor, k: int, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosines of the ``k`` nearest pool rows per query, scanning the pool in chunks."""
    idx_all, sim_all = [], []
    for q in range(0, len(queries), 1024):
        qb = queries[q:q + 1024]
        best_s = torch.full((len(qb), k), -2.0, device=qb.device)
        best_i = torch.zeros((len(qb), k), dtype=torch.long, device=qb.device)
        for c in range(0, len(pool), chunk):
            s, i = (qb @ pool[c:c + chunk].T).topk(k, dim=1)
            merged_s, pick = torch.cat([best_s, s], 1).topk(k, dim=1)
            best_i = torch.cat([best_i, i + c], 1).gather(1, pick)
            best_s = merged_s
        idx_all.append(best_i.cpu().numpy())
        sim_all.append(best_s.cpu().numpy())
    return np.concatenate(idx_all), np.concatenate(sim_all)


def train_tower(links: pl.DataFrame, s1_bytes: np.ndarray, s23_bytes: np.ndarray, args: argparse.Namespace, started: float) -> Tower:
    """Train the shared tower on (S2/S3 row, S1 row) link pairs with in-batch negatives."""
    rng = np.random.default_rng(42)
    tower = Tower(width=128).to(args.device)
    opt = torch.optim.AdamW(tower.parameters(), lr=2e-3, weight_decay=1e-4)
    n = links.height
    per_epoch = n // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * per_epoch)
    r1, r23, s1_id = links["r1"].to_numpy(), links["r23"].to_numpy(), links["s1"].to_numpy()
    prefix_order = links.with_row_index("i").with_columns(u=pl.Series(rng.random(n))).sort("prefix", "u")["i"].to_numpy()
    for epoch in range(args.epochs):
        tower.train()
        random_order = rng.permutation(n)
        hard = prefix_order[: per_epoch * args.batch].reshape(per_epoch, args.batch)[rng.permutation(per_epoch)]
        total = 0.0
        for b in range(per_epoch):
            rows = hard[b] if b % 2 else random_order[b * args.batch:(b + 1) * args.batch]
            qa = F.normalize(tower(to_device(s23_bytes[r23[rows]], args.device)), dim=1)
            sa = F.normalize(tower(to_device(s1_bytes[r1[rows]], args.device)), dim=1)
            logits = qa @ sa.T / TAU
            ids = torch.from_numpy(s1_id[rows].astype(np.int64)).to(args.device)
            same = (ids[:, None] == ids[None, :]) & ~torch.eye(len(rows), dtype=torch.bool, device=args.device)
            logits = logits.masked_fill(same, -1e4)
            target = torch.arange(len(rows), device=args.device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            if b % 500 == 0:
                print(f"phase: epoch {epoch} batch {b}/{per_epoch} loss {total / (b + 1):.4f}, {(time.time() - started) / 60:.1f} min", flush=True)
    return tower


def main() -> None:
    """Train the encoder, then print hold-out top-K recall of never-proposed links and the new-pair load per country."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--links", type=int, default=1_500_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(42)
    cols = ["idx", "country", "core", "words", "numbers"]
    empty = (pl.col("words").list.len() == 0) & (pl.col("numbers").list.len() == 0)
    s23_path = records.records_path("train", "s23")
    s1 = pl.read_parquet(records.records_path("train", "s1"), columns=cols).with_row_index("r1")
    s23 = pl.scan_parquet(s23_path).with_row_index("r23").select("r23", "idx", "country", empty=empty).collect(engine="streaming")
    truth = features.truth_links()
    known = pl.concat([
        pl.scan_parquet("data/candidates/train_pairs.parquet").select("s1", "s23"),
        pl.scan_parquet(f"{features.FEATURE_DIR}train/sibling_*.parquet").select("s1", "s23"),
        *[pl.scan_parquet(str(p)).select("s1", "s23") for name in ("namepair", "reverse") for p in Path(f"{features.FEATURE_DIR}train").glob(f"{name}_*.parquet")],
    ])
    rounds = pl.concat([refine.rounds_scan("train", features.FEATURE_DIR).select("s23", "p"), pl.scan_parquet(f"{features.FEATURE_DIR}train/reverse_*.parquet").select("s23", "p")])
    hold = truth.filter(train.fold() == train.HOLDOUT)
    found = known.join(hold.lazy(), on=["s1", "s23"], how="semi").unique().collect(engine="streaming")
    never = hold.join(found, on=["s1", "s23"], how="anti")
    evals = {}
    for country in ("US", "India"):
        c23 = s23.filter(pl.col("country") == country)
        miss = never.join(c23.select(s23="idx", r23="r23", empty="empty"), on="s23")
        miss = miss.sample(min(SAMPLE, miss.height), seed=1)
        addressed = c23.filter(~pl.col("empty")).select(s23="idx", r23="r23")
        pick = addressed.sample(5 * SAMPLE, seed=2)
        best = rounds.join(pick.select("s23").lazy(), on="s23", how="semi").group_by("s23").agg(best=pl.col("p").max()).collect(engine="streaming")
        gated = pick.join(best, on="s23", how="left").filter(pl.col("best").fill_null(0.0) < 0.5)
        n_gated = gated.height / pick.height * addressed.height
        gated = gated.head(SAMPLE)
        pairs = known.join(gated.select("s23").lazy(), on="s23", how="semi").unique().collect(engine="streaming")
        evals[country] = (miss, gated, n_gated, pairs)
    fit = (truth.filter(train.fold() != train.HOLDOUT).sample(args.links, seed=0)
           .join(s1.select(s1="idx", r1="r1", prefix=pl.col("core").fill_null("").str.slice(0, 4)), on="s1")
           .join(s23.select(s23="idx", r23="r23"), on="s23"))
    del s23, found
    needed = np.unique(np.concatenate([fit["r23"].to_numpy()] + [f["r23"].to_numpy() for m, g, _, _ in evals.values() for f in (m, g)]))
    pos = np.full(int(needed.max()) + 1, -1, np.int64)
    pos[needed] = np.arange(needed.size)
    texts = pl.scan_parquet(s23_path).with_row_index("r23").filter(pl.col("r23").is_in(pl.Series(needed.astype(np.uint32)).implode())).collect(engine="streaming").sort("r23")
    s23_bytes = encode(texts)
    del texts
    fit = fit.with_columns(r23=pl.Series(pos[fit["r23"].to_numpy()]))
    evals = {c: (m.with_columns(r23=pl.Series(pos[m["r23"].to_numpy()])), g.with_columns(r23=pl.Series(pos[g["r23"].to_numpy()])), n, k) for c, (m, g, n, k) in evals.items()}
    s1_bytes = encode(s1)
    gc.collect()
    print(f"phase: fit links {fit.height}, never-proposed hold-out links {never.height}, encoded {needed.size} S2/S3 rows, {(time.time() - started) / 60:.1f} min", flush=True)
    tower = train_tower(fit, s1_bytes, s23_bytes, args, started)
    for country, (miss, gated, n_gated, pairs) in evals.items():
        c1 = s1.filter(pl.col("country") == country)
        pool = embed(tower, s1_bytes[c1["r1"].to_numpy()], args.device)
        c1_idx = c1["idx"].to_numpy()
        idx, _ = top_k(embed(tower, s23_bytes[miss["r23"].to_numpy()], args.device), pool, max(KS))
        rank = np.full(miss.height, 99)
        truth_s1 = miss["s1"].to_numpy()
        for k in range(max(KS)):
            rank = np.where((rank == 99) & (c1_idx[idx[:, k]] == truth_s1), k + 1, rank)
        mask_empty = miss["empty"].to_numpy()
        for label, part in (("all", np.ones(miss.height, bool)), ("empty address", mask_empty), ("with address", ~mask_empty)):
            print(f"{country} never-proposed {label} (n={int(part.sum())}): " + ", ".join(f"top{k} {(rank[part] <= k).mean():.3f}" for k in KS), flush=True)
        for k in KS:
            print(f"{country} ceiling top{k}: {(rank <= k).mean() * NEVER[country]:+.5f} (never-proposed increment {NEVER[country]})", flush=True)
        idx, sim = top_k(embed(tower, s23_bytes[gated["r23"].to_numpy()], args.device), pool, 2)
        top = pl.DataFrame({"s23": gated["s23"], "s1": c1_idx[idx[:, 0]].astype(np.uint32), "cos": sim[:, 0], "gap": sim[:, 0] - sim[:, 1]},
                           schema_overrides={"s23": pl.UInt32})
        top = top.join(pairs, on=["s1", "s23"], how="anti").join(truth.with_columns(t=pl.lit(True)), on=["s1", "s23"], how="left").with_columns(pl.col("t").fill_null(False))
        scale = n_gated / c1.height / SAMPLE
        for cut in (0.0, 0.02, 0.05, 0.1):
            sel = top.filter(pl.col("gap") >= cut)
            print(f"{country} gated top1 gap >= {cut}: new pairs {sel.height} ({sel.height / SAMPLE:.3f} per gated record, {sel.height * scale:.4f} per S1), "
                  f"true share {sel['t'].mean() or 0:.4f}", flush=True)
    print(f"phase: done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
