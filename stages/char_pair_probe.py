"""Character CNN pair model on the contested hold-out band pairs, stacked on a base score.

Both sides of a pair are encoded as the UTF-8 bytes of ``name | address`` (folded
raw text, at most ``LENGTH`` bytes). A shared tower (byte embedding, three residual 1-D convolutions, masked max and mean
pooling) maps each side to a vector; the head adds a correction (initialised at zero) to the base logit from the absolute difference and product of the two vectors plus the base
score's logit, the ``refine2`` logit and the US flag. The model is cross-fitted over the two S1-hash halves of the hold-out
band (``data/kaggle_edit_r2/r2_band.parquet``) and trained only on contested pairs, where the base score lies in
``[--low, --high]``; other pairs keep the base score.

The base score is ``refine2`` p, or a cross-fitted edit-model column from ``--scores`` (for example ``r2_scores.parquet``
of ``kaggle_edit_full.py`` with ``--column "r2 800"``). Stop criterion: less than +0.0005 weighted over the base when the base
is the full-data crossed edit model, or not reproduced on a second split.

Run from the repository root: ``PYTHONPATH=. python stages/char_pair_probe.py [--scores FILE --column NAME] [--sample 20]``.
"""

import argparse
import time

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from stages.kaggle_edit_full import report

ROOT = "data/kaggle_edit_r2"
SOURCE = "data/kaggle_edit"
LENGTH = 112


def encode(raw: pl.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    """Return a byte matrix (0 is padding) of ``name | addr`` per row of ``raw`` and the row of each ``idx``."""
    texts = raw.select(t=pl.concat_str(pl.col("name").fill_null(""), pl.lit(" | "), pl.col("addr").fill_null("")))["t"].to_list()
    out = np.zeros((len(texts), LENGTH), np.uint8)
    for i, text in enumerate(texts):
        data = np.frombuffer(text.encode()[:LENGTH], np.uint8)
        out[i, :data.size] = data
    rows = {int(k): i for i, k in enumerate(raw["idx"].to_list())}
    return out, rows


class Tower(nn.Module):
    """Byte CNN encoder of one record."""

    def __init__(self, dim: int = 64, hidden: int = 192, width: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(257, dim, padding_idx=0)
        self.c1 = nn.Conv1d(dim, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.c3 = nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2)
        self.out = nn.Linear(2 * hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of byte rows (already shifted by one, 0 = padding)."""
        mask = (x > 0).unsqueeze(1).float()
        z = F.relu(self.c1(self.embed(x).transpose(1, 2)))
        z = F.relu(self.c2(z)) + z
        z = F.relu(self.c3(z)) + z
        top = (z - 1e4 * (1 - mask)).amax(2)
        mean = (z * mask).sum(2) / mask.sum(2).clamp(min=1)
        return self.out(torch.cat([top, mean], 1))


class Pair(nn.Module):
    """Siamese pair classifier with scalar side inputs."""

    def __init__(self, extra: int, width: int = 128) -> None:
        super().__init__()
        self.tower = Tower(width=width)
        self.head = nn.Sequential(nn.Linear(2 * width + extra, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """Return the pair logit: the base logit (first extra column) plus a correction that starts at zero."""
        ta, tb = self.tower(a), self.tower(b)
        return extra[:, 0] + self.head(torch.cat([(ta - tb).abs(), ta * tb, extra], 1)).squeeze(1)


def logit(p: np.ndarray) -> np.ndarray:
    """Clipped logit."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def batches(n: int, size: int, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    """Index batches over ``n`` rows."""
    order = rng.permutation(n) if shuffle else np.arange(n)
    return [order[i:i + size] for i in range(0, n, size)]


def tensors(rows: np.ndarray, sa: np.ndarray, sb: np.ndarray, extra: np.ndarray, device: str) -> tuple[torch.Tensor, ...]:
    """Move one batch to the device, shifting bytes by one so 0 stays padding."""
    a = torch.from_numpy(sa[rows].astype(np.int64) + (sa[rows] > 0)).to(device)
    b = torch.from_numpy(sb[rows].astype(np.int64) + (sb[rows] > 0)).to(device)
    return a, b, torch.from_numpy(extra[rows]).to(device)


def main() -> None:
    """Cross-fit the pair model over the two halves and report F0.5 of the base and the stacked score."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", help="parquet with s1, s23 and a cross-fitted base score column")
    parser.add_argument("--column", default="r2 800")
    parser.add_argument("--low", type=float, default=0.02)
    parser.add_argument("--high", type=float, default=0.98)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--sample", type=int, default=1, help="keep S1 entities with hash % N == 0")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    band = pl.read_parquet(f"{ROOT}/r2_band.parquet")
    ids = pl.read_parquet(f"{SOURCE}/holdout_ids.parquet")
    if args.sample > 1:
        keep = pl.col("s1").hash(3) % args.sample == 0
        band, ids = band.filter(keep), ids.filter(keep)
    truth = pl.read_parquet(f"{SOURCE}/truth.parquet").join(ids.select("s1"), on="s1", how="semi")
    low = pl.read_parquet(f"{SOURCE}/r2_low.parquet").join(ids.select("s1"), on="s1", how="semi")
    base = band["p"].to_numpy().astype(np.float32)
    if args.scores:
        base = band.select("s1", "s23").join(pl.read_parquet(args.scores).select("s1", "s23", b=args.column), on=["s1", "s23"], how="left",
                                             maintain_order="left")["b"].fill_null(pl.Series(base)).to_numpy().astype(np.float32)
    raws = {}
    for side, key in (("s1", "s1"), ("s23", "s23")):
        raw = pl.read_parquet(f"{ROOT}/train_{side}_raw.parquet", columns=["idx", "name", "addr"]).join(band.select(idx=key).unique(), on="idx", how="semi")
        raws[side] = encode(raw)
    sa = raws["s1"][0][np.array([raws["s1"][1][int(k)] for k in band["s1"].to_list()])]
    sb = raws["s23"][0][np.array([raws["s23"][1][int(k)] for k in band["s23"].to_list()])]
    extra = np.column_stack([logit(base), logit(band["p"].to_numpy()), band["us"].to_numpy().astype(np.float32)]).astype(np.float32)
    y, half = band["t"].to_numpy().astype(np.float32), band["half"].to_numpy()
    contested = (base >= args.low) & (base <= args.high)
    print(f"phase: encoded {band.height} pairs, contested {int(contested.sum())} (true {y[contested].mean():.3f}), {(time.time() - started) / 60:.1f} min", flush=True)
    stacked = base.copy()
    for k in (0, 1):
        fit_rows, score_rows = np.flatnonzero(contested & (half != k)), np.flatnonzero(contested & (half == k))
        model = Pair(extra.shape[1]).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = args.epochs * ((fit_rows.size + args.batch - 1) // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        yt = torch.from_numpy(y)
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for rows in batches(fit_rows.size, args.batch, True, rng):
                a, b, e = tensors(fit_rows[rows], sa, sb, extra, args.device)
                loss = F.binary_cross_entropy_with_logits(model(a, b, e), yt[fit_rows[rows]].to(args.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss) * rows.size
            print(f"phase: half {k} epoch {epoch} loss {total / fit_rows.size:.4f}, {(time.time() - started) / 60:.1f} min", flush=True)
        model.eval()
        out = np.zeros(score_rows.size, np.float32)
        with torch.no_grad():
            for rows in batches(score_rows.size, 4 * args.batch, False, rng):
                out[rows] = torch.sigmoid(model(*tensors(score_rows[rows], sa, sb, extra, args.device))).cpu().numpy()
        stacked[score_rows] = out
        del model, opt
    everything = np.ones(band.height, bool)
    report("base", band, everything, base, low, ids, truth)
    report("char pair stacked", band, everything, stacked, low, ids, truth)
    print(f"phase: done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
