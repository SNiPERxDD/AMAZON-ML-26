"""Write the inputs of the Kaggle band edit run (``kaggle_edit_full.py``) to ``data/kaggle_edit/``.

An earlier probe found the band edit model (H+words) gaining when the training-fold band pairs are added to
the hold-out band (share 0.3: +0.00035 US, +0.00056 India). Fitting on every training-fold pair does not fit in local
memory, so the fit runs on a Kaggle host. Everything that depends on polars ``hash`` (folds, halves) or on pipeline
intermediates is computed here, so the remote run needs no pipeline state:

- ``q_band``: hold-out and training-fold pairs with refit ``q >= LOW`` (hold-out ``refine``, training folds out of
  fold), with the ``q`` context, country, fold, half and label;
- ``q_low``: hold-out pairs below ``LOW`` (they enter the evaluation unchanged);
- ``r2_band`` / ``r2_low``: the hold-out band on ``refine2`` context, the current ``probe_edit`` setup, as the reference;
- ``test_band``: test US and India pairs with test refit ``q >= LOW`` and their context;
- ``holdout_ids`` / ``truth``: hold-out US and India S1 entities and their train links;
- raw sides (folded name and address, house, numbers) of the records in those bands.

Run from the repository root: ``PYTHONPATH=. python stages/kaggle_edit_prep.py``.
"""

from pathlib import Path

import polars as pl

from pipeline import features, records, refine, refine2, stack, train
from stages.band_edit_probe import LOW, context, raw_side

D = features.FEATURE_DIR
OUT = Path("data/kaggle_edit")
KEYS = ["s1", "s23", "p", "ratio", "p_s1_max", "p_s23_max", "rank_s1", "rank_s23", "n50_s1", "n50_s23"]


def labelled(band: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Add the label ``t``, the S1-hash ``half`` and ``us`` to ``band``."""
    return band.join(truth.with_columns(t=pl.lit(1, pl.Int8)), on=["s1", "s23"], how="left").with_columns(
        pl.col("t").fill_null(0), half=(pl.col("s1").hash(7) % 2).cast(pl.Int8), us=(pl.col("country") == "US").cast(pl.Int8))


def main() -> None:
    """Write the band tables, ids, truth and raw sides."""
    OUT.mkdir(parents=True, exist_ok=True)
    country = pl.read_parquet(records.records_path("train", "s1"), columns=["idx", "country"]).rename({"idx": "s1"}).filter(
        pl.col("country").is_in(["US", "India"]))
    hold_ids = pl.scan_parquet(f"{D}train/part_*.parquet").select("s1").unique().filter(train.fold() == train.HOLDOUT).collect().join(
        country, on="s1")
    truth = features.truth_links().select("s1", "s23")
    hold_ids.write_parquet(OUT / "holdout_ids.parquet")
    truth.join(hold_ids, on="s1", how="semi").write_parquet(OUT / "truth.parquet")

    r2 = context(pl.read_parquet(stack.prob_path("train", 0, D, refine2.PREFIX)).select("s1", "s23", "p").join(hold_ids, on="s1", how="semi"))
    r2.filter(pl.col("p") < LOW).select("s1", "s23", "p").write_parquet(OUT / "r2_low.parquet")
    r2_band = labelled(r2.filter(pl.col("p") >= LOW).select(KEYS).join(hold_ids, on="s1"), truth)
    r2_band.write_parquet(OUT / "r2_band.parquet")
    print(f"r2 band {r2_band.height}", flush=True)
    del r2

    q = pl.concat([pl.read_parquet(stack.prob_path("train", 0, D, refine.PREFIX)).select("s1", "s23", "p"),
                   pl.read_parquet(refine2.oof_path(D)).select("s1", "s23", p="q")])
    preds = context(q).select(KEYS).join(country, on="s1").with_columns(fold=train.fold().cast(pl.Int8))
    del q
    preds.filter((pl.col("p") < LOW) & (pl.col("fold") == train.HOLDOUT)).select("s1", "s23", "p").write_parquet(OUT / "q_low.parquet")
    q_band = labelled(preds.filter(pl.col("p") >= LOW), truth)
    del preds
    q_band.write_parquet(OUT / "q_band.parquet")
    print(f"q band {q_band.height} (hold-out {(q_band['fold'] == train.HOLDOUT).sum()}), true share {q_band['t'].mean():.3f}", flush=True)

    test_country = pl.read_parquet(records.records_path("test", "s1"), columns=["idx", "country"]).rename({"idx": "s1"}).filter(
        pl.col("country").is_in(["US", "India"]))
    test = context(pl.read_parquet(stack.prob_path("test", 0, D, refine.PREFIX)).select("s1", "s23", "p").join(test_country, on="s1", how="semi"))
    test_band = test.filter(pl.col("p") >= LOW).select(KEYS).join(test_country, on="s1").with_columns(us=(pl.col("country") == "US").cast(pl.Int8))
    del test
    test_band.write_parquet(OUT / "test_band.parquet")
    print(f"test band {test_band.height}", flush=True)

    s23_train = pl.concat([r2_band.select(idx="s23"), q_band.select(idx="s23")]).unique()
    for split, s23 in (("train", s23_train), ("test", test_band.select(idx="s23").unique())):
        raw_side("s1", split).write_parquet(OUT / f"{split}_s1_raw.parquet")
        raw_side("s23", split).join(s23, on="idx", how="semi").write_parquet(OUT / f"{split}_s23_raw.parquet")
        print(f"{split} raw sides written", flush=True)


if __name__ == "__main__":
    main()
