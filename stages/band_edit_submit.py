"""Write a test submission whose US and India band pairs are rescored by the edit model of ``band_edit_probe.py``.

The ``edit + grams`` model (the ``refine2`` probability context plus the raw-text edit features and
hashed one-side 3-grams) is fitted on every US and India hold-out band pair: fold 0, whose
``refine2`` probabilities are out of sample, as they are on test. It then scores the test US and
India ``refine2`` pairs with ``p >= LOW``, in chunks. Every other pair, and all of France, keeps
its ``refine2`` probability. The rule is ``submit.TEST_RULES``.

Only ``matching_results.tsv`` is written, to ``output/probes/probe_edit/``. The candidate pairs are
the same ``refine2`` pairs as in the current baseline, so ``candidate_pairs.tsv`` is linked from
``output/probes/probe_reverse/``. Printed: kept links per S1 by country for ``refine2`` and the
edit model.

Run from the repository root: ``PYTHONPATH=. python stages/band_edit_submit.py``.
"""

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import sparse

from pipeline import features, records, refine2, submit, train
from stages import band_edit_probe as probe

D = features.FEATURE_DIR
OUT = "output/probes/probe_edit/"
BASELINE = "output/probes/probe_reverse/"
CHUNK = 1_000_000


def band_matrix(band: pl.DataFrame, s1_raw: pl.DataFrame, s23_raw: pl.DataFrame) -> sparse.csr_matrix:
    """Return the ``edit + grams`` feature matrix of ``band`` pairs, in band order."""
    dense = np.hstack([band.select(*probe.BASE, "us").cast(pl.Float32).to_numpy(),
                       probe.edit_features(band, s1_raw, s23_raw).cast(pl.Float32).to_numpy()])
    return sparse.hstack([sparse.csr_matrix(dense), probe.hashed(band, s1_raw, s23_raw)], format="csr")


def side(split: str, name: str, keep: pl.DataFrame) -> pl.DataFrame:
    """Return :func:`band_edit_probe.raw_side` for ``split`` limited to the ``idx`` values in ``keep``."""
    return probe.raw_side(name, split).join(keep, on="idx", how="semi")


def band_of(split: str, ids: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the ``refine2`` pairs of ``split`` for ``ids`` with their context, and the band (``p >= LOW``) with ``us``."""
    preds = probe.context(train.prob_scan(split, D, refine2.PREFIX).join(ids.lazy(), on="s1", how="semi").collect())
    band = preds.filter(pl.col("p") >= probe.LOW).join(ids, on="s1").with_columns(us=(pl.col("country") == "US").cast(pl.Int8))
    return preds, band


def main() -> None:
    """Fit on the hold-out band, rescore the test band, and write the probe's matching file."""
    names = [*probe.BASE, "us", "name_lev", "name_norm", "name_an_eq", "glyph_eq", "legal_a", "legal_b", "legal_rel", "len_diff", "addr_norm",
             "addr_an_eq", "house_edit", "house_offset", "nums_a_only", "nums_b_only"] + [f"g{i}" for i in range(1 << probe.BITS)]
    scored = pl.scan_parquet(f"{D}train/part_*.parquet").select("s1").unique().filter(train.fold() == train.HOLDOUT).collect()
    country = pl.read_parquet(records.records_path("train", "s1"), columns=["idx", "country"]).rename({"idx": "s1"})
    scored = scored.join(country.filter(pl.col("country").is_in(["US", "India"])), on="s1")
    truth = features.truth_links().join(scored, on="s1", how="semi").select("s1", "s23")
    _, band = band_of("train", scored)
    label = band.join(truth.with_columns(t=pl.lit(1, pl.Int8)), on=["s1", "s23"], how="left", maintain_order="left")["t"].fill_null(0).to_numpy()
    x = band_matrix(band, side("train", "s1", band.select(idx="s1").unique()), side("train", "s23", band.select(idx="s23").unique()))
    booster = lgb.train(probe.PARAMS, lgb.Dataset(x, label, feature_name=names), probe.ROUNDS)
    del x, band
    print("fitted on the hold-out band", flush=True)
    test_country = pl.read_parquet(records.records_path("test", "s1"), columns=["idx", "country"]).rename({"idx": "s1"})
    ids = test_country.filter(pl.col("country").is_in(["US", "India"]))
    preds, band = band_of("test", ids)
    s1_raw, s23_raw = side("test", "s1", band.select(idx="s1").unique()), side("test", "s23", band.select(idx="s23").unique())
    scores = []
    for start in range(0, band.height, CHUNK):
        part = band.slice(start, CHUNK)
        scores.append(booster.predict(band_matrix(part, s1_raw, s23_raw)).astype(np.float32))
        print(f"scored {start + part.height} of {band.height}", flush=True)
    del s1_raw, s23_raw
    new = band.select("s1", "s23", q=pl.Series(np.concatenate(scores)))
    everything = train.prob_scan("test", D, refine2.PREFIX).collect()
    rule = json.loads(Path(train.RULE_PATH).read_text())
    edited = everything.join(new, on=["s1", "s23"], how="left").with_columns(p=pl.coalesce("q", "p")).drop("q")
    for name, frame in (("refine2", everything), ("edit", edited)):
        kept = submit.decide_by_country(train.with_maxima(frame.lazy()).collect(), rule, submit.TEST_RULES)
        per = kept.join(test_country, on="s1").group_by("country").len().join(test_country.group_by("country").len(), on="country", suffix="_s1")
        print(name, " ".join(f"{c}={n / m:.4f}" for c, n, m in per.sort("country").iter_rows()), flush=True)
    del preds, everything
    kept = submit.decide_by_country(train.with_maxima(edited.lazy()).collect(), rule, submit.TEST_RULES)
    s1, s23 = records.load_split("test")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    submit.id_lists(kept, s1.select(s1="idx", source1_entity_id="entity_id"), s23.select(s23="idx", entity_id="entity_id"), "matched_entity_ids").write_csv(
        f"{OUT}matching_results.tsv", separator="\t", quote_style="never")
    if not Path(f"{OUT}candidate_pairs.tsv").exists():
        os.link(f"{BASELINE}candidate_pairs.tsv", f"{OUT}candidate_pairs.tsv")
    print(f"wrote {OUT}: kept links {kept.height}", flush=True)


if __name__ == "__main__":
    main()
