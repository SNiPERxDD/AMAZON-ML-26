"""Competition metric: macro F0.5 over S1 entities."""

import polars as pl


def macro_f05(predicted: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame) -> float:
    """Return macro F0.5 over every S1 entity.

    ``predicted`` and ``truth`` hold columns ``s1`` and ``s23`` (one row per link);
    ``s1_ids`` holds column ``s1`` listing every S1 entity to score. An entity with
    no true links scores 1 for an empty prediction and 0 otherwise.
    """
    return per_entity(predicted, truth, s1_ids)["f05"].mean()


def per_entity(predicted: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.DataFrame) -> pl.DataFrame:
    """Return ``s1_ids`` with ``n_true``, ``n_pred``, ``tp`` and ``f05`` for each S1 entity."""
    predicted, truth = predicted.select("s1", "s23").unique(), truth.select("s1", "s23").unique()
    n_true = truth.group_by("s1").len().rename({"len": "n_true"})
    n_pred = predicted.group_by("s1").len().rename({"len": "n_pred"})
    hits = predicted.join(truth, on=["s1", "s23"], how="semi").group_by("s1").len().rename({"len": "tp"})
    scores = (
        s1_ids.join(n_true, on="s1", how="left")
        .join(n_pred, on="s1", how="left")
        .join(hits, on="s1", how="left")
        .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0))
        .with_columns(precision=pl.col("tp") / pl.col("n_pred"), recall=pl.col("tp") / pl.col("n_true"))
        .with_columns(
            pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0))
            .then(1.0)
            .when(pl.col("tp") == 0)
            .then(0.0)
            .otherwise(1.25 * pl.col("precision") * pl.col("recall") / (0.25 * pl.col("precision") + pl.col("recall")))
            .alias("f05")
        )
        .drop("precision", "recall")
    )
    return scores
