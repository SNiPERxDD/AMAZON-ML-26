"""Convert the competition TSV files to the Parquet cache that every later step reads.

Run from the repository root: ``python -m pipeline.convert [--dataset-dir DIR]``. It reads
every ``DIR/*/*.tsv`` (``train/`` and ``test/``) and writes ``data/parquet/{stem}.parquet``.
Every column is read as a string, with quoting disabled, so IDs and free text are kept
exactly as written. Empty fields are read as null.
"""

import argparse
from pathlib import Path

import polars as pl

from pipeline import records

DATASET_DIR = "data/student_resource/dataset"


def convert(dataset_dir: str) -> None:
    """Write one Parquet file per TSV file under ``dataset_dir`` and print its row count."""
    sources = sorted(Path(dataset_dir).glob("*/*.tsv"))
    if not sources:
        raise SystemExit(f"no TSV files found under {dataset_dir}/*/")
    Path(records.PARQUET_DIR).mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = Path(records.PARQUET_DIR) / f"{source.stem}.parquet"
        pl.scan_csv(source, separator="\t", quote_char=None, infer_schema=False).sink_parquet(target)
        rows = pl.scan_parquet(target).select(pl.len()).collect().item()
        print(f"{source.name}: {rows} rows -> {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="folder holding train/ and test/ TSV files")
    convert(parser.parse_args().dataset_dir)
