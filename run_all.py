"""Build the final submission from the raw competition data, stage by stage.

Run from the repository root after placing the dataset in ``data/student_resource/dataset/`` (see README.md).
Stages write their intermediates under ``data/`` and their submissions under ``output/probes/<name>/``; the final
submission is ``output/probes/probe_expf_k47/``.

On macOS every heavy command runs under ``tools/run_guarded.sh``, which stops it before it exhausts memory;
``--no-guard`` (and any other platform) runs the commands directly.

Usage: ``python run_all.py [--from STEP] [--only STEP] [--no-guard] [--dry-run]``
with STEP one of ``base``, ``edit``, ``learned``, ``addronly``, ``final``.
"""

import argparse
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = "output/probes"
FOLDS = range(5)
SPLITS = ("train", "test")


def module(name: str, *args: str) -> list[str]:
    return ["-m", name, *args]


def script(name: str, *args: str) -> list[str]:
    return [f"stages/{name}.py", *args]


def five_fold(name: str, *extra: str) -> list[list[str]]:
    """Fit one model per left-out fold, then score train out of fold and test."""
    fits = [module(name, "fit", "--fold", str(k), *extra) for k in FOLDS]
    return fits + [module(name, "predict", "train", "--oof", *extra), module(name, "predict", "test", *extra)]


def base() -> list[list[str]]:
    """Records, blocking, pair classifier, stacked stages, extra candidate rounds, band refits, baseline submission."""
    cmds = [module("pipeline.convert"), module("pipeline.records", "train"), module("pipeline.records", "test"),
            module("pipeline.candidates", "run", "train", "--save", "data/candidates/train_pairs.parquet"),
            module("pipeline.candidates", "run", "test")]
    for split in SPLITS:
        cmds += [module("pipeline.features", "pairs", split, "--chunk-s1", "50000"), module("pipeline.features", "context", split)]
    cmds += five_fold("pipeline.train")
    cmds += [module("pipeline.stack", "peers", split) for split in SPLITS]
    cmds += five_fold("pipeline.stack")
    stage3 = ("--base", "stack", "--name", "stage3")
    cmds += [module("pipeline.stack", "peers", split, *stage3) for split in SPLITS]
    cmds += five_fold("pipeline.stack", *stage3)
    cmds += [module("pipeline.stack", "blend", split, "--inputs", "stack,stage3", "--name", "blend") for split in SPLITS]
    for name in ("pipeline.siblings", "pipeline.namepairs", "pipeline.reverse"):
        for split in SPLITS:
            cmds += [module(name, "pairs", split), module(name, "features", split)]
        cmds += five_fold(name)
    cmds += [module("pipeline.refine", "fit")] + [module("pipeline.refine", "predict", split) for split in SPLITS]
    cmds += [module("pipeline.refine2", "fit", "--threads", "8"), module("pipeline.refine2", "predict", "train"),
             module("pipeline.refine2", "predict", "test", "--threads", "8"),
             module("pipeline.train", "tune", "--pred", "refine2"),
             module("pipeline.submit", "--output", f"{P}/probe_reverse/")]
    return cmds


def edit() -> list[list[str]]:
    """Band edit model with crossed-word logits, then the stricter France rule."""
    return [script("kaggle_edit_prep"), script("band_edit_submit"),
            script("crossed_edit_submit", "--xedit-only", "--tag", "_r", "--workers", "2"),
            script("crossed_words_probe", "--save", "data/kaggle_edit_r2/xedit_holdout_scores.parquet"),
            script("france_rule_submit", "--base", f"{P}/probe_xedit_r/", "--name", "probe_xedit_fr", "--relative", "0.95")]


def learned() -> list[list[str]]:
    """Byte-CNN two-tower retrieval of links that no candidate round proposed."""
    return [script("learned_round_holdout", "--step", "all"),
            script("learned_round_test", "--base", f"{P}/probe_xedit_fr950/", "--out", f"{P}/probe_learned/",
                   "--cut", "0.8", "--countries", "US,India"),
            script("learned_round_test", "--base", f"{P}/probe_learned/", "--out", f"{P}/probe_learned_france/",
                   "--countries", "France")]


def addronly() -> list[list[str]]:
    """Address-only links, the France rescue and no-address name-typo links."""
    k300 = ("--max-key", "300", "--chunk", "500000", "--tag", "_k300")
    model = ("--model", "data/addronly/booster_shape2.txt")
    return [script("addr_only_match", "--split", "train", *k300), script("addr_only_match", "--split", "test", *k300),
            script("addr_only_eval", "--pairs", "data/addronly/train_pairs_k300.parquet", "--save", "data/addronly/booster_shape2.txt"),
            script("addr_only_probe", "--base", f"{P}/probe_learned_france/matching_results.tsv", *model,
                   "--countries", "US,India", "--cut", "0.8", "--name", "probe_addronly2"),
            script("addr_only_probe", "--base", f"{P}/probe_addronly2/matching_results.tsv", *model,
                   "--countries", "France", "--cut", "0.65", "--name", "probe_addronly2_fr065"),
            script("france_rescue", "--probe", "probe_fr_rescue_fr065", "--base", f"{P}/probe_addronly2_fr065/matching_results.tsv",
                   "--noaddr", "--all-nums"),
            script("noaddr_fuzzy", "--eval"),
            script("noaddr_fuzzy", "--probe", "probe_noaddr_fuzzy", "--base", f"{P}/probe_fr_rescue_fr065/matching_results.tsv",
                   "--cut", "0.8")]


def final() -> list[list[str]]:
    """US/India links re-chosen by expected F0.5 at test odds 0.47 of the hold-out's, then the official validator."""
    out = f"{P}/probe_expf_k47"
    return [script("expected_f", "--probe", "probe_expf_k47", "--base", f"{P}/probe_noaddr_fuzzy/matching_results.tsv", "--odds", "0.47"),
            ["data/student_resource/utils/validate_submission.py", "-m", f"{out}/matching_results.tsv",
             "-c", f"{out}/candidate_pairs.tsv", "--test-dir", "data/student_resource/dataset/test", "--check-ids"]]


STEPS = {"base": base, "edit": edit, "learned": learned, "addronly": addronly, "final": final}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", choices=STEPS, default="base", help="first step to run")
    parser.add_argument("--only", choices=STEPS, help="run this step alone")
    parser.add_argument("--no-guard", action="store_true", help="run commands without the memory guard")
    parser.add_argument("--max-gb", default="18", help="memory ceiling of the guard, in GB")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    names = list(STEPS)
    names = [args.only] if args.only else names[names.index(args.start):]
    guard = [] if args.no_guard or platform.system() != "Darwin" else ["tools/run_guarded.sh", "--max-gb", args.max_gb, "--"]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS", "4")}
    for name in names:
        print(f"== {name}", flush=True)
        for cmd in STEPS[name]():
            full = guard + [sys.executable, *cmd]
            print("$", shlex.join(full), flush=True)
            if args.dry_run:
                continue
            start = time.time()
            subprocess.run(full, cwd=ROOT, env=env, check=True)
            print(f"   done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
