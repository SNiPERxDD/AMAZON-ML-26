# Business Entity Resolution: Amazon ML Challenge 2026

A cascade of blocking, gradient-boosted pair classifiers and targeted retrieval rounds. It links noisy business
records from two secondary sources to a clean reference list.

| | |
|---|---|
| Public leaderboard (macro F0.5) | **0.980553** |
| Final rank | ~1,100 / 10,600 teams (about 90,000 participants in teams of 3–4) |
| Winning score | 0.992082 |

## Contents

- [Problem](#problem)
- [Approach](#approach)
- [Results](#results)
- [Reproducing](#reproducing)
- [Repository layout](#repository-layout)
- [License](#license)

## Problem

- **Source 1 (S1)** is a clean, deduplicated list of businesses, each with a name and an address.
- **Sources 2 and 3 (S2/S3)** hold records of the same businesses from other data providers:
  - they contain abbreviations, typos, transliterations, and missing or partial addresses;
  - they also contain look-alike decoy records.
- For every S1 entity, the task is to list all S2/S3 records that describe the same business. An entity may have
  many matches, or none.

**Metric.** Scores are macro F0.5 over S1 entities, which weights precision twice as heavily as recall. An entity
with no predicted links and no true links scores 1.

**Data.** The training set covers the US and India, with labels. The test set covers the US, India and France;
France has no labelled training data.

**Deliverables.** Two files: `matching_results.tsv`, which is scored, and `candidate_pairs.tsv`, the output of
blocking.

## Approach

```
records ─► blocking ─► pair classifier ─► stacked stages ─► extra candidate rounds ─► band refits
                                                                                          │
   expected-F0.5 link count ◄─ address / typo rounds ◄─ learned retrieval ◄─ band edit model
```

Each stage reads the previous submission and adds, removes or re-scores links.

- **Validation.** Model choices were measured on a fixed hold-out: a random 20% of training S1 entities
  (`s1.hash(seed=3) % 5 == 0`).
- **Test rules.** Leaderboard submissions were used to calibrate the decision rules for test, whose class balance
  differs from the training data.

| # | Stage | Code | What it does |
|---|---|---|---|
| 1 | Records and blocking | `pipeline/records.py`, `normalize.py`, `candidates.py` | Normalises names and addresses into tokens, compact names, house numbers and street tokens. Learns a transliteration and phrase table from training links. Eleven blocking families (name cores, house numbers, streets, rare tokens, prefixes) propose candidate pairs under per-entity caps. |
| 2 | Pair classifier | `pipeline/features.py`, `train.py` | A LightGBM model on string-similarity, token-overlap and number-agreement features, cross-fitted over five folds. |
| 3 | Stacking | `pipeline/stack.py` | Two stacked LightGBM stages add context: how the other candidates of the same S1 and of the same S2/S3 record scored, and agreement with the S1's confident links. |
| 4 | Extra candidate rounds | `pipeline/siblings.py`, `namepairs.py`, `reverse.py` | Proposes pairs that blocking missed, drawn from the neighbourhood of confident links: sibling records, name pairs, and reverse S2/S3 → S1 retrieval. |
| 5 | Band refits | `pipeline/refine.py`, `refine2.py`, `submit.py` | Two refits re-score the uncertain probability band using per-entity context. Links are kept when the probability is at least *t* and at least *r* × the S1's best. |
| 6 | Band edit model | `stages/band_edit_*.py`, `crossed_words_probe.py`, `crossed_edit_submit.py` | Re-scores the US/India band from character-edit and hashed word features. It adds out-of-fold logits of linear models over crossed word pairs, such as an abbreviation against the full word. |
| 7 | France rule | `stages/france_rule_submit.py` | France has no labels, so it keeps the refit probabilities under a stricter rule (0.7 / 0.95). |
| 8 | Learned retrieval | `stages/learned_round_holdout.py`, `learned_round_test.py` | A byte-level CNN two-tower encoder retrieves S1 neighbours for records that no earlier round linked. A cross-fitted LightGBM accepts pairs scoring 0.8 or above. |
| 9 | Address-only round | `stages/addr_only_*.py` | Links records whose address matches an S1 but whose name differs. It blocks on address keys and scores with name and address shape features. |
| 10 | France rescue and typo links | `stages/france_rescue.py`, `noaddr_fuzzy.py` | Restores France pairs whose name cores and address numbers agree after normalising accents, legal forms and street abbreviations. Adds name-typo matches for records without an address. |
| 11 | Expected-F0.5 link count | `stages/expected_f.py` | For each US/India entity, keeps the number of top links that maximises Monte Carlo expected F0.5 under calibrated probabilities. Probabilities are first adjusted to test odds (0.47 × the hold-out's), estimated from earlier leaderboard results. |

`stages/` also contains helper modules imported by these stages, such as feature builders and the byte-CNN tower.

## Results

Public leaderboard scores as the stages were added. Each row builds on the one above.

| Submission | Change | Macro F0.5 |
|---|---|---|
| `probe_reverse` | Stages 1–5 | 0.970867 |
| `probe_edit` | Band edit model | 0.977646 |
| `probe_xedit` | Crossed-word logits | 0.978181 |
| `probe_xedit_fr950` | France rule 0.7 / 0.95 | 0.978393 |
| `probe_learned` | Learned retrieval, US/India | 0.979764 |
| `probe_learned_france` | Learned retrieval, France | 0.979830 |
| `probe_addronly2` | Address-only round, US/India | 0.980043 |
| `probe_addronly2_france` | Address-only round, France | 0.980181 |
| `probe_expf_k47` | France cut 0.65, France rescue, typo links, expected-F0.5 | **0.980553** |

- **Final row.** It was submitted as a single upload, so its four changes were not scored separately.
- **Scored lower and not used:**
  - odds 0.3 instead of 0.47, which scored 0.980422;
  - an extra no-address round plus address-typo links, which scored 0.980324.

## Reproducing

### Data

The competition data is not included. Place the competition kit so that these paths exist:

```
data/student_resource/dataset/train/
data/student_resource/dataset/test/
data/student_resource/utils/validate_submission.py
```

### Environment

Python 3.12:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`polars` is pinned exactly: the hold-out fold is defined with `polars` hashing, which is not stable across versions.

### Run

From the repository root:

```sh
python run_all.py                  # full cascade
python run_all.py --from addronly  # resume from a step: base, edit, learned, addronly, final
python run_all.py --only final     # run one step
python run_all.py --dry-run        # print the commands without running them
```

- The final submission is written to `output/probes/probe_expf_k47/` and checked with the official validator.
- Intermediate files go to `data/`.
- On macOS each command runs under `tools/run_guarded.sh`, which stops a process tree before it exhausts memory
  (`--max-gb`, default 18). On other platforms, or with `--no-guard`, commands run directly.

### Hardware

- Development used a 24 GB Apple-silicon machine. The largest measured memory peak was about 13 GB, during
  test scoring of the band edit model.
- A full run takes hours, most of it in stages 1–5 (not timed end to end). The retrieval tower trains on MPS when available.

### Determinism

- Multi-threaded LightGBM training and process-pool feature building are not bit-reproducible. Re-running the
  band edit stage changed about 0.9% of the output rows.
- The retrieval tower is seeded, but GPU and MPS kernels are not guaranteed to be deterministic.
- Stages 10 and 11 are deterministic. Rebuilt from the same intermediate files, they reproduce the submitted files
  exactly.

## Repository layout

```
pipeline/     records, blocking, pair features, classifiers, stacking, refits, submission writer
stages/       the later rounds of the cascade and their helper modules
tools/        run_guarded.sh (memory guard)
run_all.py    ordered commands for the full cascade
```

## License

[MIT](LICENSE)
