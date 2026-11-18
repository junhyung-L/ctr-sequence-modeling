# CTR Prediction with DCN-V2 and Behaviour-Sequence Encoders

[한국어](README.ko.md) · [Case study](PORTFOLIO.md)

Predicts whether a user clicks an advertisement from 117 tabular features plus
the user's recent behaviour sequence, by attaching interchangeable DIN, DIEN,
or BST sequence encoders to a DCN-V2-style cross/deep network.

**Why it matters.** Clicks are rare: 204,179 of 10,704,179 training rows
(about 1.91%) are positive, so accuracy says little and the useful question is
how well a model ranks the few rows that will be clicked. That ranking decides
which impressions are worth serving.

**Result.** DCN-V2 + DIEN reached validation ROC-AUC 0.7413 and PR-AUC 0.0792
(stratified 85/15 split, seed 42, one run per configuration). The PR-AUC is
about four times the 0.0191 a random ranking would score at this click rate. A
DIN model without the cross tower reached 0.7412 / 0.0779, so in these
single-seed runs the cross tower shows no clear gain; the sequence encoder
carries most of the signal.

These are saved validation metrics, not leaderboard or external-test scores.
The raw data is not included.

## Problem

Given user, inventory, temporal, and historical-sequence features, estimate
the probability of an advertisement click. The training pipeline handles class
imbalance with a weighted binary cross-entropy loss and compares models by
ROC-AUC and PR-AUC on a validation split that keeps the original click rate.

## Analysis flow

```mermaid
flowchart LR
    A[Parquet train/test files<br/>not included] --> B[Stratified train/validation split]
    B --> C[Continuous and categorical features<br/>+ hashed behaviour sequence]
    C --> D[DCN-V2 cross and deep towers]
    C --> E[DIN, DIEN, or BST<br/>sequence encoder]
    D --> F[CTR prediction head]
    E --> F
    F --> G[Validation ROC-AUC and PR-AUC<br/>early stopping]
    G --> H[Submission CSV and run metadata]
```

## Implemented approach

- **Feature handling:** continuous values are cast to `float32`; categorical
  values are mapped from the training split with an out-of-vocabulary index.
- **Sequence handling:** parsed item histories are truncated/padded to 50 items
  and hashed into 262,144 buckets to bound embedding vocabulary size.
- **Model:** `DCN_SEQ_Model` combines CrossNetMix layers, a deep MLP tower, and
  a selectable DIN, DIEN, or BST interest encoder.
- **Training:** a stratified 85/15 train/validation split uses seed 42,
  `BCEWithLogitsLoss(pos_weight=negative/positive)`, AdamW, cosine annealing,
  and AUC-based early stopping. The full training set is used; negatives are
  not downsampled.

The design choices and the alternatives rejected are in
[PORTFOLIO.md](PORTFOLIO.md#decisions).

## Retained experiment results

Saved validation metadata, not external test-set or leaderboard scores. Every
run uses the full training data, the same stratified 85/15 split (seed 42), and
at most three epochs. Each configuration was run once.

| Experiment | Validation ROC-AUC | Validation PR-AUC | Evidence |
|---|---:|---:|---|
| Random ranking (reference) | 0.5 | 0.0191 | click rate in `notebooks/baseline.ipynb` output |
| DIN without cross tower | 0.7412 | 0.0779 | `results/din_meta.json` (`notebooks/CAT+DIN.ipynb`) |
| DCN-V2 + DIN | 0.7402 | 0.0775 | `results/din_dcnv2_meta.json` |
| DCN-V2 + auto-BST | 0.7403 | 0.0783 | `results/dcnv2_auto_bst_meta.json` |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | `results/dcnv2_dien_meta.json` |
| DCN-V2 + DIEN + DUSIN (full) | **0.7417** | 0.0780 | `results/dcnv2_dien_dusin_full_meta.json` |

How to read the table:

- The highest ROC-AUC and the highest PR-AUC occur in different
  configurations, so the repository does not name a single winner.
- The spread between the DCN-V2 variants is about 0.001 ROC-AUC from one run
  each; without repeated seeds this is not separable from run-to-run noise.
- The DIN-without-cross-tower run also types features differently (111
  continuous / 6 categorical versus 85 / 32 in the DCN-V2 runs), so it is a
  comparison point rather than a clean ablation of the cross tower.

## How to run

The raw Parquet files are not included, so the results cannot be reproduced
from this checkout alone. The retained results were produced in the notebooks;
`src/train.py` is a refactor of that path into one CLI and was not re-run after
the refactor (see `research/RUN_MANIFEST.md`).

Check the CLI without data (needs the packages in `requirements.txt`):

```powershell
pip install -r requirements.txt
cd src
python train.py --help
```

Train with compatible source files:

```powershell
cd src
python train.py `
  --train_path ..\train.parquet `
  --test_path ..\test.parquet `
  --output_path ..\submit_dcn_seq.csv `
  --meta_path ..\meta_dcn_seq.json `
  --seq_backbone dien
```

Expected input columns include `clicked` (label), `seq` (behaviour sequence),
and `ID` (submission identifier); other columns are partitioned by the script.
`--seq_backbone` accepts `din`, `dien`, or `bst`. `requirements.txt` lists
packages derived from the imports; the original package versions were not
recorded. Shared defaults are in `src/config.py`.

## Limitations

- No raw data, external test labels, leaderboard result, or trained checkpoint
  is versioned in the repository.
- One seed per configuration; no confidence intervals, calibration, or
  Precision/Recall@K are recorded.
- The validation split is stratified at random, not time-ordered, so it does
  not test how well the model holds up on later traffic.
- The CLI implements DIN/DIEN/BST selection; the auto-BST tuning and DUSIN runs
  exist only in notebooks.
- Several DCN-V2 notebook logs print `Val nan` for the validation loss while
  ROC-AUC and PR-AUC are computed normally; the cause was not traced.
- `main.py` is a standalone LightGBM feature-engineering stub and is not the
  same path as `src/train.py`.

## Repository layout

```text
.
├── src/
│   ├── data.py                 # Dataset, sequence parsing, and hashing
│   ├── models.py               # DCN-V2-style and sequence model components
│   └── train.py                # CLI training, validation, and output metadata
├── notebooks/                  # Experiments with saved outputs (source of results/)
├── results/                    # Retained JSON run metadata
├── prior-research/             # Background material
└── main.py                     # Standalone LightGBM feature-engineering stub
```

## Documentation

- [Portfolio case study](PORTFOLIO.md)
- [Project review](docs/PROJECT_REVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [CV bullets](docs/CV_BULLETS.md)
- [Run manifest](research/RUN_MANIFEST.md)
