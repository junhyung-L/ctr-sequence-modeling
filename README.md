# CTR Prediction with DCN-V2 and Behaviour-Sequence Encoders

[한국어](README.ko.md) · [Case study](PORTFOLIO.md)

Predicts ad click-through (CTR) from 117 tabular features plus each user's
recent behaviour sequence. A DCN-V2 (Deep & Cross Network v2) cross/deep
network is paired with interchangeable sequence encoders: DIN (Deep Interest
Network), DIEN (Deep Interest Evolution Network), or BST (Behavior Sequence
Transformer).

**Why it matters.** Only 204,179 of 10,704,179 training rows (about 1.91%) are
clicks, so accuracy says little. What counts is how well a model ranks the few
rows that will be clicked, because that ranking decides which impressions to
serve.

**Result.** DCN-V2 + DIEN reaches validation ROC-AUC 0.7413 and PR-AUC (area
under the precision-recall curve) 0.0792, about four times the 0.0191 of a
random ranking (stratified 85/15 split, seed 42, one run per configuration).
DIN without the cross tower reaches 0.7412 / 0.0779: the sequence encoder
carries most of the signal.

## Approach

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

- **Features:** casts continuous values to `float32`; maps categorical values
  from the training split, with an out-of-vocabulary (OOV) index for unseen
  values.
- **Sequences:** truncates/pads item histories to 50 items and hashes them into
  262,144 buckets to bound the embedding vocabulary.
- **Model:** `DCN_SEQ_Model` combines CrossNetMix layers, a deep MLP tower, and
  a selectable DIN, DIEN, or BST interest encoder.
- **Training:** stratified 85/15 train/validation split (seed 42) on the full
  training set without negative downsampling;
  `BCEWithLogitsLoss(pos_weight=negative/positive)`, AdamW, cosine annealing,
  and AUC-based early stopping. The validation split keeps the original click
  rate.

Design choices and rejected alternatives: [PORTFOLIO.md](PORTFOLIO.md#decisions).

## Results

Validation metrics from saved run metadata. Every run uses the full training
data, the same stratified 85/15 split (seed 42), and at most three epochs.

| Experiment | Validation ROC-AUC | Validation PR-AUC | Evidence |
|---|---:|---:|---|
| Random ranking (reference) | 0.5 | 0.0191 | click rate in `notebooks/baseline.ipynb` output |
| DIN without cross tower | 0.7412 | 0.0779 | `results/din_meta.json` (`notebooks/CAT+DIN.ipynb`) |
| DCN-V2 + DIN | 0.7402 | 0.0775 | `results/din_dcnv2_meta.json` |
| DCN-V2 + auto-BST | 0.7403 | 0.0783 | `results/dcnv2_auto_bst_meta.json` |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | `results/dcnv2_dien_meta.json` |
| DCN-V2 + DIEN + DUSIN (full) | **0.7417** | 0.0780 | `results/dcnv2_dien_dusin_full_meta.json` |

The highest ROC-AUC and the highest PR-AUC come from different configurations,
so the choice depends on the operating metric.

## How to run

Results were produced in the notebooks; `src/train.py` consolidates that
pipeline into one CLI.

Check the CLI without data:

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

Inputs need `clicked` (label), `seq` (behaviour sequence), and `ID`
(submission identifier); the script partitions the other columns.
`--seq_backbone` accepts `din`, `dien`, or `bst`. Shared defaults are in
`src/config.py`.

## Limitations

- Validation metrics only; no leaderboard, external-test, or online A/B result.
- Raw data and trained checkpoints not included, so results cannot be reproduced from this checkout alone.
- Single seed per configuration; ROC-AUC differences of about 0.001 are within run-to-run noise. No confidence intervals, calibration, or Precision/Recall@K.
- Random stratified split, not time-ordered; performance on later traffic is untested.
- The no-cross-tower DIN run types features differently (111 continuous / 6 categorical vs 85 / 32), so it is a comparison point, not a clean ablation.
- `src/train.py` was not re-run after the refactor; auto-BST tuning and DUSIN runs exist only in notebooks.
- `requirements.txt` is derived from imports; original package versions are unpinned.
- Several DCN-V2 notebook logs print `Val nan` for validation loss while ROC-AUC/PR-AUC compute normally; cause not traced.
- `main.py` is a standalone LightGBM feature-engineering stub, separate from `src/train.py`.

## Repository layout

```text
.
├── src/
│   ├── data.py                 # Dataset, sequence parsing, and hashing
│   ├── models.py               # DCN-V2-style and sequence model components
│   └── train.py                # CLI training, validation, and output metadata
├── notebooks/                  # Experiments with saved outputs (source of results/)
├── results/                    # Retained JSON run metadata
└── prior-research/             # Background material
```

## Documentation

- [Portfolio case study](PORTFOLIO.md)
- [Project review](docs/PROJECT_REVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [CV bullets](docs/CV_BULLETS.md)
- [Run manifest](research/RUN_MANIFEST.md)
