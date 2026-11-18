# CTR Prediction with DCN-V2 and Behaviour Sequences

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

Click-through-rate (CTR) prediction needs both tabular context (user, inventory, and time features) and the sequence of what a user recently interacted with. This project pairs a DCN-V2 (Deep & Cross Network v2) cross/deep tower with interchangeable DIN (Deep Interest Network), DIEN (Deep Interest Evolution Network), and BST (Behavior Sequence Transformer) encoders to test both signals together.

On validation, DCN-V2 + DIEN reaches ROC-AUC 0.7413 and PR-AUC (area under the precision-recall curve) 0.0792, about four times the 0.0191 PR-AUC of a random ranking at this click rate. A DUSIN-labelled extension reaches ROC-AUC 0.7417 and PR-AUC 0.0780, so the best configuration depends on the metric. A DIN model without the cross tower reaches 0.7412 / 0.0779; the most useful finding for the next iteration is that the cross tower adds no clear gain in these runs.

## The challenge

Two problems: learning nonlinear interactions among tabular features, and representing variable-length behaviour histories within a fixed memory budget. The training file is large and imbalanced: 10,704,179 rows, 119 columns, and 204,179 clicks (about 1.91%), per the saved output of `notebooks/baseline.ipynb`.

CrossNetMix supplies explicit feature crosses through low-rank, gated experts; a parallel MLP learns broader nonlinear patterns. Their outputs join the sequence-interest vector only in the final prediction head. The sequence encoders swap within the same tabular backbone:

- **DIN** attends from the current candidate to historical item embeddings.
- **DIEN** passes the history through a GRU (gated recurrent unit), then attends over evolving interest states.
- **BST** adds positional encoding and a Transformer encoder before attention pooling.

## Decisions

**1. Representing the behaviour sequence.**
Options: feed the raw comma-separated values into an LSTM as numbers (the first notebook, `baseline.ipynb`); embed every distinct item ID; or truncate to the most recent 50 items and hash IDs into 262,144 buckets.
Choice: truncation plus hashing, with left padding for fixed-length batches and a guard against `NaN` from attention over an empty sequence.
Why: items are categories, not magnitudes, so they need embeddings; a full-ID vocabulary has no memory bound on a 10.7M-row file, while hashing does.
Cost: hash collisions merge unrelated items, and history older than 50 events is dropped; neither effect was measured.

**2. Handling the 1.91% click rate.**
Options: downsample negatives (the baseline notebook keeps clicks plus twice as many non-clicks, 612,537 rows); or keep all rows and weight positives in the loss.
Choice: full data with `BCEWithLogitsLoss(pos_weight = negative / positive)` (51.4 on the training split).
Why: validation keeps the real click rate, so PR-AUC is read against the true base rate instead of an artificial 33%.
Cost: full-file training limited runs to three epochs, and the weighted loss shifts predicted probabilities, so scores are rankings, not calibrated click probabilities.

**3. Comparing encoders fairly.**
Options: tune each architecture separately; or fix the tabular backbone, split, seed, and training budget and swap only the sequence encoder.
Choice: the fixed-backbone swap, plus one run without the cross tower as a comparison point.
Why: it isolates what each history representation adds on top of the same tabular features.
Cost: one seed per configuration (see Limitations).

**4. Choosing the deciding metric.**
Options: report one metric and name a winner; or report ROC-AUC and PR-AUC side by side.
Choice: both, with ROC-AUC for early stopping and PR-AUC as the reference for positive-class ranking.
Why: the best configuration differs by metric, and which matters depends on whether the downstream use is overall ranking or precision among top candidates.
Cost: no single recommended model.

## Data and training protocol

Inputs: `clicked` (label), `seq` (behaviour sequence; bracketed comma lists and numeric strings are both parsed), and `ID` (submission identifier, kept in the submission file). Continuous fields are converted to `float32` and zero-filled. Category mappings are fit on the training split only; unseen validation/test values use an out-of-vocabulary (OOV) index.

The CLI (`src/train.py`) uses seed 42 and a stratified 85/15 train/validation split, `BCEWithLogitsLoss` with `pos_weight = negative / positive`, AdamW, cosine annealing, and ROC-AUC-based early stopping, and writes predictions plus run metadata. The principal saved runs use full-data training (`sample_subset=1.0`), batch size 512, learning rate 0.001, dropout 0.2, and three epochs.

## Validation results

| Saved run | ROC-AUC | PR-AUC | Reading |
|---|---:|---:|---|
| Random ranking (reference) | 0.5 | 0.0191 | equals the click rate |
| DIN without cross tower | 0.7412 | 0.0779 | comparison point without DCN-V2 |
| DCN-V2 + DIN | 0.7402 | 0.0775 | attention-based sequence encoder |
| DCN-V2 + auto-BST | 0.7403 | 0.0783 | Transformer-based sequence encoder |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | highest PR-AUC |
| DCN-V2 + DIEN + DUSIN (full) | **0.7417** | 0.0780 | highest ROC-AUC |

DIEN is the reference configuration for PR-AUC; the final model choice should follow the operating objective. The no-cross-tower DIN run is within 0.0001 ROC-AUC of DCN-V2 + DIEN.

## Limitations

- Validation metrics only; no leaderboard, external-test, or online A/B result.
- Raw Parquet files and data card not included; provenance and collection period are not stated.
- Single seed per configuration; ROC-AUC differences of about 0.001 are within noise. No confidence intervals, calibration, or ranking-at-K business metrics.
- Random stratified split, not time-ordered; performance on later traffic is untested.
- The no-cross-tower DIN run types features differently, so it is a comparison point, not a clean ablation.
- Results come from the notebooks; `src/train.py` refactors that code and was not re-run. Auto-BST and DUSIN exist only in notebooks.
- Several DCN-V2 notebook logs print `Val nan` for validation loss while ROC-AUC/PR-AUC compute normally; cause not traced.
- `main.py` is a standalone LightGBM feature-engineering stub, separate from `src/train.py`.

## Next iteration

A complete experiment manifest (data version, split counts, package versions, hardware, artefact paths). Time-ordered evaluation, repeated seeds, calibration, and Precision/Recall@K to support an advertising decision. A matched ablation (same feature typing, with and without the cross tower, several seeds) to settle whether DCN-V2 earns its cost here.

## Evidence

- [Training pipeline](src/train.py)
- [Data and sequence handling](src/data.py)
- [DCN-V2, DIN, DIEN, and BST implementation](src/models.py)
- [DIEN metadata](results/dcnv2_dien_meta.json), [DUSIN metadata](results/dcnv2_dien_dusin_full_meta.json), and [no-cross-tower DIN metadata](results/din_meta.json)
- [Row and click counts](notebooks/baseline.ipynb) (saved output of the first cells)
- [Run manifest](research/RUN_MANIFEST.md)
