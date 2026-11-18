# CTR Prediction with DCN-V2 and Behaviour Sequences

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

CTR prediction needs both tabular context—user, inventory, and time features—and the sequence of what a user has recently interacted with. This project combines a DCN-V2-style cross/deep tower with interchangeable DIN, DIEN, and BST sequence encoders to test both forms of signal together.

In the retained validation metadata, DCN-V2 + DIEN reaches ROC-AUC 0.7413 and PR-AUC 0.0792, about four times the 0.0191 PR-AUC of a random ranking at this click rate. A DUSIN-labelled extension reaches ROC-AUC 0.7417 and PR-AUC 0.0780. Because the best ROC-AUC and PR-AUC belong to different configurations, the project does not claim one universal winner. A DIN model without the cross tower reaches 0.7412 / 0.0779, which is the most useful finding for the next iteration: in these single-seed runs the cross tower adds no clear gain.

## The challenge

The project addresses two separate problems: learning nonlinear interactions among tabular features and representing variable-length behaviour histories within a fixed memory budget. The training file is large and imbalanced: the saved output of `notebooks/baseline.ipynb` records 10,704,179 rows, 119 columns, and 204,179 clicks (about 1.91%).

CrossNetMix supplies explicit feature crosses through low-rank, gated experts; a parallel MLP learns broader nonlinear patterns. Their outputs are combined with the sequence-interest vector only in the final prediction head. The sequence encoders are deliberately comparable within the same tabular backbone:

- **DIN** attends from the current candidate to historical item embeddings.
- **DIEN** first passes the history through a GRU, then attends over evolving interest states.
- **BST** adds positional encoding and a Transformer encoder before attention pooling.

## Decisions

**1. How to represent the behaviour sequence.**
Options: feed the raw comma-separated values into an LSTM as numbers (the first notebook, `baseline.ipynb`, does this); embed every distinct item ID; or truncate to the most recent 50 items and hash IDs into 262,144 buckets.
Choice: truncation plus hashing, with left padding so batches have a fixed length and a guard so attention over an empty sequence does not produce `NaN`.
Why: items are categories, not magnitudes, so they need embeddings; a full-ID vocabulary has no memory bound on a 10.7M-row file, while hashing does.
Cost: hash collisions merge unrelated items, and history older than 50 events is dropped. Neither effect was measured.

**2. How to handle the 1.91% click rate.**
Options: downsample negatives (the baseline notebook keeps clicks and twice as many non-clicks, 612,537 rows); or keep all rows and weight positives in the loss.
Choice: full data with `BCEWithLogitsLoss(pos_weight = negative / positive)` (51.4 on the training split).
Why: the validation split then keeps the real click rate, so PR-AUC is read against the true base rate instead of an artificial 33%.
Cost: training on the full file limited runs to three epochs, and the weighted loss shifts predicted probabilities, so the scores are rankings, not calibrated click probabilities.

**3. How to compare encoders fairly.**
Options: tune each architecture separately; or hold the tabular backbone, split, seed, and training budget fixed and swap only the sequence encoder.
Choice: the fixed-backbone swap, plus one run without the cross tower as a comparison point.
Why: it isolates what each history representation adds on top of the same tabular features.
Cost: one seed per configuration, so differences of about 0.001 ROC-AUC cannot be separated from noise; the no-cross-tower run also types features differently, so it is not a clean ablation.

**4. Which metric decides.**
Options: report one metric and name a winner; or report ROC-AUC and PR-AUC side by side.
Choice: both, with ROC-AUC for early stopping and PR-AUC as the reference for positive-class ranking.
Why: the best configuration differs by metric, and which matters depends on whether the downstream use is overall ranking or precision among top candidates.
Cost: no single recommended model.

## Data boundary and training protocol

The source Parquet files and data card are not versioned, so provenance and collection period are not stated. The row and click counts above come from a saved notebook output, not from data in the repository. The maintained code establishes the observable interface: `clicked` is the default label, `seq` the behaviour sequence, and `ID` the submission identifier. Continuous fields are converted to `float32` and zero-filled. Category mappings are fit on the training split only, and unseen validation/test values use an OOV index.

The supported CLI uses seed 42 and a stratified 85/15 train/validation split. It applies `BCEWithLogitsLoss` with `pos_weight = negative / positive`, AdamW, cosine annealing, and ROC-AUC-based early stopping. The principal saved runs record full-data training, batch size 512, learning rate 0.001, dropout 0.2, and three epochs.

## Retained validation comparison

| Saved run | ROC-AUC | PR-AUC | Reading of the result |
|---|---:|---:|---|
| Random ranking (reference) | 0.5 | 0.0191 | equals the click rate |
| DIN without cross tower | 0.7412 | 0.0779 | comparison point without DCN-V2 |
| DCN-V2 + DIN | 0.7402 | 0.0775 | attention-based sequence encoder |
| DCN-V2 + auto-BST | 0.7403 | 0.0783 | Transformer-based sequence encoder |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | highest retained PR-AUC |
| DCN-V2 + DIEN + DUSIN (full) | **0.7417** | 0.0780 | highest retained ROC-AUC |

These are saved validation metrics, not external-test or leaderboard results. There are no repeated-seed confidence intervals, calibration results, or ranking-at-K business metrics. DIEN is therefore a useful reference configuration for PR-AUC, while the model choice should still follow the operating objective.

## What the result does not show

- It is not a leaderboard, external-test, or online A/B result.
- The split is random-stratified, not time-ordered, so it does not show performance on later traffic.
- It does not show that the cross tower helps: the no-cross-tower DIN run is within 0.0001 ROC-AUC of the best DCN-V2 + DIEN run.
- Several DCN-V2 notebook logs print `Val nan` for the validation loss while ROC-AUC and PR-AUC are computed normally; the cause was not traced.

## What is implemented—and what is archival

`src/train.py` is the maintained path: it loads files, splits data, builds training-only category maps, trains a selected sequence backbone, and writes predictions plus run metadata. It makes DIN/DIEN/BST comparisons repeatable from the same CLI contract. The retained results were produced in the notebooks; the CLI is a refactor of that code and was not re-run afterwards.

`main.py` is a separate LightGBM feature-engineering stub. The auto-BST and DUSIN experiments are notebook-only extensions and are not equivalent to the maintained CLI path. The documentation keeps this distinction explicit rather than presenting all notebook experiments as one deployed system.

## Next iteration

The immediate improvement is a complete experiment manifest: data version, split counts, package versions, hardware, and artefact paths. Time-ordered evaluation, repeated seeds, calibration, and Precision/Recall@K would turn the current modelling prototype into stronger evidence for an advertising decision workflow. A matched ablation—same feature typing, with and without the cross tower, over several seeds—would settle whether DCN-V2 earns its cost here.

## Evidence

- [Training pipeline](src/train.py)
- [Data and sequence handling](src/data.py)
- [DCN-V2, DIN, DIEN, and BST implementation](src/models.py)
- [DIEN metadata](results/dcnv2_dien_meta.json), [DUSIN metadata](results/dcnv2_dien_dusin_full_meta.json), and [no-cross-tower DIN metadata](results/din_meta.json)
- [Row and click counts](notebooks/baseline.ipynb) (saved output of the first cells)
- [Run manifest](research/RUN_MANIFEST.md)
