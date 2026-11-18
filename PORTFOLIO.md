# CTR Prediction with DCN-V2 and Behavior Sequences

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

Ad click-through-rate (CTR) model that combines tabular context with each user's recent behavior sequence: a DCN-V2 (Deep & Cross Network v2) cross/deep tower plus an interchangeable DIN, DIEN, or BST sequence encoder.

**Problem.** The training file has 10,704,179 rows and 119 columns, with 204,179 clicks (1.9%). The model must learn nonlinear interactions among tabular features and represent variable-length histories within a fixed memory budget, and it is judged on how well it ranks the rare clicks.

**Result.** On a stratified 85/15 validation split (seed 42, one run per configuration), DCN-V2 + DIEN reaches ROC-AUC 0.7413 and PR-AUC 0.0792, about four times the 0.0191 PR-AUC of a random ranking. DIN without the cross tower reaches 0.7412 / 0.0779: in these single-seed runs the cross tower adds no measurable gain.

[README](README.md) · [Code](src/) · [Run metadata](results/)

**Model.** CrossNetMix supplies explicit feature crosses through low-rank, gated experts; a parallel MLP learns broader nonlinear patterns. Their outputs join the sequence-interest vector only in the final prediction head. The sequence encoders swap within the same tabular backbone:

- **DIN** (Deep Interest Network) attends from the current candidate to historical item embeddings.
- **DIEN** (Deep Interest Evolution Network) passes the history through a GRU (gated recurrent unit), then attends over evolving interest states.
- **BST** (Behavior Sequence Transformer) adds positional encoding and a Transformer encoder before attention pooling.

**Data and protocol.** Inputs are `clicked` (label), `seq` (behavior sequence), and `ID` (submission identifier). Continuous fields are converted to `float32` and zero-filled; category mappings are fit on the training split only, with an out-of-vocabulary (OOV) index for unseen values. Runs use full-data training, batch size 512, learning rate 0.001, dropout 0.2, three epochs, AdamW, cosine annealing, and ROC-AUC-based early stopping.

## Key decisions

**1. Representing the behavior sequence.**
Options: feed the raw comma-separated values into an LSTM as numbers (the first notebook, `baseline.ipynb`); embed every distinct item ID; or truncate to the most recent 50 items and hash IDs into 262,144 buckets.
Choice: truncation plus hashing, with left padding for fixed-length batches and a guard against `NaN` from attention over an empty sequence.
Why: items are categories, not magnitudes, so they need embeddings; a full-ID vocabulary has no memory bound on a 10.7M-row file, while hashing does.
Cost: hash collisions merge unrelated items, and history older than 50 events is dropped; neither effect was measured.

**2. Handling the 1.9% click rate.**
Options: downsample negatives (the baseline notebook keeps clicks plus twice as many non-clicks, 612,537 rows); or keep all rows and weight positives in the loss.
Choice: full data with `BCEWithLogitsLoss(pos_weight = negative / positive)` (51.4 on the training split).
Why: validation keeps the real click rate, so PR-AUC is read against the true base rate instead of an artificial 33%.
Cost: full-file training limited runs to three epochs, and the weighted loss shifts predicted probabilities, so scores are rankings, not calibrated click probabilities.

**3. Comparing encoders.**
Options: tune each architecture separately; or fix the tabular backbone, split, seed, and training budget and swap only the sequence encoder.
Choice: the fixed-backbone swap, plus one run without the cross tower as a comparison point.
Why: it isolates what each history representation adds on top of the same tabular features.
Cost: one seed per configuration, and the no-cross-tower run does not share the same feature typing (see Limitations).

**4. Choosing the deciding metric.**
Options: report one metric and name a winner; or report ROC-AUC and PR-AUC side by side.
Choice: both, with ROC-AUC for early stopping and PR-AUC as the reference for positive-class ranking.
Why: the best configuration differs by metric, and which matters depends on whether the downstream use is overall ranking or precision among top candidates.
Cost: no single recommended model.

## Results

| Run | Validation ROC-AUC | Validation PR-AUC | Reading |
|---|---:|---:|---|
| Random ranking (reference) | 0.5000 | 0.0191 | equals the click rate |
| DIN without cross tower | 0.7412 | 0.0779 | comparison point without DCN-V2 |
| DCN-V2 + DIN | 0.7402 | 0.0775 | attention-based sequence encoder |
| DCN-V2 + auto-tuned BST | 0.7403 | 0.0783 | Transformer-based sequence encoder |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | highest PR-AUC |
| DCN-V2 + DIEN + segment-interest module (DUSIN, full) | **0.7417** | 0.0780 | highest ROC-AUC |

DIEN is the reference configuration for PR-AUC; the final model choice should follow the operating objective. The no-cross-tower DIN run is within 0.0001 ROC-AUC of DCN-V2 + DIEN.

## Limitations

- Validation metrics only; no leaderboard, external-test, or online A/B result.
- One seed per configuration; ROC-AUC differences of about 0.001 are within run-to-run noise.
- Random stratified split, not time-ordered; performance on later traffic is untested.
- Comparison conditions are not matched: the no-cross-tower DIN run types features differently (111 continuous / 6 categorical vs 85 / 32), so it is a comparison point, not an ablation.

## Next steps

- A matched ablation (same feature typing, with and without the cross tower, several seeds) to settle whether DCN-V2 earns its cost here.
- Time-ordered evaluation, calibration, and Precision/Recall@K to support an advertising decision.
- An experiment manifest with data version, split counts, package versions, and hardware.

## Evidence

- [Training pipeline](src/train.py)
- [Data and sequence handling](src/data.py)
- [DCN-V2, DIN, DIEN, and BST implementation](src/models.py)
- [DIEN metadata](results/dcnv2_dien_meta.json), [DUSIN metadata](results/dcnv2_dien_dusin_full_meta.json), and [no-cross-tower DIN metadata](results/din_meta.json)
- [Row and click counts](notebooks/baseline.ipynb)
- [Run manifest](research/RUN_MANIFEST.md)
