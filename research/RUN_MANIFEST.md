# Run Manifest

## Supported entry point

The maintained experiment path is `src/train.py`. Run it from the repository
root with either form:

```powershell
python src\train.py --train_path .\train.parquet --test_path .\test.parquet
# or
python -m src.train --train_path .\train.parquet --test_path .\test.parquet
```

The first form requires explicit input paths because the legacy defaults were
written for execution from `src/`. The defaults themselves have not changed.

## Result-preservation boundary

This refactor centralizes command-line defaults and import paths only. It keeps
the prior default seed (42), split fraction (0.15), architecture defaults,
optimizer, scheduler, loss, early-stopping rule, and JSON output schema.

Exact numerical reproduction still requires the original Parquet inputs, which
are not stored in this repository. The notebooks ran on Python 3.12.11 (notebook
metadata); their package versions were not recorded. `requirements.txt` pins the
2026-11 review environment (Python 3.13.2).

## Smoke test (2026-11 review)

`python scripts\smoke_test.py` writes tiny synthetic train/test Parquet files to
a temporary directory and runs `src/train.py` for one epoch with each backbone
(`din`, `dien`, `bst`) on CPU. It passed for all three. It checks that the CLI
executes and writes the submission and metadata files; its metrics are
meaningless.

## Validation-loss NaN fix (2026-11 review)

The notebook logs print `Val nan` while validation AUC stays finite. Two causes
were found and fixed in the CLI:

1. **BST, fully padded sequences.** In eval mode the PyTorch Transformer fast
   path returns NaN for a row whose sequence is all padding (empty history).
   The attention weight for that row is 0, but `NaN * 0` is still NaN, so the
   logit was NaN. Reproduced on CPU with the smoke test (`Val Loss: nan` for
   `bst` only). `BSTBackbone` now zeroes padded encoder outputs.
2. **fp16 autocast at evaluation (CUDA runs, all backbones).** Validation and
   test forward passes ran under fp16 autocast. In eval mode BatchNorm uses
   running statistics, so a raw-scale continuous value far from the running
   mean can exceed the fp16 range and give inf/NaN logits; with
   `BCEWithLogitsLoss`, an infinite logit gives a NaN loss. This was shown with a
   synthetic CPU fp16 autocast check, not on the original data (no GPU or data
   available). Validation and test now run in float32, and the number of
   non-finite validation logits is printed if any remain.

The notebook metrics were computed after `nan_to_num` mapped non-finite
probabilities to 0 or 1, so a rerun with the fixed CLI may differ slightly from
the reported numbers. The reported numbers were not changed.
