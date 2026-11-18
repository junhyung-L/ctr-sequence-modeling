"""End-to-end smoke test for ``src/train.py`` on tiny synthetic data.

Builds small train/test Parquet files with the column layout the CLI expects
(``clicked`` label, ``seq`` behaviour string, ``ID``, fixed categorical columns,
and ``l_feat_*`` / numeric columns), then runs one epoch for the requested
sequence backbones. It checks that the CLI executes and writes its outputs; the
metrics it prints are meaningless and must not be reported as results.

Usage (from the repository root)::

    python scripts/smoke_test.py                 # din, dien, bst; one epoch each
    python scripts/smoke_test.py --backbones dien
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"


def make_frame(n_rows: int, rng: np.random.Generator, with_label: bool) -> pd.DataFrame:
    """Create a synthetic frame with the column groups that train.py partitions."""
    frame = pd.DataFrame(
        {
            "ID": [f"ID_{i:06d}" for i in range(n_rows)],
            "gender": rng.integers(1, 3, n_rows),
            "age_group": rng.integers(1, 9, n_rows),
            "inventory_id": rng.integers(1, 20, n_rows),
            "day_of_week": rng.integers(1, 8, n_rows),
            "hour": rng.integers(0, 24, n_rows),
        }
    )
    for k in range(1, 6):
        frame[f"l_feat_{k}"] = rng.integers(0, 10, n_rows)  # low-cardinality -> categorical
    frame["l_feat_14"] = rng.integers(0, 50, n_rows)  # query/target column
    for k in range(1, 6):
        frame[f"feat_e_{k}"] = rng.normal(size=n_rows).astype(np.float32)  # continuous
    frame.loc[rng.random(n_rows) < 0.05, "feat_e_1"] = np.nan

    def random_seq() -> str:
        length = int(rng.integers(0, 80))  # includes empty and over-length sequences
        return ",".join(str(int(v)) for v in rng.integers(1, 10_000, length))

    frame["seq"] = [random_seq() for _ in range(n_rows)]
    if with_label:
        frame["clicked"] = (rng.random(n_rows) < 0.1).astype(np.int64)
    return frame


def run(backbone: str, workdir: Path) -> dict:
    """Run train.py once and return the written metadata."""
    output_path = workdir / f"submit_{backbone}.csv"
    meta_path = workdir / f"meta_{backbone}.json"
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--train_path", str(workdir / "train.parquet"),
        "--test_path", str(workdir / "test.parquet"),
        "--output_path", str(output_path),
        "--meta_path", str(meta_path),
        "--seq_backbone", backbone,
        "--epochs", "1",
        "--batch_size", "128",
        "--hash_buckets", "1024",
        "--seq_emb_dim", "16",
        "--deep_units", "32", "16",
        "--device", "cpu",
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    submission = pd.read_csv(output_path)
    assert len(submission) == 300 and submission["clicked"].between(0, 1).all()
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbones", nargs="+", default=["din", "dien", "bst"],
                        choices=["din", "dien", "bst"])
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="ctr_smoke_") as tmp:
        workdir = Path(tmp)
        make_frame(2_000, rng, with_label=True).to_parquet(workdir / "train.parquet", index=False)
        make_frame(300, rng, with_label=False).to_parquet(workdir / "test.parquet", index=False)
        for backbone in args.backbones:
            meta = run(backbone, workdir)
            print(f"[smoke] {backbone}: OK (best_epoch={meta['performance']['best_epoch']})")


if __name__ == "__main__":
    main()
