"""Second TCN sweep: kernel_size=5 with L=5 and L=6 (both ch=96).

Tests whether widening the kernel improves on the k=3 winner (variant
C_L6_ch96, val_pearson=+0.2740). Wider kernels see more steps per layer
and double the receptive field, at ~67% more conv params.

  E: k=5, L=5, ch=96 → RF = 249  (matches A's RF with a wider kernel)
  F: k=5, L=6, ch=96 → RF = 505  (biggest tested so far)
"""

from __future__ import annotations

# IMPORTANT: pandas/pyarrow before torch (Windows DLL ordering).
import pandas as pd  # noqa: F401

import json
import time
from pathlib import Path

import yaml

from pipeline.config import RunConfig
from pipeline.run import _run_sequence

ROOT = Path(__file__).resolve().parent.parent
BASE_CFG = ROOT / "configs" / "models" / "tcn.yaml"


SWEEP = [
    ("E_k5_L5_ch96", {"channels": 96, "num_layers": 5, "kernel_size": 5, "dropout": 0.10}),
    ("F_k5_L6_ch96", {"channels": 96, "num_layers": 6, "kernel_size": 5, "dropout": 0.10}),
]


def _load_base() -> dict:
    with BASE_CFG.open() as fh:
        return yaml.safe_load(fh)


def _run_variant(tag: str, params: dict) -> dict:
    base = _load_base()
    base["model"]["params"] = params
    base["training"]["epochs"] = 25
    cfg = RunConfig(raw=base)
    print(f"\n========== {tag}  params={params} ==========")
    t0 = time.time()
    result = _run_sequence(cfg, "tcn")
    elapsed = time.time() - t0
    metrics_path = Path(result["run_dir"]) / "val" / "metrics.json"
    with metrics_path.open() as fh:
        m = json.load(fh)
    return {
        "tag": tag,
        "params": params,
        "run_dir": result["run_dir"],
        "val_pearson": m["weighted_pearson"],
        "t0": m["per_target"]["t0"],
        "t1": m["per_target"]["t1"],
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    rows = []
    for tag, params in SWEEP:
        rows.append(_run_variant(tag, params))

    print("\n\n" + "=" * 80)
    print("TCN KERNEL=5 SWEEP RESULTS")
    print("=" * 80)
    print(f"{'tag':<18} {'corr':>8}  {'t0':>8}  {'t1':>8}  {'ratio':>6}  {'sec':>6}")
    print("-" * 80)
    for r in sorted(rows, key=lambda r: -r["val_pearson"]):
        ratio = r["t0"] / r["t1"] if r["t1"] != 0 else float("inf")
        print(
            f"{r['tag']:<18} {r['val_pearson']:+.4f}  {r['t0']:+.4f}  {r['t1']:+.4f}  "
            f"{ratio:5.2f}x  {r['elapsed_seconds']:6.1f}"
        )

    out_csv = ROOT / "experiments" / "tcn_sweep_kernel5_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
