"""CLI entry point.

Usage::

    # Train + evaluate + package a submission for a model:
    python -m pipeline.cli train --config configs/models/gru.yaml

    # List all registered models:
    python -m pipeline.cli list

    # Sanity-check a trained run end-to-end with the official scorer:
    python -m pipeline.cli score --run experiments/gru/<run_id>

For convenience the project's ``pyproject.toml`` installs this module
as the ``lob`` console script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

CONSOLE = Console()


def _cmd_list(_args) -> int:
    # Lazy import: pulling models at module level loads torch before pandas,
    # which on Windows segfaults pyarrow when the parquet is later read.
    from models import list_models
    classical = list_models("classical")
    sequence = list_models("sequence")
    CONSOLE.print("[bold]Classical models:[/]")
    for n in classical:
        CONSOLE.print(f"  - {n}")
    CONSOLE.print("[bold]Sequence models:[/]")
    for n in sequence:
        CONSOLE.print(f"  - {n}")
    return 0


def _cmd_train(args) -> int:
    from pipeline.run import run_from_config
    result = run_from_config(args.config)
    CONSOLE.print(f"[bold green]Done.[/] {json.dumps(result, indent=2)}")
    return 0


def _cmd_score(args) -> int:
    """Replay the official ScorerStepByStep on a packaged submission."""
    from pathlib import Path
    import zipfile
    import tempfile
    import importlib.util

    run_dir = Path(args.run)
    zip_path = run_dir / "solution.zip"
    if not zip_path.exists():
        CONSOLE.print(f"[red]No solution.zip in {run_dir}[/]")
        return 1

    valid_path = args.valid or "competition_package/datasets/valid.parquet"
    sys.path.insert(0, "competition_package")

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        sys.path.insert(0, td)
        # Reload utils.py from the package's own copy.
        spec = importlib.util.spec_from_file_location("utils", Path(td) / "utils.py")
        utils_mod = importlib.util.module_from_spec(spec)
        sys.modules["utils"] = utils_mod
        spec.loader.exec_module(utils_mod)
        # Load solution.
        spec = importlib.util.spec_from_file_location("solution", Path(td) / "solution.py")
        solution_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_mod)

        model = solution_mod.PredictionModel()
        scorer = utils_mod.ScorerStepByStep(valid_path)
        results = scorer.score(model)
        CONSOLE.print(f"[bold]Score:[/] {json.dumps(results, indent=2)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("lob")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List registered models")
    p_list.set_defaults(func=_cmd_list)

    p_train = sub.add_parser("train", help="Train a model from a YAML config")
    p_train.add_argument("--config", required=True, help="Path to a YAML config")
    p_train.set_defaults(func=_cmd_train)

    p_score = sub.add_parser("score", help="Run the official scorer on a packaged run")
    p_score.add_argument("--run", required=True, help="Path to an experiments/<model>/<run_id> directory")
    p_score.add_argument("--valid", default=None, help="Path to the valid.parquet")
    p_score.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
