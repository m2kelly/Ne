#!/usr/bin/env python3
"""
Parse CpG-removal recurrence vectors from a log file, run PopSizeCalculator for each
complete CpG-removal block, and save one corrected-mutation plot per block.

Recommended use from the parent directory of the NeParallel package:

    python -m NeParallel.parse_cpg_remove_logs_get_pop \
        --log logs_forward_simulated_dicts.txt \
        --out-dir test_N_e/from_log \
        --pop-min 10000 \
        --pop-max 800000 \
        --ns 20 \
        --workers 1

Or, if you keep this script outside the package, make sure NeParallel is importable
on PYTHONPATH and run:

    python parse_cpg_remove_logs_get_pop.py --log path/to/pipeline.log
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brute
from sklearn.linear_model import LinearRegression

try:
    # Works when run as: python -m NeParallel.parse_cpg_remove_logs_get_pop
    from .base_operations import Operations
    from .pop_size import PopSizeCalculator
except ImportError:
    # Works when run as a normal script, as long as NeParallel is on PYTHONPATH
    from NeParallel.base_operations import Operations
    from NeParallel.pop_size import PopSizeCalculator


VECTOR_NAMES = (
    "cpg_subs",
    "non_cpg_subs",
    "cpg_subs_bckwrds",
    "non_cpg_subs_bckwrds",
)

BLOCK_START_RE = re.compile(
    r"^Running pipeline with CpG remove percentage:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    re.MULTILINE,
)

VECTOR_RE = re.compile(
    rf"^({'|'.join(VECTOR_NAMES)})\s*=\s*(\[[^\]]*\])",
    re.MULTILINE | re.DOTALL,
)


def safe_percentage_label(value: float) -> str:
    """Turn 0.05 into 0p05 for file names."""
    return f"{value:g}".replace("-", "minus").replace(".", "p")


def parse_recurrence_blocks(log_path: str | Path, *, skip_incomplete: bool = True) -> list[dict[str, Any]]:
    """
    Parse recurrence vectors from pipeline logs.

    Returns one dict per complete CpG-removal block:
        {
            'cpg_remove_percentage': float,
            'cpg_subs': list[float],
            'non_cpg_subs': list[float],
            'cpg_subs_bckwrds': list[float],
            'non_cpg_subs_bckwrds': list[float],
        }
    """
    text = Path(log_path).read_text()
    starts = list(BLOCK_START_RE.finditer(text))

    if not starts:
        raise ValueError(f"No CpG remove percentage blocks found in {log_path}")

    blocks: list[dict[str, Any]] = []
    for i, match in enumerate(starts):
        percentage = float(match.group(1))
        block_start = match.start()
        block_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        block_text = text[block_start:block_end]

        parsed: dict[str, Any] = {"cpg_remove_percentage": percentage}
        for name, literal in VECTOR_RE.findall(block_text):
            try:
                parsed[name] = [float(x) for x in ast.literal_eval(literal)]
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse vector {name!r} in CpG-remove block {percentage}"
                ) from exc

        missing = [name for name in VECTOR_NAMES if name not in parsed]
        if missing:
            msg = (
                f"CpG-remove block {percentage} is missing vectors: "
                + ", ".join(missing)
            )
            if skip_incomplete:
                print(f"Skipping incomplete block: {msg}")
                continue
            raise ValueError(msg)

        lengths = {name: len(parsed[name]) for name in VECTOR_NAMES}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Vector lengths differ in CpG-remove block {percentage}: {lengths}"
            )

        blocks.append(parsed)

    if not blocks:
        raise ValueError("No complete CpG-removal blocks found after parsing")

    return blocks


def make_popcalc(block: dict[str, Any], *, directory: str | Path, gens: int = 1) -> PopSizeCalculator:
    operations = Operations(collapse=False, muts_dict_raw={}, occ_dict_raw={})
    return PopSizeCalculator(
        gens=gens,
        cpg_subs=block["cpg_subs"],
        non_cpg_subs=block["non_cpg_subs"],
        cpg_subs_bckwrds=block["cpg_subs_bckwrds"],
        non_cpg_subs_bckwrds=block["non_cpg_subs_bckwrds"],
        cpg_occs=[],
        non_cpg_occs=[],
        cpg_occs_bckwrds=[],
        non_cpg_occs_bckwrds=[],
        directory=str(directory),
        operations=operations,
    )


def get_pop(
    block: dict[str, Any],
    *,
    out_dir: str | Path,
    pop_min: float = 10_000,
    pop_max: float = 800_000,
    ns: int = 50,
    workers: int = 5,
    gens: int = 1,
) -> tuple[PopSizeCalculator, dict[str, Any]]:
    """Run brute-force population search and save the plot for one parsed block."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpg_pct = float(block["cpg_remove_percentage"])
    pct_label = safe_percentage_label(cpg_pct)

    popcalc = make_popcalc(block, directory=out_dir, gens=gens)

    def error_fn(x: float | np.ndarray) -> float:
        pop = float(np.asarray(x).squeeze())
        return float(popcalc.get_error_regress_per_category(pop))

    res = brute(
        error_fn,
        ((pop_min, pop_max),),
        Ns=ns,
        full_output=False,
        workers=workers,
    )

    best_pop = float(np.asarray(res).squeeze())
    min_error = error_fn(best_pop)
    popcalc.best_pop = best_pop
    popcalc.min_error = min_error

    plot_path = out_dir / f"corrected_muts_cpg_remove_{pct_label}.png"
    plot_corrected_muts(popcalc, cpg_pct=cpg_pct, plot_path=plot_path)

    result = {
        "cpg_remove_percentage": cpg_pct,
        "best_pop": best_pop,
        "min_error": min_error,
        "plot_path": str(plot_path),
        "n_bins": len(block["cpg_subs"]),
    }
    return popcalc, result


def plot_corrected_muts(popcalc: PopSizeCalculator, *, cpg_pct: float, plot_path: str | Path) -> None:
    """Plot cpg_muts vs non_cpg_muts for a fitted PopSizeCalculator."""
    cpg_muts, non_cpg_muts = popcalc.get_muts_per_cat(popcalc.best_pop)
    cpg_muts = np.asarray(cpg_muts, dtype=float)
    non_cpg_muts = np.asarray(non_cpg_muts, dtype=float)

    x = non_cpg_muts.reshape(-1, 1)
    y = cpg_muts

    plt.figure(figsize=(7, 5))
    plt.scatter(non_cpg_muts, cpg_muts, label="corrected mutations")

    # Match your previous weighted regression diagnostic.
    weights = np.arange(1, len(non_cpg_muts) + 1)
    model_weight = LinearRegression()
    model_weight.fit(x, y, sample_weight=weights)
    y_pred_weight = model_weight.predict(x)
    plt.plot(
        non_cpg_muts,
        y_pred_weight,
        label=(
            "weighted regression "
            f"coef={model_weight.coef_[0]:.2e}, "
            f"intercept={model_weight.intercept_:.2e}"
        ),
    )

    model_zero = LinearRegression(fit_intercept=False)
    model_zero.fit(x, y)
    y_pred_zero = model_zero.predict(x)
    plt.plot(
        non_cpg_muts,
        y_pred_zero,
        label=f"through-zero regression coef={model_zero.coef_[0]:.2e}",
    )

    plt.xlabel("non_cpg")
    plt.ylabel("cpg")
    plt.legend()
    plt.title(
        f"CpG removed={cpg_pct:g}; "
        f"best pop={popcalc.best_pop:.3e}; "
        f"error={popcalc.min_error:.3e}"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def plot_uncorrected_muts(popcalc: PopSizeCalculator, *, cpg_pct: float, plot_path: str | Path) -> None:
    """Plot cpg_subs vs non_cpg_subs for a fitted PopSizeCalculator."""
    cpg_muts, non_cpg_muts = popcalc.get_muts(popcalc.best_pop)
    cpg_muts = np.asarray(cpg_muts, dtype=float)
    non_cpg_muts = np.asarray(non_cpg_muts, dtype=float)

    x = non_cpg_muts.reshape(-1, 1)
    y = cpg_muts

    plt.figure(figsize=(7, 5))
    plt.scatter(non_cpg_muts, cpg_muts, label="corrected mutations")

    # Match your previous weighted regression diagnostic.
    weights = np.arange(1, len(non_cpg_muts) + 1)
    model_weight = LinearRegression()
    model_weight.fit(x, y, sample_weight=weights)
    y_pred_weight = model_weight.predict(x)
    plt.plot(
        non_cpg_muts,
        y_pred_weight,
        label=(
            "weighted regression "
            f"coef={model_weight.coef_[0]:.2e}, "
            f"intercept={model_weight.intercept_:.2e}"
        ),
    )

    model_zero = LinearRegression(fit_intercept=False)
    model_zero.fit(x, y)
    y_pred_zero = model_zero.predict(x)
    plt.plot(
        non_cpg_muts,
        y_pred_zero,
        label=f"through-zero regression coef={model_zero.coef_[0]:.2e}",
    )

    plt.xlabel("non_cpg")
    plt.ylabel("cpg")
    plt.legend()
    plt.title(
        f"CpG removed={cpg_pct:g}; "
        f"best pop={popcalc.best_pop:.3e}; "
        f"error={popcalc.min_error:.3e}"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

def run_all(
    *,
    log_path: str | Path,
    out_dir: str | Path,
    pop_min: float,
    pop_max: float,
    ns: int,
    workers: int,
    gens: int,
    skip_incomplete: bool,
) -> pd.DataFrame:
    blocks = parse_recurrence_blocks(log_path, skip_incomplete=skip_incomplete)
    results: list[dict[str, Any]] = []

    for block in blocks:
        cpg_pct = block["cpg_remove_percentage"]
        print(f"Running get_pop for CpG remove percentage {cpg_pct:g}")
        _, result = get_pop(
            block,
            out_dir=out_dir,
            pop_min=pop_min,
            pop_max=pop_max,
            ns=ns,
            workers=workers,
            gens=gens,
        )
        print(
            f"  best_pop={result['best_pop']:.6g}, "
            f"min_error={result['min_error']:.6g}, "
            f"plot={result['plot_path']}"
        )
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("cpg_remove_percentage")
    results_path = Path(out_dir) / "pop_results_by_cpg_remove_percentage.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved summary: {results_path}")
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse CpG-removal recurrence vectors from a log and run "
            "PopSizeCalculator for each complete block."
        )
    )
    parser.add_argument("--log", required=True, help="Path to pipeline log file")
    parser.add_argument("--out-dir", default="test_N_e/from_log", help="Output directory")
    parser.add_argument("--pop-min", type=float, default=10_000)
    parser.add_argument("--pop-max", type=float, default=800_000)
    parser.add_argument("--ns", type=int, default=50, help="Number of brute grid points")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--gens", type=int, default=1)
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Raise an error instead of skipping incomplete final log blocks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(
        log_path=args.log,
        out_dir=args.out_dir,
        pop_min=args.pop_min,
        pop_max=args.pop_max,
        ns=args.ns,
        workers=args.workers,
        gens=args.gens,
        skip_incomplete=not args.fail_on_incomplete,
    )


if __name__ == "__main__":
    main()
