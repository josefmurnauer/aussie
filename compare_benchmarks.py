"""
Standalone comparison script for the wwbb (L-GATr) vs wwbb_multi
(MLP+Kernel) benchmark sweep. Reads cached metrics.npz files (produced
by ClassificationExperiment.plot() / UnfoldingExperiment.plot()) --
no dataset reloading or model inference needed.

Usage:
    python compare_benchmarks.py
(edit RUNS_UNFOLDING / RUNS_ITERATION below to match your actual run
directory names)
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RUNS_DIR = "runs"

RUNS_UNFOLDING = {
    "wwbb_multi (MLP+Kernel)": "unfold_wwbb_multi/2026-07-30_14-08-59",
    "bench_small":              "unf_wwbb_bench_small/2026-07-30_16-48-10",
    "bench_medium":             "unf_wwbb_bench_medium/2026-07-30_17-00-44",
    "bench_large":              "unf_wwbb_bench_large/2026-07-30_16-51-21",
    "bench_xlarge":             "unf_wwbb_bench_xlarge/2026-07-31_09-12-25",
    "bench_medium_notag":       "unf_wwbb_bench_medium_notag/2026-07-31_12-42-02",
}

# iteration experiment runs to compare (top-level IterationExperiment dirs,
# each containing it_1/unf, it_2/unf, ... subdirectories)
RUNS_ITERATION = {
    "wwbb_multi (MLP+Kernel)": "iterate_wwbb_multi/2026-07-24_13-51-45",
    "bench_small":              "itr_wwbb_bench_small/2026-07-30_14-54-36",
    "bench_medium":             "iterate_wwbb/2026-07-30_14-38-14",
    "bench_large":              "itr_wwbb_bench_large/2026-07-30_14-53-24",
    "bench_xlarge":             "itr_wwbb_bench_xlarge/2026-07-30_14-56-18",
    "bench_medium_notag":       "itr_wwbb_bench_medium_notag/2026-07-31_09-15-59",
}

BENCHMARK_OBS_X = ["mu_pt", "mu_e", "j1_pt", "j1_e", "j2_pt", "j2_e", "j3_pt", "j3_e"]
BENCHMARK_OBS_Z = [f"{n}_truth" for n in BENCHMARK_OBS_X]

METRIC_INFO = (
    ("chi2", r"$\chi^2/N$", 1.0, False),
    ("wasserstein", "Wasserstein distance", None, False),
    ("mse", "MSE", None, True),
)

CURVE_COLORS = {"Classifier": "#009826", "AUSSIE": "#FFC300"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _load_metrics_npz(path):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _mean_over_filter(data, level, metric_name, curve_name, obs_filter):
    names_key = f"{level}/names"
    metric_key = f"{level}/{metric_name}/{curve_name}"
    if names_key not in data or metric_key not in data:
        return np.nan

    names = list(data[names_key])
    values = np.array(data[metric_key], dtype=float)

    mask = np.array([n in obs_filter for n in names])
    if not mask.any():
        return np.nan

    return float(np.nanmean(values[mask]))


def _discover_iterations(exp_dir):
    pattern = os.path.join(exp_dir, "it_*")
    dirs = glob.glob(pattern)

    def _it_num(d):
        m = re.search(r"it_(\d+)", os.path.basename(d))
        return int(m.group(1)) if m else -1

    dirs = [d for d in dirs if _it_num(d) >= 0]
    dirs.sort(key=_it_num)
    return [(_it_num(d), d) for d in dirs]


def _last_available_metrics(exp_dir):
    """Return the metrics.npz dict from the LAST iteration (highest it_N)
    that actually has a metrics.npz -- accounts for runs that stopped
    early (e.g. crashed at it_17 instead of reaching it_20)."""
    it_dirs = _discover_iterations(exp_dir)
    for it_num, it_dir in reversed(it_dirs):  # walk backwards from highest N
        npz_path = os.path.join(it_dir, "unf", "plots", "metrics.npz")
        data = _load_metrics_npz(npz_path)
        if data is not None:
            return it_num, data
    return None, None


# ----------------------------------------------------------------------------
# Part 1: single-shot unfolding comparison (bar charts)
# ----------------------------------------------------------------------------

def build_unfolding_comparison(runs_dict, out_path):
    variant_names = list(runs_dict.keys())
    curve_names = ["Classifier", "AUSSIE"]

    cache = {}
    for variant, rel_path in runs_dict.items():
        npz_path = os.path.join(RUNS_DIR, rel_path, "plots", "metrics.npz")
        data = _load_metrics_npz(npz_path)
        cache[variant] = data
        print(f"{variant}: {'OK' if data is not None else 'MISSING metrics.npz'}")

    with PdfPages(out_path) as pdf:
        for level, obs_filter, level_label in (
            ("observables_x", BENCHMARK_OBS_X, "Reco-level"),
            ("observables_z", BENCHMARK_OBS_Z, "Truth-level"),
        ):
            for metric_name, ylabel, hline, logy in METRIC_INFO:

                values = {curve: [] for curve in curve_names}
                for variant in variant_names:
                    data = cache[variant]
                    for curve in curve_names:
                        if data is None:
                            values[curve].append(np.nan)
                        else:
                            values[curve].append(
                                _mean_over_filter(data, level, metric_name, curve, obs_filter)
                            )

                if all(np.all(np.isnan(v)) for v in values.values()):
                    continue

                fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(variant_names)), 5))
                x = np.arange(len(variant_names))
                n_curves = len(curve_names)
                width = 0.8 / n_curves

                for i, curve in enumerate(curve_names):
                    offset = (i - (n_curves - 1) / 2) * width
                    vals = np.nan_to_num(np.array(values[curve]), nan=0.0)
                    ax.bar(x + offset, vals, width=width, label=curve,
                           color=CURVE_COLORS.get(curve, f"C{i}"))

                if hline is not None:
                    ax.axhline(hline, color="black", linestyle="--", linewidth=1.2,
                               label=f"Optimum ({hline:g})")

                ax.set_xticks(x)
                ax.set_xticklabels(variant_names, rotation=30, ha="right", fontsize=9)
                ax.set_ylabel(f"Mean {ylabel} (8 benchmark observables)", fontsize=10)
                if logy:
                    ax.set_yscale("log")
                ax.legend(frameon=False, fontsize=9)
                ax.set_title(f"{level_label}: {metric_name}", fontsize=12)
                ax.tick_params(axis="both", direction="in", top=True, right=True)
                fig.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)

    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------------
# Part 2a: iteration experiment comparison (line plots vs iteration)
# ----------------------------------------------------------------------------

def _add_iteration_lineplots(pdf, runs_dict):
    for level, obs_filter, level_label in (
        ("observables_x", BENCHMARK_OBS_X, "Reco-level"),
        ("observables_z", BENCHMARK_OBS_Z, "Truth-level"),
    ):
        for metric_name, ylabel, hline, logy in METRIC_INFO:

            fig, ax = plt.subplots(figsize=(8, 5.5))
            any_data = False
            cmap = plt.get_cmap("tab10")

            for i, (variant, rel_path) in enumerate(runs_dict.items()):
                exp_dir = os.path.join(RUNS_DIR, rel_path)
                it_dirs = _discover_iterations(exp_dir)

                iterations, means = [], []
                for it_num, it_dir in it_dirs:
                    npz_path = os.path.join(it_dir, "unf", "plots", "metrics.npz")
                    data = _load_metrics_npz(npz_path)
                    if data is None:
                        continue
                    mean_val = _mean_over_filter(data, level, metric_name, "AUSSIE", obs_filter)
                    iterations.append(it_num)
                    means.append(mean_val)

                if not iterations:
                    print(f"  {variant}: no iteration data found for {level}/{metric_name}")
                    continue

                ax.plot(
                    iterations, means,
                    marker="o", ms=4, lw=1.8,
                    color=cmap(i % cmap.N), label=variant,
                )
                any_data = True

            if not any_data:
                plt.close(fig)
                continue

            if hline is not None:
                ax.axhline(hline, color="black", linestyle="--", linewidth=1.0,
                           label=f"Optimum ({hline:g})")

            ax.set_xlabel("Iteration", fontsize=11)
            ax.set_ylabel(f"Mean {ylabel} (AUSSIE, 8 benchmark observables)", fontsize=10)
            if logy:
                ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=8, loc="best")
            ax.set_title(f"{level_label}: {metric_name} vs. iteration", fontsize=12)
            ax.tick_params(axis="both", direction="in", top=True, right=True)
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)


# ----------------------------------------------------------------------------
# Part 2b: iteration experiment comparison (bar chart, LAST iteration only)
# ----------------------------------------------------------------------------

def _add_iteration_final_barplots(pdf, runs_dict):
    variant_names = list(runs_dict.keys())
    curve_names = ["Classifier", "AUSSIE"]

    cache = {}
    for variant, rel_path in runs_dict.items():
        exp_dir = os.path.join(RUNS_DIR, rel_path)
        last_it, data = _last_available_metrics(exp_dir)
        cache[variant] = (last_it, data)
        status = f"it_{last_it}" if data is not None else "MISSING"
        print(f"{variant}: last available iteration = {status}")

    for level, obs_filter, level_label in (
        ("observables_x", BENCHMARK_OBS_X, "Reco-level"),
        ("observables_z", BENCHMARK_OBS_Z, "Truth-level"),
    ):
        for metric_name, ylabel, hline, logy in METRIC_INFO:

            values = {curve: [] for curve in curve_names}
            for variant in variant_names:
                last_it, data = cache[variant]
                for curve in curve_names:
                    if data is None:
                        values[curve].append(np.nan)
                    else:
                        values[curve].append(
                            _mean_over_filter(data, level, metric_name, curve, obs_filter)
                        )

            if all(np.all(np.isnan(v)) for v in values.values()):
                continue

            fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(variant_names)), 5))
            x = np.arange(len(variant_names))
            n_curves = len(curve_names)
            width = 0.8 / n_curves

            for i, curve in enumerate(curve_names):
                offset = (i - (n_curves - 1) / 2) * width
                vals = np.nan_to_num(np.array(values[curve]), nan=0.0)
                ax.bar(x + offset, vals, width=width, label=curve,
                       color=CURVE_COLORS.get(curve, f"C{i}"))

            if hline is not None:
                ax.axhline(hline, color="black", linestyle="--", linewidth=1.2,
                           label=f"Optimum ({hline:g})")

            # annotate each variant's group with which iteration was used,
            # since variants may have stopped at different iteration counts
            xticklabels = [
                f"{v}\n(it_{cache[v][0]})" if cache[v][1] is not None else f"{v}\n(missing)"
                for v in variant_names
            ]

            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(f"Mean {ylabel} (8 benchmark observables)", fontsize=10)
            if logy:
                ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=9)
            ax.set_title(f"{level_label}: {metric_name} (final iteration)", fontsize=12)
            ax.tick_params(axis="both", direction="in", top=True, right=True)
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)


def build_iteration_comparison(runs_dict, out_path):
    with PdfPages(out_path) as pdf:
        _add_iteration_lineplots(pdf, runs_dict)
        _add_iteration_final_barplots(pdf, runs_dict)

    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    build_unfolding_comparison(
        RUNS_UNFOLDING, "benchmark_comparison_unfolding.pdf"
    )
    build_iteration_comparison(
        RUNS_ITERATION, "benchmark_comparison_iteration.pdf"
    )