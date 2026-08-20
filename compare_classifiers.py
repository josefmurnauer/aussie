"""
Extended reco-level classifier comparison for wwbb vs wwbb_multi:

  1. Relative Wasserstein distance (W / std(observable)) -- makes the
     per-observable Wasserstein numbers comparable across observables
     with very different physical scales (e.g. mu_pt in GeV vs j1_e in
     GeV but with very different typical magnitudes/spreads).

  2. Wasserstein noise floor -- the Wasserstein distance you'd measure
     between two random halves of the SAME data population (zero true
     mismatch, pure finite-sample noise). Reporting W_measured /
     W_floor tells you how many "noise-widths" away from a perfect
     reweighting each classifier actually is.

  3. Effective sample size (Kish's ESS) of the classifier weights --
     N_eff = (sum w)^2 / sum(w^2). Drops sharply below N if a small
     number of events carry disproportionately large weight, which is
     exactly the failure mode that destabilizes AUSSIE's step-2
     kernel-based / NTK-gradient-matching loss, even when the
     classifier's MARGINAL fit quality (chi2/Wasserstein/MSE) looks
     fine. This is the more directly AUSSIE-relevant diagnostic.

A good marginal Wasserstein/chi2 fit at step 1 is NECESSARY but NOT
SUFFICIENT for a good AUSSIE result at step 2 -- see effective_sample_size
and the associated printed warnings below for why.

Usage:
    python compare_classifiers_uncertainty.py \
        --run_wwbb runs/classify_wwbb/<timestamp> \
        --run_wwbb_multi runs/classify_wwbb_multi/<timestamp> \
        --n_splits 50
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.metrics import compute_wasserstein

BENCHMARK_OBS = ["mu_pt", "mu_e", "j1_pt", "j1_e", "j2_pt", "j2_e", "j3_pt", "j3_e"]

COLOR_WWBB       = "#1B2A4A"
COLOR_WWBB_MULTI = "#009826"


# ----------------------------------------------------------------------------
# Data loading (same seeded split as TrainingExperiment.split_dataset)
# ----------------------------------------------------------------------------

def load_run(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    dset = instantiate(cfg.dataset.reader)
    process = instantiate(cfg.dataset.process)

    fixed_rng = torch.Generator().manual_seed(1729)
    dcfg = cfg.data
    splits = random_split(
        dset,
        [1 - dcfg.val_frac - dcfg.test_frac, dcfg.val_frac, dcfg.test_frac],
        generator=fixed_rng,
    )
    batch = splits[2][:]

    labels = batch.labels
    mask_sim = (labels == 0).numpy()
    mask_dat = ~mask_sim

    record = np.load(os.path.join(run_dir, "predictions_test.npz"))
    lw_x = record["lw_x"].mean(0)
    lw_x_sim = lw_x[mask_sim]

    sample_logweights = batch.sample_logweights
    if sample_logweights is not None:
        lw_sample = sample_logweights.numpy()
        lw_x_sim = lw_x_sim + lw_sample[mask_sim]
        sim_weights = np.exp(lw_sample[mask_sim])
        exp_weights = np.exp(lw_sample[mask_dat])
    else:
        sim_weights = None
        exp_weights = None

    classifier_weight = np.exp(lw_x_sim)
    x_all = batch.x if dset.aux_x is None else batch.aux_x

    return dict(
        process=process,
        x_sim=x_all[mask_sim],
        x_dat=x_all[mask_dat],
        classifier_weight=classifier_weight,
        sim_weights=sim_weights,
        exp_weights=exp_weights,
    )


def get_observable(process, name):
    for obs in process.observables_x:
        if obs.name == name:
            return obs
    raise KeyError(f"Observable '{name}' not found in this process")


# ----------------------------------------------------------------------------
# 1 & 2: relative Wasserstein + noise floor
# ----------------------------------------------------------------------------

def weighted_std(values, weights=None):
    if weights is None:
        return float(np.std(values))
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return float(np.sqrt(var))


def wasserstein_noise_floor(dat_v, exp_w, n_splits=50, rng=None):
    """Bootstrap floor: Wasserstein distance between two random halves
    of the SAME data population -- estimates the noise level you'd see
    from finite statistics alone, with zero true reweighting error.
    Returns (mean, std) across n_splits random halvings."""
    rng = rng or np.random.default_rng(0)
    n = len(dat_v)
    vals = np.empty(n_splits)
    for i in range(n_splits):
        idx = rng.permutation(n)
        h1, h2 = idx[: n // 2], idx[n // 2:]
        w1 = exp_w[h1] if exp_w is not None else None
        w2 = exp_w[h2] if exp_w is not None else None
        vals[i] = compute_wasserstein(dat_v[h1], dat_v[h2], w1, w2)
    return float(vals.mean()), float(vals.std())


# ----------------------------------------------------------------------------
# 3: effective sample size
# ----------------------------------------------------------------------------

def effective_sample_size(weights):
    """Kish's effective sample size: N_eff = (sum w)^2 / sum(w^2).
    Equal to N if all weights are equal; drops sharply if a few events
    carry disproportionately large weight -- directly relevant to how
    noisy/unstable the per-event weight target is for AUSSIE's
    kernel-based (KernelUnfolder) or gradient-matching (AutoDiffUnfolder)
    loss in step 2, independent of how good the classifier's MARGINAL
    fit quality looks."""
    weights = np.asarray(weights, dtype=np.float64)
    return float((weights.sum() ** 2) / (weights ** 2).sum())


def weight_tail_summary(weights):
    """Additional diagnostics: max weight relative to mean, and the
    fraction of total weight carried by the top 1% of events -- both
    directly indicate how much a small number of events could dominate
    AUSSIE's per-batch loss terms."""
    weights = np.asarray(weights, dtype=np.float64)
    mean_w = weights.mean()
    max_over_mean = weights.max() / mean_w if mean_w > 0 else np.nan

    n_top = max(1, int(0.01 * len(weights)))
    sorted_w = np.sort(weights)[::-1]
    top1pct_frac = sorted_w[:n_top].sum() / weights.sum()

    return max_over_mean, top1pct_frac


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(run_wwbb, run_wwbb_multi, out_path="compare_classifiers_uncertainty.pdf",
         n_splits=50, seed=0):

    rng = np.random.default_rng(seed)

    print(f"Loading wwbb run:       {run_wwbb}")
    a = load_run(run_wwbb)
    print(f"Loading wwbb_multi run: {run_wwbb_multi}")
    b = load_run(run_wwbb_multi)

    rows = []

    for name in BENCHMARK_OBS:
        obs_a = get_observable(a["process"], name)
        obs_b = get_observable(b["process"], name)

        xa_sim, xa_dat = obs_a.compute(a["x_sim"]).numpy(), obs_a.compute(a["x_dat"]).numpy()
        xb_sim, xb_dat = obs_b.compute(b["x_sim"]).numpy(), obs_b.compute(b["x_dat"]).numpy()
        wa, wb = a["classifier_weight"], b["classifier_weight"]

        # ---- measured Wasserstein ----
        w_a = compute_wasserstein(xa_dat, xa_sim, a["exp_weights"], wa)
        w_b = compute_wasserstein(xb_dat, xb_sim, b["exp_weights"], wb)

        # ---- normalize by data-population std (weighted if applicable) ----
        std_a = weighted_std(xa_dat, a["exp_weights"])
        std_b = weighted_std(xb_dat, b["exp_weights"])

        w_rel_a = w_a / std_a if std_a > 0 else np.nan
        w_rel_b = w_b / std_b if std_b > 0 else np.nan

        # ---- noise floor (each run scored against its OWN data population) ----
        floor_a_mean, floor_a_std = wasserstein_noise_floor(xa_dat, a["exp_weights"], n_splits, rng)
        floor_b_mean, floor_b_std = wasserstein_noise_floor(xb_dat, b["exp_weights"], n_splits, rng)

        ratio_a = w_a / floor_a_mean if floor_a_mean > 0 else np.nan
        ratio_b = w_b / floor_b_mean if floor_b_mean > 0 else np.nan

        rows.append(dict(
            name=name,
            w_a=w_a, w_b=w_b,
            std_a=std_a, std_b=std_b,
            w_rel_a=w_rel_a, w_rel_b=w_rel_b,
            floor_a=floor_a_mean, floor_a_std=floor_a_std,
            floor_b=floor_b_mean, floor_b_std=floor_b_std,
            ratio_a=ratio_a, ratio_b=ratio_b,
        ))

    # ---- effective sample size + weight-tail diagnostics (event-level,
    # not per-observable -- one classifier weight array per network) ----
    n_sim_a, n_sim_b = len(a["classifier_weight"]), len(b["classifier_weight"])
    ess_a = effective_sample_size(a["classifier_weight"])
    ess_b = effective_sample_size(b["classifier_weight"])
    max_over_mean_a, top1pct_a = weight_tail_summary(a["classifier_weight"])
    max_over_mean_b, top1pct_b = weight_tail_summary(b["classifier_weight"])

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("1-2. RELATIVE WASSERSTEIN + NOISE FLOOR (per observable)")
    print("=" * 100)
    header = (
        f"{'observable':<10} {'W(wwbb)':>9} {'W(multi)':>9} "
        f"{'Wrel(wwbb)':>11} {'Wrel(multi)':>12} "
        f"{'W/floor(wwbb)':>14} {'W/floor(multi)':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<10} {r['w_a']:>9.4f} {r['w_b']:>9.4f} "
            f"{r['w_rel_a']:>11.4f} {r['w_rel_b']:>12.4f} "
            f"{r['ratio_a']:>14.2f} {r['ratio_b']:>15.2f}"
        )
    print(
        "\nWrel = W / std(data observable)                     -- dimensionless,"
        " comparable across observables\n"
        "W/floor = W_measured / W_noise_floor (from n_splits="
        f"{n_splits} random data-population halvings)\n"
        "  ~1   : classifier's mismatch is indistinguishable from pure"
        " sampling noise (excellent)\n"
        "  >>1  : classifier has a genuine, statistically significant"
        " residual mismatch on this observable"
    )

    print("\n" + "=" * 100)
    print("3. EFFECTIVE SAMPLE SIZE + WEIGHT TAIL DIAGNOSTICS (classifier weights, event-level)")
    print("=" * 100)
    print(f"{'':<28} {'wwbb':>15} {'wwbb_multi':>15}")
    print("-" * 60)
    print(f"{'N (sim test events)':<28} {n_sim_a:>15d} {n_sim_b:>15d}")
    print(f"{'N_eff (Kish ESS)':<28} {ess_a:>15.1f} {ess_b:>15.1f}")
    print(f"{'N_eff / N':<28} {ess_a / n_sim_a:>15.4f} {ess_b / n_sim_b:>15.4f}")
    print(f"{'max(w) / mean(w)':<28} {max_over_mean_a:>15.2f} {max_over_mean_b:>15.2f}")
    print(f"{'top 1% weight fraction':<28} {top1pct_a:>15.4f} {top1pct_b:>15.4f}")
    print(
        "\nN_eff/N close to 1.0: weights are nearly uniform -- stable target for"
        " AUSSIE step 2.\n"
        "N_eff/N << 1.0: a small number of events dominate the weight sum --"
        " expect a noisier,\n"
        "  less stable AUSSIE training signal (KernelUnfolder's per-batch"
        " kernel norm and\n"
        "  AutoDiffUnfolder's NTK gradient are both disproportionately"
        " sensitive to weight outliers),\n"
        "  REGARDLESS of how good the classifier's marginal"
        " chi2/Wasserstein/MSE looks."
    )

    if ess_a / n_sim_a < 0.5 * (ess_b / n_sim_b) or ess_b / n_sim_b < 0.5 * (ess_a / n_sim_a):
        worse = "wwbb" if ess_a / n_sim_a < ess_b / n_sim_b else "wwbb_multi"
        print(
            f"\n*** WARNING: {worse}'s classifier weights have a substantially "
            f"lower effective-sample-size fraction than the other pipeline. "
            f"Even if its marginal observable fits look comparable or better, "
            f"expect {worse}'s step-2 (AUSSIE) training to be relatively less "
            f"stable due to weight-outlier sensitivity. ***"
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    with PdfPages(out_path) as pdf:

        # ---- relative Wasserstein bar chart ----
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows)), 5))
        x = np.arange(len(rows))
        width = 0.35
        ax.bar(x - width / 2, [r["w_rel_a"] for r in rows], width=width,
               color=COLOR_WWBB, label="wwbb")
        ax.bar(x + width / 2, [r["w_rel_b"] for r in rows], width=width,
               color=COLOR_WWBB_MULTI, label="wwbb_multi")
        ax.set_xticks(x)
        ax.set_xticklabels([r["name"] for r in rows], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Relative Wasserstein  (W / std(data))", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Relative Wasserstein distance per observable", fontsize=12)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- W / noise-floor bar chart (log scale, with floor=1 reference) ----
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows)), 5))
        ax.bar(x - width / 2, [r["ratio_a"] for r in rows], width=width,
               color=COLOR_WWBB, label="wwbb")
        ax.bar(x + width / 2, [r["ratio_b"] for r in rows], width=width,
               color=COLOR_WWBB_MULTI, label="wwbb_multi")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                   label="Noise floor (=1, statistically perfect fit)")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([r["name"] for r in rows], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("W$_{measured}$ / W$_{noise floor}$ (log scale)", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"Wasserstein distance relative to noise floor (n_splits={n_splits})", fontsize=12)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- effective sample size fraction bar chart ----
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(["wwbb", "wwbb_multi"], [ess_a / n_sim_a, ess_b / n_sim_b],
               color=[COLOR_WWBB, COLOR_WWBB_MULTI])
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="N_eff/N = 1 (uniform weights)")
        ax.set_ylabel("N$_{eff}$ / N", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title("Classifier weight effective sample size fraction", fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- weight distribution histogram (log-x, log-y), for visual
        # inspection of the tail behavior underlying the ESS numbers ----
        fig, ax = plt.subplots(figsize=(6, 5))
        bins = np.logspace(
            np.log10(min(a["classifier_weight"].min(), b["classifier_weight"].min(), 1e-3)),
            np.log10(max(a["classifier_weight"].max(), b["classifier_weight"].max())),
            50,
        )
        ax.hist(a["classifier_weight"], bins=bins, histtype="step", color=COLOR_WWBB,
                label=f"wwbb (N_eff/N={ess_a / n_sim_a:.3f})", density=True, lw=1.5)
        ax.hist(b["classifier_weight"], bins=bins, histtype="step", color=COLOR_WWBB_MULTI,
                label=f"wwbb_multi (N_eff/N={ess_b / n_sim_b:.3f})", density=True, lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Classifier weight (per sim event)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Classifier weight distribution (log-log)", fontsize=12)
        ax.tick_params(axis="both", direction="in", top=True, right=True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_wwbb", required=True)
    parser.add_argument("--run_wwbb_multi", required=True)
    parser.add_argument("--out", default="compare_classifiers_uncertainty.pdf")
    parser.add_argument("--n_splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.run_wwbb, args.run_wwbb_multi, args.out, args.n_splits, args.seed)