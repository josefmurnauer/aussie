import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, binned_statistic


# ---------------------------------------------------------------------------
# Per-observable metric computation
# ---------------------------------------------------------------------------

def _filter_finite(exp, curve, exp_weights, curve_weights):
    exp = np.asarray(exp)
    curve = np.asarray(curve)

    mask_exp = np.isfinite(exp)
    mask_curve = np.isfinite(curve)
    exp = exp[mask_exp]
    curve = curve[mask_curve]

    if exp_weights is not None:
        exp_weights = np.asarray(exp_weights)[mask_exp]
    if curve_weights is not None:
        curve_weights = np.asarray(curve_weights)[mask_curve]

    return exp, curve, exp_weights, curve_weights


def compute_chi2(exp, curve, exp_weights=None, curve_weights=None,
                  num_bins=45, qlims=(0.005, 0.995), xlims=None):
    exp, curve, exp_weights, curve_weights = _filter_finite(
        exp, curve, exp_weights, curve_weights
    )
    if len(exp) == 0 or len(curve) == 0:
        return np.nan

    lo, hi = xlims or np.quantile(np.hstack([exp, curve]), qlims)
    if hi <= lo:
        return np.nan
    bins = np.linspace(lo, hi, num_bins)

    y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
    if exp_weights is None:
        err_exp = np.sqrt(y_exp)
    else:
        sum_w2 = binned_statistic(exp, exp_weights ** 2, "sum", bins=bins)[0]
        err_exp = np.sqrt(sum_w2)

    y_curve = np.histogram(curve, bins=bins, weights=curve_weights)[0]
    if curve_weights is None:
        err_curve = np.sqrt(y_curve)
    else:
        sum_w2 = binned_statistic(curve, curve_weights ** 2, "sum", bins=bins)[0]
        err_curve = np.sqrt(sum_w2)

    norm_exp = y_exp.sum()
    norm_curve = y_curve.sum()
    if norm_exp <= 0 or norm_curve <= 0:
        return np.nan

    y_exp_n = y_exp / norm_exp
    y_curve_n = y_curve / norm_curve
    err_exp_n = err_exp / norm_exp
    err_curve_n = err_curve / norm_curve

    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (y_curve_n - y_exp_n) / np.sqrt(err_curve_n ** 2 + err_exp_n ** 2)

    nonempty = (y_curve != 0) & (y_exp != 0)
    if nonempty.sum() == 0:
        return np.nan

    return float((pull[nonempty] ** 2).sum() / (num_bins - 1))


def compute_wasserstein(exp, curve, exp_weights=None, curve_weights=None):
    exp, curve, exp_weights, curve_weights = _filter_finite(
        exp, curve, exp_weights, curve_weights
    )
    if len(exp) == 0 or len(curve) == 0:
        return np.nan

    return float(
        wasserstein_distance(
            curve, exp, u_weights=curve_weights, v_weights=exp_weights
        )
    )


def compute_mse(exp, curve, exp_weights=None, curve_weights=None,
                 num_bins=45, qlims=(0.005, 0.995), xlims=None):
    exp, curve, exp_weights, curve_weights = _filter_finite(
        exp, curve, exp_weights, curve_weights
    )
    if len(exp) == 0 or len(curve) == 0:
        return np.nan

    lo, hi = xlims or np.quantile(np.hstack([exp, curve]), qlims)
    if hi <= lo:
        return np.nan
    bins = np.linspace(lo, hi, num_bins)

    y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
    y_curve = np.histogram(curve, bins=bins, weights=curve_weights)[0]

    norm_exp = y_exp.sum()
    norm_curve = y_curve.sum()
    if norm_exp <= 0 or norm_curve <= 0:
        return np.nan

    y_exp_n = y_exp / norm_exp
    y_curve_n = y_curve / norm_curve

    return float(np.mean((y_curve_n - y_exp_n) ** 2))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_observable_metrics(
    observables,
    exp_data,
    sim_data,
    exp_weights,
    sim_weights,
    weights_list,
    names_list,
    num_bins=45,
    name_sim="MC Simulation Pythia",
):
    obs_names = []
    curve_names = [name_sim] + list(names_list)

    metrics = {
        "chi2": {name: [] for name in curve_names},
        "wasserstein": {name: [] for name in curve_names},
        "mse": {name: [] for name in curve_names},
    }

    for obs in observables:
        exp_vals = obs.compute(exp_data).numpy()
        sim_vals = obs.compute(sim_data).numpy()
        obs_names.append(obs.name)

        curves = [(name_sim, sim_vals, sim_weights)]
        for ws, name in zip(weights_list, names_list):
            w = ws.mean(0) if np.ndim(ws) > 1 else ws
            curves.append((name, sim_vals, w))

        for name, curve_vals, curve_weights in curves:
            metrics["chi2"][name].append(
                compute_chi2(
                    exp_vals, curve_vals, exp_weights, curve_weights,
                    num_bins=num_bins, qlims=obs.qlims, xlims=obs.xlims,
                )
            )
            metrics["wasserstein"][name].append(
                compute_wasserstein(exp_vals, curve_vals, exp_weights, curve_weights)
            )
            metrics["mse"][name].append(
                compute_mse(
                    exp_vals, curve_vals, exp_weights, curve_weights,
                    num_bins=num_bins, qlims=obs.qlims, xlims=obs.xlims,
                )
            )

    return obs_names, metrics


def _flatten_metrics_dict(metrics_to_save):
    """Flatten the nested {level: {"names": [...], metric: {curve: [...]}}}
    structure into a flat dict of numpy arrays suitable for np.savez,
    using '/'-joined keys, e.g. 'observables_x/chi2/Classifier'."""
    flat = {}
    for level, level_dict in metrics_to_save.items():
        names = level_dict["names"]
        flat[f"{level}/names"] = np.array(names, dtype=object)
        for metric_name in ("chi2", "wasserstein", "mse"):
            if metric_name not in level_dict:
                continue
            for curve_name, values in level_dict[metric_name].items():
                key = f"{level}/{metric_name}/{curve_name}"
                flat[key] = np.array(values, dtype=float)
    return flat


# ---------------------------------------------------------------------------
# Bar plot
# ---------------------------------------------------------------------------

def plot_metric_bars(
    metric_values,
    obs_names,
    ylabel,
    figsize=(10, 5),
    title=None,
    colors=None,
    logy=False,
    hline=None,
):
    fig, ax = plt.subplots(figsize=figsize)

    curve_names = list(metric_values.keys())
    n_curves = len(curve_names)
    n_obs = len(obs_names)

    x = np.arange(n_obs)
    width = 0.8 / max(n_curves, 1)

    colors = colors or ["#B22222", "#009826", "C1", "C3", "#13b2b7"]

    for i, name in enumerate(curve_names):
        vals = np.array(metric_values[name], dtype=float)
        offset = (i - (n_curves - 1) / 2) * width
        ax.bar(
            x + offset, np.nan_to_num(vals, nan=0.0), width=width,
            label=name, color=colors[i % len(colors)],
        )

    if hline is not None:
        ax.axhline(
            hline, color="black", linestyle="--", linewidth=1.2,
            zorder=10, label=f"Optimum ({hline:g})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(obs_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=12)

    ax.tick_params(axis="both", direction="in", top=True, right=True)
    fig.tight_layout()

    return fig, ax


def save_metrics_pdf(pdf, obs_names, metrics, prefix=""):
    fig, ax = plot_metric_bars(
        metrics["chi2"], obs_names,
        ylabel=r"$\chi^2/N$",
        title=f"{prefix}Chi-squared per observable",
        hline=1.0,
    )
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plot_metric_bars(
        metrics["wasserstein"], obs_names,
        ylabel="Wasserstein distance",
        title=f"{prefix}Wasserstein distance per observable",
    )
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plot_metric_bars(
        metrics["mse"], obs_names,
        ylabel="MSE",
        title=f"{prefix}MSE per observable",
        logy=True,
    )
    pdf.savefig(fig)
    plt.close(fig)