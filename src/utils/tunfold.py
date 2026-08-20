from dataclasses import dataclass

import numpy as np


def make_response_bins(values, obs, num_edges):
    """Compute bin edges honoring an Observable's own qlims/xlims/
    discrete/log_bins settings -- used for the internal response-matrix
    construction. Decoupled from the linear, qlims-based bins used by
    compute_chi2/wasserstein/mse for scoring elsewhere."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    lo, hi = obs.xlims or tuple(np.quantile(values, obs.qlims or (0.005, 0.995)))

    if obs.log_bins:
        lo = max(lo, 1e-10)
        return np.logspace(np.log10(lo), np.log10(hi), num_edges)
    if obs.discrete:
        d = obs.discrete
        return np.arange(lo, hi + d + 1, d) - d / 2
    return np.linspace(lo, hi, num_edges)


@dataclass
class TUnfoldResult:
    """
    weight      : per-event weight, aligned with z_sim_v/x_sim_v -- for
                  plugging into plot_reweighting's `weights_list`
    logvar      : per-event log-variance, aligned with z_sim_v -- for
                  plugging into plot_reweighting's `variance_list`
                  (log-normal moment-matching convention, consistent
                  with how every other curve's uncertainty is shown)
    x_hat       : unfolded truth-level bin heights, shape (n_truth_bins,)
    truth_edges : bin edges used for the truth histogram
    cov         : full bin-level covariance matrix of x_hat,
                  shape (n_truth_bins, n_truth_bins), propagated via
                  cov = A_pinv @ diag(var_y) @ A_pinv.T (assumes
                  UNCORRELATED statistical uncertainties across reco
                  bins as input; correlations in `cov`/`corr` arise
                  purely from the (unregularized) matrix inversion)
    corr        : correlation matrix derived from `cov`,
                  shape (n_truth_bins, n_truth_bins)
    """
    weight: np.ndarray
    logvar: np.ndarray
    x_hat: np.ndarray
    truth_edges: np.ndarray
    cov: np.ndarray
    corr: np.ndarray


def compute_tunfold_result(
    x_sim_v,
    x_dat_v,
    z_sim_v,
    z_dat_v,
    sim_weights,
    exp_weights,
    obs_x,
    obs_z,
    num_bins_truth,
    reco_bin_factor=2,
    clip_negative=True,
    weight_floor=1e-12,
):
    """
    Build the 1D truth-reco migration matrix from simulation, invert it
    via the UNREGULARIZED Moore-Penrose pseudo-inverse (tau=0, i.e. the
    "naive TUnfold" limit -- cf. Schmitt, TUnfold, JINST 2012), and
    propagate the reco-level statistical uncertainty (assumed
    UNCORRELATED across reco bins, i.e. diagonal V_y) through the
    inversion via standard linear error propagation:

        x_hat = A+ y
        V_x   = A+ . V_y . (A+)^T

    Note V_x is generally NOT diagonal even though V_y is -- any
    correlations appearing in `cov`/`corr` are induced entirely by the
    (unregularized) inversion mixing reco bins together, which is
    exactly the instability this baseline is meant to illustrate.

    The observed reco histogram `y` (and its variance) is rescaled by a
    single GLOBAL CONSTANT (sim_w.sum() / y.sum()) BEFORE inversion, so
    the result is calibrated to Sim's raw weight-sum total rather than
    Data's -- matching the convention plotting.plot_reweighting uses for
    Classifier/AUSSIE (both naturally satisfy sum(weight*sim_w) ~
    sum(sim_w) for a converged density-ratio estimator). Without this,
    TUnfold's curve would be displayed offset by roughly
    sum(exp_weights)/sum(sim_weights) relative to Classifier/AUSSIE, a
    pure display artifact that does NOT affect chi2/Wasserstein/MSE
    (already normalization-invariant) or the correlation matrix (already
    scale-invariant).

    Returns a TUnfoldResult with both a per-event weight/log-variance
    pair (for direct use with plotting.plot_reweighting) and the full
    bin-level covariance/correlation matrices (for a dedicated
    diagnostic heatmap).
    """
    num_bins_reco = int(round(num_bins_truth * reco_bin_factor))

    sim_w = sim_weights if sim_weights is not None else np.ones_like(x_sim_v)
    exp_w = exp_weights if exp_weights is not None else np.ones_like(x_dat_v)

    truth_vals = np.hstack([z_sim_v, z_dat_v]) if z_dat_v is not None else z_sim_v
    reco_vals = np.hstack([x_sim_v, x_dat_v])

    truth_edges = make_response_bins(truth_vals, obs_z, num_bins_truth)
    reco_edges = make_response_bins(reco_vals, obs_x, num_bins_reco)
    n_truth_bins = len(truth_edges) - 1

    # ------------------------------------------------------------------
    # Response matrix: M[i, j] = sum of sim weights with reco in bin i
    # AND truth in bin j; normalize each truth column to a conditional
    # probability P(reco bin i | truth bin j)
    # ------------------------------------------------------------------
    M, _, _ = np.histogram2d(
        x_sim_v, z_sim_v, bins=[reco_edges, truth_edges], weights=sim_w
    )
    col_sums = M.sum(axis=0)
    A = np.divide(M, col_sums, out=np.zeros_like(M), where=col_sums > 0)

    # ------------------------------------------------------------------
    # Observed reco-level histogram (data / pseudodata) and its
    # statistical variance PER BIN, assumed UNCORRELATED across bins
    # (diagonal V_y) -- standard weighted-sum-of-squares estimator.
    # ------------------------------------------------------------------
    y, _ = np.histogram(x_dat_v, bins=reco_edges, weights=exp_w)
    var_y, _ = np.histogram(x_dat_v, bins=reco_edges, weights=exp_w ** 2)

    # ------------------------------------------------------------------
    # Rescale y and V_y to Sim's raw total BEFORE inverting (see
    # docstring above for rationale). Single global constant -- does not
    # affect chi2/Wasserstein/MSE or the correlation matrix.
    # ------------------------------------------------------------------
    total_y = y.sum()
    total_sim = sim_w.sum()
    display_rescale = (total_sim / total_y) if total_y > 0 else 1.0

    y = y * display_rescale
    var_y = var_y * display_rescale ** 2
    V_y = np.diag(var_y)

    # ------------------------------------------------------------------
    # UNREGULARIZED pseudo-inverse solve + full covariance propagation
    # ------------------------------------------------------------------
    A_pinv = np.linalg.pinv(A)
    x_hat = A_pinv @ y
    cov_x = A_pinv @ V_y @ A_pinv.T

    var_x = np.clip(np.diag(cov_x), 0, None)
    std_x = np.sqrt(var_x)
    denom = np.outer(std_x, std_x)
    corr_x = np.divide(
        cov_x, denom, out=np.zeros_like(cov_x), where=denom > 0
    )
    np.fill_diagonal(corr_x, np.where(std_x > 0, 1.0, 0.0))

    x_hat_central = x_hat.copy()
    if clip_negative:
        x_hat = np.clip(x_hat, 0, None)

    # ------------------------------------------------------------------
    # Convert unfolded bin heights into a per-sim-truth-event weight:
    # weight_i = (x_hat[bin(z_i)] / sim_truth_hist[bin(z_i)]) * sim_w_i
    # so that histogramming sim events with this weight over truth_edges
    # exactly reproduces x_hat.
    # ------------------------------------------------------------------
    sim_truth_hist, _ = np.histogram(z_sim_v, bins=truth_edges, weights=sim_w)
    scale = np.divide(
        x_hat, sim_truth_hist, out=np.zeros_like(x_hat), where=sim_truth_hist > 0
    )

    bin_idx = np.digitize(z_sim_v, truth_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_truth_bins)

    per_event_scale = np.zeros_like(z_sim_v, dtype=np.float64)
    per_event_scale[valid] = scale[bin_idx[valid]]
    weight = (per_event_scale * sim_w).astype(np.float64)
    weight = np.clip(weight, weight_floor, None)

    # ------------------------------------------------------------------
    # Per-event log-variance, calibrated so that plot_reweighting's own
    # log-normal moment-matching formula reproduces the CORRECT
    # propagated per-bin variance exactly.
    # ------------------------------------------------------------------
    sum_w2_bin = np.zeros(n_truth_bins, dtype=np.float64)
    np.add.at(sum_w2_bin, bin_idx[valid], weight[valid] ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            var_x, sum_w2_bin, out=np.zeros_like(var_x), where=sum_w2_bin > 0
        )
        logvar_bin = np.where(ratio > 0, 0.5 * np.log(ratio), 0.0)

    per_event_logvar = np.zeros_like(z_sim_v, dtype=np.float64)
    per_event_logvar[valid] = logvar_bin[bin_idx[valid]]

    return TUnfoldResult(
        weight=weight.astype(np.float32),
        logvar=per_event_logvar.astype(np.float32),
        x_hat=x_hat_central,
        truth_edges=truth_edges,
        cov=cov_x,
        corr=corr_x,
    )


def plot_correlation_matrix(corr, truth_edges, xlabel, title=None, figsize=(5.5, 5)):
    """Simple, self-contained heatmap of a bin-level correlation matrix
    (independent of the ATLAS-style vs. default plotting module choice
    used elsewhere, since this is a new diagnostic without an existing
    convention to match)."""
    import matplotlib.pyplot as plt

    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")

    bin_centers = 0.5 * (truth_edges[1:] + truth_edges[:-1])
    tick_idx = np.linspace(0, n - 1, min(n, 8)).astype(int)
    tick_labels = [f"{bin_centers[i]:.3g}" for i in tick_idx]

    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(xlabel, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=9)

    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig


def plot_matrix_heatmap(
    matrix, reco_edges, truth_edges, xlabel, ylabel="Reco bin",
    title=None, figsize=(6, 5.5), log_counts=False, cmap="viridis",
    vmin=None, vmax=None, diverging=False,
):
    """Generic heatmap for a response/joint matrix with shape
    (n_reco, n_truth) -- reco on the y-axis, truth on the x-axis,
    matching the (i=reco, j=truth) convention used throughout
    src.utils.tunfold / src.utils.tunfold_ab.

    log_counts=True applies a log10(1+x) transform, useful for the raw
    joint histogram M (which spans orders of magnitude in bin
    population, unlike A/B which are bounded probabilities in [0, 1]).

    diverging=True uses a zero-centered red/blue colormap (RdBu_r),
    appropriate for a DIFFERENCE matrix (e.g. A* - A) rather than an
    absolute-value matrix.
    """
    import matplotlib.pyplot as plt

    display = matrix.copy()
    if log_counts:
        display = np.log10(1 + np.clip(display, 0, None))

    fig, ax = plt.subplots(figsize=figsize)

    if diverging:
        vext = vmax if vmax is not None else np.abs(display).max()
        im = ax.imshow(
            display, origin="lower", aspect="auto",
            cmap="RdBu_r", vmin=-vext, vmax=vext,
        )
    else:
        im = ax.imshow(
            display, origin="lower", aspect="auto",
            cmap=cmap, vmin=vmin, vmax=vmax,
        )

    n_reco, n_truth = matrix.shape
    truth_centers = 0.5 * (truth_edges[1:] + truth_edges[:-1])
    reco_centers = 0.5 * (reco_edges[1:] + reco_edges[:-1])

    tick_idx_x = np.linspace(0, n_truth - 1, min(n_truth, 8)).astype(int)
    tick_idx_y = np.linspace(0, n_reco - 1, min(n_reco, 8)).astype(int)

    ax.set_xticks(tick_idx_x)
    ax.set_xticklabels([f"{truth_centers[i]:.3g}" for i in tick_idx_x],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_idx_y)
    ax.set_yticklabels([f"{reco_centers[i]:.3g}" for i in tick_idx_y], fontsize=7)

    ax.set_xlabel(f"Truth {xlabel}", fontsize=10)
    ax.set_ylabel(f"{ylabel} ({xlabel})", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "log10(1 + counts)" if log_counts else ("Diff." if diverging else "Probability"),
        fontsize=9,
    )

    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig