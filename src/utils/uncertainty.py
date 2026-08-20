import numpy as np

from src.utils.tunfold import plot_correlation_matrix  # reused generic heatmap


def _draw_bootstrap_multipliers(rng, n_events, distribution, lognormal_sigma):
    """
    Draw n_events i.i.d. multiplicative bootstrap weights, mean exactly
    1.0 by construction, under the chosen distribution:

      - "poisson":   m ~ Poisson(lambda=1)  -- classic Poisson bootstrap,
                     integer-valued, E[m]=1, Var[m]=1.
      - "lognormal": m ~ LogNormal(mu, sigma), with mu = -sigma^2 / 2 so
                     that E[m] = exp(mu + sigma^2/2) = 1 EXACTLY,
                     regardless of the chosen sigma (the underlying
                     NORMAL's standard deviation, not m's own std).
                     Continuous-valued, strictly positive, right-skewed.
                     Var[m] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
                             = exp(sigma^2) - 1   (given mu = -sigma^2/2)
                     e.g. sigma=1 -> Var[m] = e - 1 ~= 1.718.

    Both are independent, per-event draws (no sharing of randomness
    between events) -- so, exactly as with the Poisson case, two
    DISJOINT histogram bins built from either distribution have
    provably ZERO true covariance; any observed off-diagonal signal in
    bootstrap_histogram_covariance's output is pure finite-n_boot
    estimator noise, not a real effect (see the accompanying discussion
    in the codebase's development notes for the full derivation).
    """
    if distribution == "poisson":
        return rng.poisson(lam=1.0, size=n_events)
    elif distribution == "lognormal":
        mu = -0.5 * lognormal_sigma ** 2
        return rng.lognormal(mean=mu, sigma=lognormal_sigma, size=n_events)
    else:
        raise ValueError(f"Unknown bootstrap distribution '{distribution}'")


def bootstrap_histogram_covariance(
    values, weights, bins, n_boot=200, seed=0,
    distribution="lognormal", lognormal_sigma=1.0,
):
    """
    Bootstrap covariance/correlation of a weighted histogram, using
    ALREADY-COMPUTED per-event weights -- no retraining needed.

    SCOPE: captures statistical uncertainty due to the FINITE SIZE of
    the population being histogrammed, holding the per-event weight
    FUNCTION fixed. This is MC-stat / data-reference-stat uncertainty,
    NOT the data-statistical uncertainty of a TRAINED result (see
    histogram_covariance_from_replicas / decompose_data_stat_covariance
    for that).

    distribution: "lognormal" (default) or "poisson" -- see
    _draw_bootstrap_multipliers for the exact parameterization. Both
    give i.i.d. per-event multipliers with mean exactly 1, so both
    leave the mean histogram unbiased; they differ in higher moments
    (lognormal is continuous and right-skewed, closer in spirit to a
    smoothly-varying systematic-style perturbation; Poisson is the
    classical discrete nonparametric-bootstrap approximation).

    Returns
    -------
    mean : ndarray, shape (n_bins,)
    cov  : ndarray, shape (n_bins, n_bins)
    corr : ndarray, shape (n_bins, n_bins)
    """
    rng = np.random.default_rng(seed)
    n_bins = len(bins) - 1

    bin_idx = np.digitize(values, bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    v_bin_idx = bin_idx[valid]
    v_weights = np.asarray(weights)[valid].astype(np.float64)
    n_events = len(v_weights)

    histograms = np.empty((n_boot, n_bins), dtype=np.float64)
    for b in range(n_boot):
        m = _draw_bootstrap_multipliers(rng, n_events, distribution, lognormal_sigma)
        w = m * v_weights
        histograms[b] = np.bincount(v_bin_idx, weights=w, minlength=n_bins)

    mean = histograms.mean(axis=0)
    cov = np.cov(histograms, rowvar=False, ddof=1)

    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, np.where(std > 0, 1.0, 0.0))

    return mean, cov, corr


def compute_simple_bin_errors(values, weights, bins):
    """
    The SIMPLE, diagonal-only per-bin statistical error formula already
    used elsewhere in this codebase (e.g. plotting.plot_reweighting's
    own error bands):

        mean[i] = sum of weights in bin i
        err[i]  = sqrt(sum of weights^2 in bin i)

    This is the exact closed-form result for the diagonal of a
    bootstrap covariance matrix in the n_boot -> infinity limit (for
    EITHER the Poisson or the lognormal-with-matched-variance
    bootstrap), so comparing this against
    sqrt(diag(bootstrap_histogram_covariance(...)[1])) is a direct
    closure/validation check of the bootstrap machinery, in addition to
    being useful on its own as a cheap, non-stochastic error estimate.

    Returns
    -------
    mean : ndarray, shape (n_bins,)
    err  : ndarray, shape (n_bins,)
    """
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)
    n_bins = len(bins) - 1

    mean, _ = np.histogram(values, bins=bins, weights=weights)
    sum_w2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    err = np.sqrt(np.clip(sum_w2, 0, None))

    return mean, err


def histogram_covariance_from_replicas(values, weights, bins):
    """
    Compute the mean histogram and its covariance/correlation directly
    from K DISTINCT full weight arrays (`weights`, shape (K, N)) -- e.g.
    K independently-trained ensemble members, each evaluated on the
    SAME fixed event population `values`. Performs NO resampling
    itself.

    Returns
    -------
    mean : ndarray, shape (n_bins,)
    cov  : ndarray, shape (n_bins, n_bins)
    corr : ndarray, shape (n_bins, n_bins)
    """
    weights = np.atleast_2d(weights)
    K, N = weights.shape
    n_bins = len(bins) - 1

    histograms = np.empty((K, n_bins), dtype=np.float64)
    for k in range(K):
        histograms[k], _ = np.histogram(values, bins=bins, weights=weights[k])

    mean = histograms.mean(axis=0)
    if K > 1:
        cov = np.cov(histograms, rowvar=False, ddof=1)
    else:
        cov = np.zeros((n_bins, n_bins))

    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, np.where(std > 0, 1.0, 0.0))

    return mean, cov, corr


def decompose_data_stat_covariance(cov_total, cov_training_only, log=None, label=""):
    """
    Approximate decomposition of a K-replica covariance matrix
    (`cov_total`, from a DATA-BOOTSTRAP-VARYING ensemble) into a pure
    DATA-statistical component, given a matching CONTROL ensemble
    covariance (`cov_training_only`):

        Cov_data ~= Cov_total - Cov_training_only

    Negative resulting diagonal entries are clipped to zero, with a
    count logged.
    """
    assert cov_total.shape == cov_training_only.shape, (
        f"Shape mismatch: cov_total {cov_total.shape} vs "
        f"cov_training_only {cov_training_only.shape}"
    )

    cov_data = cov_total - cov_training_only

    diag = np.diag(cov_data).copy()
    n_negative = int((diag < 0).sum())
    n_total = len(diag)

    if log is not None:
        frac = n_negative / n_total if n_total > 0 else 0.0
        if n_negative > 0:
            log.warning(
                f"decompose_data_stat_covariance{f' [{label}]' if label else ''}: "
                f"{n_negative}/{n_total} bins ({frac:.1%}) had negative variance "
                f"after subtraction -- clipped to zero. Training/initialization "
                f"stochasticity is NOT subdominant to data-bootstrap variance in "
                f"these bins."
            )
        else:
            log.info(
                f"decompose_data_stat_covariance{f' [{label}]' if label else ''}: "
                f"all {n_total} bins had non-negative variance after subtraction."
            )

    diag_clipped = np.clip(diag, 0, None)
    np.fill_diagonal(cov_data, diag_clipped)

    std = np.sqrt(diag_clipped)
    denom = np.outer(std, std)
    corr_data = np.divide(cov_data, denom, out=np.zeros_like(cov_data), where=denom > 0)
    np.fill_diagonal(corr_data, np.where(std > 0, 1.0, 0.0))

    return cov_data, corr_data, n_negative


def plot_covariance_matrix(cov, bin_edges, xlabel, title=None, figsize=(5.5, 5)):
    """Heatmap of a bin-level covariance matrix."""
    import matplotlib.pyplot as plt

    n = cov.shape[0]
    vext = np.abs(cov).max() if np.abs(cov).max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cov, vmin=-vext, vmax=vext, cmap="RdBu_r", origin="lower")

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    tick_idx = np.linspace(0, n - 1, min(n, 8)).astype(int)
    tick_labels = [f"{bin_centers[i]:.3g}" for i in tick_idx]

    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(xlabel, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance", fontsize=9)

    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig


def plot_relative_error_comparison(
    bin_edges, mean_vals, err_simple, err_cov, xlabel,
    mean_before=None, err_before=None,
    label_simple=r"$\sqrt{\Sigma w^2}$ (simple, diagonal-only)",
    label_cov=r"$\sqrt{\mathrm{diag}(\mathrm{Cov})}$ (bootstrap)",
    label_before=r"$\sqrt{\Sigma w_{\rm sim}^2}$ (before unfolding)",
    title=None, figsize=(6, 4.5),
):
    """
    Step-histogram comparison of per-bin RELATIVE statistical error
    estimates:

        rel_simple[i] = err_simple[i] / mean_vals[i]
        rel_cov[i]    = err_cov[i]    / mean_vals[i]
        rel_before[i] = err_before[i] / mean_before[i]   (if provided)

    err_simple is the closed-form sqrt(sum(w^2)) formula for the
    POST-unfolding weight (Classifier or AUSSIE); err_cov is
    sqrt(diag(Cov)) from a full bootstrap covariance matrix built from
    the SAME weight. err_before/mean_before (both optional, pass both
    or neither) are the same closed-form formula applied to the
    PRE-unfolding (raw simulation / MC sample) weight, over the SAME
    binning -- letting you see directly whether reweighting inflates
    the per-bin relative statistical uncertainty relative to the
    original, unweighted-or-uniformly-weighted simulation (a generic
    consequence of reweighting by a non-uniform density ratio).
    """
    import matplotlib.pyplot as plt

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_simple = np.divide(
            err_simple, mean_vals, out=np.full_like(err_simple, np.nan), where=mean_vals > 0
        )
        rel_cov = np.divide(
            err_cov, mean_vals, out=np.full_like(err_cov, np.nan), where=mean_vals > 0
        )

    def dup(a):
        return np.append(a, a[-1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(bin_edges, dup(rel_simple), where="post", color="#B22222", lw=1.6, label=label_simple)
    ax.step(bin_edges, dup(rel_cov), where="post", color="#1B2A4A", lw=1.6, ls="--", label=label_cov)

    if mean_before is not None and err_before is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_before = np.divide(
                err_before, mean_before, out=np.full_like(err_before, np.nan), where=mean_before > 0
            )
        ax.step(bin_edges, dup(rel_before), where="post", color="#009826", lw=1.6, ls=":", label=label_before)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Relative statistical error", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    fig.tight_layout()
    return fig

def plot_scalar_evolution(
    iterations, series_dict, ylabel, title=None, figsize=(7, 5), logy=False,
):
    """
    Line plot of one or more scalar uncertainty summaries vs. OmniFold
    iteration number. `series_dict` maps a label (e.g. "Conflated
    (bootstrap-ensemble)", "Clean (control-subtracted)") to an array of
    values aligned with `iterations`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    for i, (label, values) in enumerate(series_dict.items()):
        ax.plot(iterations, values, marker="o", ms=4, lw=1.8,
                color=cmap(i % cmap.N), label=label)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=12)
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    fig.tight_layout()
    return fig


def plot_correlation_matrix_grid(
    corr_list, iteration_labels, bin_edges, xlabel, title=None, figsize=None,
):
    """
    Small-multiples grid of correlation-matrix heatmaps at several
    representative iterations, sharing one color scale ([-1, 1]) so
    they're directly comparable by eye -- much more readable than one
    full-page heatmap per iteration when scanning for convergence.
    """
    import matplotlib.pyplot as plt

    n = len(corr_list)
    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (3.0 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    n_bins = corr_list[0].shape[0]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    tick_idx = np.linspace(0, n_bins - 1, min(n_bins, 4)).astype(int)
    tick_labels = [f"{bin_centers[i]:.2g}" for i in tick_idx]

    im = None
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if idx < n:
            im = ax.imshow(corr_list[idx], vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
            ax.set_title(iteration_labels[idx], fontsize=9)
            ax.set_xticks(tick_idx)
            ax.set_yticks(tick_idx)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(tick_labels, fontsize=6)
        else:
            ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Correlation", fontsize=9)

    if title:
        fig.suptitle(f"{title}\n({xlabel})", fontsize=12)

    return fig