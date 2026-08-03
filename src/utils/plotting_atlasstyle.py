import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from scipy.stats import binned_statistic

try:
    import atlas_mpl_style as ampl
    ampl.use_atlas_style()

    _FONT_RESET = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.serif": ["DejaVu Serif"],
        "font.monospace": ["DejaVu Sans Mono"],
        "font.cursive": ["DejaVu Sans"],
        "font.fantasy": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    }
    for _k, _v in _FONT_RESET.items():
        plt.rcParams[_k] = _v

    HAS_AMPL = True
except ImportError:
    HAS_AMPL = False
    print("WARNING: atlas-mpl-style not installed. ATLAS labels/style will be skipped.")


# ---------------------------------------------------------------------------
# Legend / label style
# ---------------------------------------------------------------------------

LEGEND_KWARGS = dict(
    frameon=False,
    fontsize=8,
    handlelength=1.2,
    handletextpad=0.4,
    labelspacing=0.3,
    columnspacing=0.8,
    borderaxespad=0.4,
    markerscale=0.8,
)

FIGSIZE_SCALE = 0.92
SUBPLOT_MARGINS = dict(left=0.18, right=0.95, top=0.93, bottom=0.13)

ATLAS_LABEL_FONTSIZE = 10
AXIS_LABEL_FONTSIZE = 11


# ---------------------------------------------------------------------------
# ATLAS label helper
# ---------------------------------------------------------------------------

def _parse_atlas_status(atlas_label: str):
    label_lower = (atlas_label or "").lower()
    simulation = "simulation" in label_lower
    remainder = label_lower.replace("simulation", "").strip()

    if "preliminary" in remainder:
        status = "prelim"
    elif "internal" in remainder:
        status = "int"
    elif "work in progress" in remainder or remainder == "wip":
        status = "wip"
    elif remainder in ("", "final", "approved"):
        status = "final"
    else:
        status = "int"

    return status, simulation


def add_atlas_label(
    ax,
    atlas_label:  str = "Simulation",
    subtext:      str = "Work in progress",
    atlas_second: str = None,
    x: float = 0.05,
    y: float = 0.95,
    fontsize: float = ATLAS_LABEL_FONTSIZE,
):
    if not HAS_AMPL:
        return

    status, simulation = _parse_atlas_status(atlas_label)
    desc_lines = [t for t in (subtext, atlas_second) if t]
    desc = "\n".join(desc_lines) if desc_lines else None

    try:
        ampl.draw_atlas_label(
            x, y, ax=ax, status=status, simulation=simulation, desc=desc,
            fontsize=fontsize,
        )
    except TypeError:
        try:
            ampl.draw_atlas_label(
                x, y, ax=ax, status=status, simulation=simulation, desc=desc,
            )
        except TypeError:
            try:
                ampl.draw_atlas_label(x, y, ax=ax)
            except Exception as e:
                print(f"WARNING: atlas_mpl_style.draw_atlas_label failed: {e}")
        except Exception as e:
            print(f"WARNING: atlas_mpl_style.draw_atlas_label failed: {e}")


# ---------------------------------------------------------------------------
# Legend helper -- forces the pseudo-data entry to the top
# ---------------------------------------------------------------------------

def _legend_with_data_first(ax, name_exp, **legend_kwargs):
    handles, labels = ax.get_legend_handles_labels()
    data_idx = next((i for i, lab in enumerate(labels) if lab == name_exp), None)
    if data_idx is not None and data_idx != 0:
        handles = [handles[data_idx]] + handles[:data_idx] + handles[data_idx + 1:]
        labels = [labels[data_idx]] + labels[:data_idx] + labels[data_idx + 1:]
    ax.legend(handles, labels, **legend_kwargs)


# ---------------------------------------------------------------------------
# Axis / style helpers
# ---------------------------------------------------------------------------

def _atlas_ticks(ax):
    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True, length=7,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True, length=4,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def _make_main_ratio_axes(figsize, n_ratios=1):
    scaled_figsize = (figsize[0] * FIGSIZE_SCALE, figsize[1] * FIGSIZE_SCALE)

    fig = plt.figure(figsize=scaled_figsize, constrained_layout=False)
    height_ratios = [3.2] + [1.0] * n_ratios
    grid = gridspec.GridSpec(
        1 + n_ratios, 1, figure=fig,
        height_ratios=height_ratios, hspace=0.0,
    )
    main_ax = fig.add_subplot(grid[0])
    ratio_axes = [
        fig.add_subplot(grid[i + 1], sharex=main_ax) for i in range(n_ratios)
    ]

    fig.subplots_adjust(**SUBPLOT_MARGINS)

    _atlas_ticks(main_ax)
    for ax in ratio_axes:
        _atlas_ticks(ax)

    main_ax.tick_params(labelbottom=False)
    for ax in ratio_axes[:-1]:
        ax.tick_params(labelbottom=False)

    if n_ratios == 1:
        return fig, main_ax, ratio_axes[0]
    return fig, main_ax, ratio_axes


def _bin_width_label(bins, log_bins, discrete, density):
    if density:
        return "Norm. to unit area"
    if discrete or log_bins:
        return "Events"
    width = bins[1] - bins[0]
    return rf"Events / {width:.2g}"


def _dup_last(a):
    return np.append(a, a[-1])


def _draw_data(ax, bins, y, err, label, zorder, color="black"):
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    obj = ax.errorbar(
        bin_centers, y, yerr=err,
        fmt="o", color=color, ms=4, elinewidth=1.2,
        capsize=0, label=label, zorder=zorder,
    )
    return obj


def _draw_mc_filled(ax, bins, y, err, color, label, zorder, alpha=0.25):
    band_obj = ax.fill_between(
        bins,
        _dup_last(y - err), _dup_last(y + err),
        step="post", facecolor=color, alpha=alpha, zorder=zorder,
    )
    (line_obj,) = ax.step(
        bins, _dup_last(y), where="post",
        color=color, lw=1.4, label=label, zorder=zorder + 0.1,
    )
    return (line_obj, band_obj)


def _draw_curve_line(ax, bins, y, err, color, label, zorder, ls="-", lw=1.4):
    band_obj = ax.fill_between(
        bins,
        _dup_last(y - err), _dup_last(y + err),
        step="post", facecolor=color, alpha=0.25, zorder=zorder,
    )
    (line_obj,) = ax.step(
        bins, _dup_last(y), where="post",
        color=color, lw=lw, ls=ls, label=label, zorder=zorder + 0.1,
    )
    return (line_obj, band_obj)


def _draw_ratio_data(ax, bins, ratio, ratio_err, color="black"):
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax.errorbar(
        bin_centers, ratio, yerr=ratio_err,
        fmt="o", color=color, ms=4, elinewidth=1.2, capsize=0, zorder=5,
    )


def _draw_ratio_curve(ax, bins, ratio, ratio_err, color, ls="-", lw=1.4, zorder=3):
    ax.fill_between(
        bins,
        _dup_last(ratio - ratio_err), _dup_last(ratio + ratio_err),
        step="post", facecolor=color, alpha=0.25, zorder=zorder,
    )
    ax.step(
        bins, _dup_last(ratio), where="post",
        color=color, lw=lw, ls=ls, zorder=zorder + 0.1,
    )


def _compute_pull(y, err, y_exp, err_exp, density):
    """Per-bin pull of a curve relative to the data/exp histogram:
        pull = (curve - exp) / sqrt(err_curve^2 + err_exp^2)
    computed on unit-area-normalized histograms when density=True (to
    match the same convention used for the chi2/N summary statistic),
    or on raw counts otherwise. Returns an array with one entry per bin
    (NaN where undefined, e.g. both histograms empty in that bin)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if density:
            y_n = y / y.sum()
            err_n = err / y.sum()
            y_exp_n = y_exp / y_exp.sum()
            err_exp_n = err_exp / y_exp.sum()
            pull = (y_n - y_exp_n) / np.sqrt(err_n ** 2 + err_exp_n ** 2)
        else:
            pull = (y - y_exp) / np.sqrt(err ** 2 + err_exp ** 2)
    return pull


def _draw_pull_data_marker(ax, bins, pull, color="black"):
    """Draw a curve's pull as markers connected by a thin line at bin
    centers -- used for every non-data curve in the pull panel."""
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    finite = np.isfinite(pull)
    ax.plot(
        bin_centers[finite], pull[finite],
        marker="o", ms=3, lw=1.2, color=color, zorder=4,
    )


def _compute_bins(exp, sim, xlims, qlims, discrete, log_bins, num_bins,
                   logx, exp_weights, sim_weights, weights_list):
    exp_mask = np.isfinite(exp)
    sim_mask = np.isfinite(sim)
    if exp_weights is not None:
        exp_weights = exp_weights[exp_mask]
    if sim_weights is not None:
        sim_weights = sim_weights[sim_mask]
    exp = exp[exp_mask]
    sim = sim[sim_mask]
    weights_list = [ws[..., sim_mask] for ws in weights_list]

    if logx or log_bins:
        exp_pos = exp > 0
        sim_pos = sim > 0
        exp = exp[exp_pos]
        sim = sim[sim_pos]
        if exp_weights is not None:
            exp_weights = exp_weights[exp_pos]
        if sim_weights is not None:
            sim_weights = sim_weights[sim_pos]
        weights_list = [ws[..., sim_pos] for ws in weights_list]

    lo, hi = xlims or np.quantile(np.hstack([sim, exp]), qlims)
    if log_bins:
        lo = max(lo, 1e-10)
    bins = (
        np.arange(lo, hi + discrete + 1, discrete) - discrete / 2
        if discrete
        else np.logspace(np.log10(lo), np.log10(hi), num_bins)
        if log_bins
        else np.linspace(lo, hi, num_bins)
    )
    return exp, sim, exp_weights, sim_weights, weights_list, bins


# ---------------------------------------------------------------------------
# plot_reweighting
# ---------------------------------------------------------------------------

def plot_reweighting(
    exp,
    sim,
    weights_list,
    variance_list,
    names_list,
    xlabel,
    figsize,
    num_bins=45,
    discrete=False,
    log_bins=False,
    title=None,
    logx=False,
    logy=False,
    qlims=(0.005, 0.995),
    xlims=None,
    quantiles_from_sim=False,
    name_exp="Herwig Pseudo-Data",
    name_sim="MC Simulation Pythia",
    show_sim=True,
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    pull_lims=(-5, 5),
    density=False,
    add_chi2=True,
    add_legend=True,
    exp_weights=None,
    sim_weights=None,
    colors=None,
    atlas_label:   str  = "Simulation",
    atlas_subtext: str  = "Work in progress",
    atlas_second:  str  = None,
    add_atlas:     bool = True,
):
    exp, sim, exp_weights, sim_weights, weights_list, bins = _compute_bins(
        exp, sim, xlims, qlims, discrete, log_bins, num_bins,
        logx, exp_weights, sim_weights, weights_list,
    )

    fig, main_ax, (ratio_ax, pull_ax) = _make_main_ratio_axes(figsize, n_ratios=2)
    ratio_ax.axhline(1.0, color="gray", lw=1.0, zorder=0)
    pull_ax.axhline(0.0, color="gray", lw=1.0, zorder=0)

    # --- data counts -----------------------------------------------------
    if exp_weights is None:
        y_exp = np.histogram(exp, bins=bins)[0]
        err_exp = np.sqrt(y_exp)
    else:
        y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
        sum_w2s = binned_statistic(exp, exp_weights ** 2, "sum", bins=bins)[0]
        err_exp = np.sqrt(sum_w2s)

    # --- sim counts ------------------------------------------------------
    if sim_weights is None:
        y_sim = np.histogram(sim, bins=bins)[0]
        err_sim = np.sqrt(y_sim)
    else:
        y_sim = np.histogram(sim, bins=bins, weights=sim_weights)[0]
        sum_w2s = binned_statistic(sim, sim_weights ** 2, "sum", bins=bins)[0]
        err_sim = np.sqrt(sum_w2s)

    if add_chi2:
        pull_sim_for_chi2 = _compute_pull(y_sim, err_sim, y_exp, err_exp, density)
        nonempty_sim = (y_sim != 0) & (y_exp != 0)
        chi2_sim = (pull_sim_for_chi2[nonempty_sim] ** 2).sum() / num_bins
        name_sim_label = f"{name_sim} ($\\chi^2$/N={chi2_sim:.2f})"
    else:
        name_sim_label = name_sim

    # --- weighted (reweighted) counts and errors --------------------------
    y_rews, err_rews = [], []
    for ws, vs in zip(weights_list, variance_list):
        if ws.ndim > 1:
            y_rews.append(np.histogram(sim, bins=bins, weights=ws.mean(0))[0])
            sum_w2s = binned_statistic(sim, (ws ** 2).mean(0), "sum", bins=bins)[0]
            err = np.sqrt(sum_w2s)
        else:
            if vs is None:
                vs = np.zeros_like(ws)
            mom1 = np.exp(np.log(ws) + vs / 2)
            y_rews.append(np.histogram(sim, bins=bins, weights=mom1)[0])
            mom2 = np.exp(2 * (np.log(ws) + vs))
            sum_w2s = binned_statistic(sim, mom2, "sum", bins=bins)[0]
            err = np.sqrt(sum_w2s)
        err_rews.append(err)

    ys = (y_exp, y_sim, *y_rews)
    errs = (err_exp, err_sim, *err_rews)
    labels_rew = names_list
    labels = (name_exp, name_sim_label, *names_list)

    colors = colors or ["#3B4CC0", "#B22222", "#009826", "C1", "C3", "#13b2b7"]
    denom = ys[denom_idx]

    for i, (y, err, label, color) in enumerate(zip(ys, errs, labels, colors)):

        if (i == 1) and not show_sim:
            continue

        if add_chi2 and (label in labels_rew):
            pull_for_chi2 = _compute_pull(y, err, y_exp, err_exp, density)
            nonempty = (y != 0) & (y_exp != 0)
            chi2 = (pull_for_chi2[nonempty] ** 2).sum() / num_bins
            label += f" ($\\chi^2$/N={chi2:.2f})"

        scale = (
            1 if not density
            else 1 / y_sim.sum() if (label in labels_rew or (i == 1))
            else 1 / y.sum()
        )
        ratio_scale = denom.sum() / y.sum() if density else 1

        y_s, err_s = y * scale, err * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom > 0, y / denom * ratio_scale, np.nan)
            ratio_err = np.where(denom > 0, err / denom * ratio_scale, np.nan)

        # pull relative to data/exp, drawn for every curve except data itself
        pull = _compute_pull(y, err, y_exp, err_exp, density) if i > 0 else None

        if i == 0:
            _draw_data(main_ax, bins, y_s, err_s, label, zorder=6, color="black")
            _draw_ratio_data(ratio_ax, bins, ratio, ratio_err, color="black")

        elif i == 1:
            _draw_mc_filled(main_ax, bins, y_s, err_s, color, label, zorder=2)
            _draw_ratio_curve(ratio_ax, bins, ratio, ratio_err, color, zorder=2)
            _draw_pull_data_marker(pull_ax, bins, pull, color=color)

        else:
            _draw_curve_line(main_ax, bins, y_s, err_s, color, label, zorder=4)
            _draw_ratio_curve(ratio_ax, bins, ratio, ratio_err, color, zorder=3)
            _draw_pull_data_marker(pull_ax, bins, pull, color=color)

    if logx:
        main_ax.semilogx()
        ratio_ax.semilogx()
        pull_ax.semilogx()
    if logy:
        main_ax.semilogy()

    ratio_ax.set_ylim(*ratio_lims)
    pull_ax.set_ylim(*pull_lims)
    main_ax.set_xlim(bins[0], bins[-1])
    ratio_ax.set_xlim(bins[0], bins[-1])
    pull_ax.set_xlim(bins[0], bins[-1])

    main_ax.set_ylabel(_bin_width_label(bins, log_bins, discrete, density), fontsize=AXIS_LABEL_FONTSIZE)
    ratio_ax.set_ylabel(r"$\frac{\mathrm{Data}}{\mathrm{Pred.}}$", fontsize=AXIS_LABEL_FONTSIZE)
    pull_ax.set_ylabel("Pull", fontsize=AXIS_LABEL_FONTSIZE)
    pull_ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)

    if add_legend:
        _legend_with_data_first(main_ax, name_exp, loc="upper right", **LEGEND_KWARGS)

    if title is not None:
        main_ax.set_title(title, loc="right", fontsize=12)

    if add_atlas:
        add_atlas_label(
            main_ax,
            atlas_label=atlas_label,
            subtext=atlas_subtext,
            atlas_second=atlas_second,
        )

    return fig, (main_ax, ratio_ax, pull_ax)


# ---------------------------------------------------------------------------
# plot_reweighting_ensemble
# ---------------------------------------------------------------------------

def plot_reweighting_ensemble(
    exp,
    sim,
    weights_list,
    variance_list,
    names_list,
    xlabel,
    figsize,
    num_bins=45,
    discrete=False,
    log_bins=False,
    title=None,
    logx=False,
    logy=False,
    qlims=(0.005, 0.995),
    xlims=None,
    quantiles_from_sim=False,
    name_exp="Herwig Pseudo-Data",
    name_sim="MC Simulation Pythia",
    show_sim=True,
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    pull_lims=(-5, 5),
    density=False,
    add_chi2=True,
    exp_weights=None,
    add_legend=False,
    colors=None,
    atlas_label:   str  = "Simulation",
    atlas_subtext: str  = "Work in progress",
    atlas_second:  str  = None,
    add_atlas:     bool = True,
):
    exp, sim, exp_weights, _, weights_list, bins = _compute_bins(
        exp, sim, xlims, qlims, discrete, log_bins, num_bins,
        logx, exp_weights, None, weights_list,
    )

    fig, main_ax, (ratio_ax, pull_ax) = _make_main_ratio_axes(figsize, n_ratios=2)
    ratio_ax.axhline(1.0, color="gray", lw=1.0, zorder=0)
    pull_ax.axhline(0.0, color="gray", lw=1.0, zorder=0)

    if exp_weights is None:
        exp_weights = np.ones_like(exp)

    y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
    sum_w2s = binned_statistic(exp, exp_weights ** 2, "sum", bins=bins)[0]
    if density:
        norm_exp = y_exp.sum()
        y_exp = y_exp / norm_exp
        sum_w2s = sum_w2s / (norm_exp ** 2)
    err_exp = np.sqrt(sum_w2s)

    y_sim = np.histogram(sim, bins=bins)[0]
    err_sim = np.sqrt(y_sim)
    if density:
        norm_sim = y_sim.sum()
        err_sim = err_sim / norm_sim
        y_sim = y_sim / norm_sim

    y_rews, err_rews = [], []
    for ws, vs in zip(weights_list, variance_list):
        assert ws.ndim > 1
        all_y = np.apply_along_axis(
            lambda w: np.histogram(sim, bins=bins, weights=w)[0], 1, ws
        )
        all_sum_w2s = np.apply_along_axis(
            lambda w: binned_statistic(sim, w ** 2, "sum", bins=bins)[0], 1, ws
        )
        if density:
            all_y = all_y / norm_sim
            all_sum_w2s = all_sum_w2s / (norm_sim ** 2)

        y_rew = np.quantile(all_y, 0.5, axis=0)
        var_across = np.var(all_y, axis=0)
        mean_within = np.mean(all_sum_w2s, axis=0)
        err = np.sqrt(var_across + mean_within)

        y_rews.append(y_rew)
        err_rews.append(err)

    ys = (y_exp, y_sim, *y_rews)
    errs = (err_exp, err_sim, *err_rews)
    labels_rew = names_list
    labels = (name_exp, name_sim, *names_list)

    colors = colors or ["#3B4CC0", "#B22222", "#009826", "C1", "C3", "#13b2b7"]
    denom = ys[denom_idx]

    for i, (y, err, label, color) in enumerate(zip(ys, errs, labels, colors)):

        if (i == 1) and not show_sim:
            continue

        ratio_scale = denom.sum() / y.sum() if density else 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom > 0, y / denom * ratio_scale, np.nan)
            ratio_err = np.where(denom > 0, err / denom * ratio_scale, np.nan)

        pull = _compute_pull(y, err, y_exp, err_exp, density=False) if i > 0 else None
        # note: y_exp/y_sim/y_rew here are ALREADY density-normalized above
        # when density=True, so pass density=False to _compute_pull to
        # avoid double-normalizing

        if i == 0:
            _draw_data(main_ax, bins, y, err, label, zorder=6, color="black")
            _draw_ratio_data(ratio_ax, bins, ratio, ratio_err, color="black")
        elif i == 1:
            _draw_mc_filled(main_ax, bins, y, err, color, label, zorder=2)
            _draw_ratio_curve(ratio_ax, bins, ratio, ratio_err, color, zorder=2)
            _draw_pull_data_marker(pull_ax, bins, pull, color=color)
        else:
            _draw_curve_line(main_ax, bins, y, err, color, label, zorder=4)
            _draw_ratio_curve(ratio_ax, bins, ratio, ratio_err, color, zorder=3)
            _draw_pull_data_marker(pull_ax, bins, pull, color=color)

    if logx:
        main_ax.semilogx()
        ratio_ax.semilogx()
        pull_ax.semilogx()
    if logy:
        main_ax.semilogy()

    ratio_ax.set_ylim(*ratio_lims)
    pull_ax.set_ylim(*pull_lims)
    main_ax.set_xlim(bins[0], bins[-1])
    ratio_ax.set_xlim(bins[0], bins[-1])
    pull_ax.set_xlim(bins[0], bins[-1])

    main_ax.set_ylabel(_bin_width_label(bins, log_bins, discrete, density), fontsize=AXIS_LABEL_FONTSIZE)
    ratio_ax.set_ylabel(r"$\frac{\mathrm{Data}}{\mathrm{Pred.}}$", fontsize=AXIS_LABEL_FONTSIZE)
    pull_ax.set_ylabel("Pull", fontsize=AXIS_LABEL_FONTSIZE)
    pull_ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)

    if add_legend:
        _legend_with_data_first(main_ax, name_exp, loc="upper right", **LEGEND_KWARGS)

    if title is not None:
        main_ax.set_title(title, loc="right", fontsize=12)

    if add_atlas:
        add_atlas_label(
            main_ax,
            atlas_label=atlas_label,
            subtext=atlas_subtext,
            atlas_second=atlas_second,
        )

    return fig, (main_ax, ratio_ax, pull_ax)


# ---------------------------------------------------------------------------
# plot_reweighting_multi_ratio
# ---------------------------------------------------------------------------

def plot_reweighting_multi_ratio(
    exp,
    sim,
    weights_list,
    variance_list,
    names_list,
    ratio_idx,
    ratio_names,
    xlabel,
    figsize,
    num_bins=45,
    discrete=False,
    log_bins=False,
    title=None,
    logx=False,
    logy=False,
    qlims=(0.005, 0.995),
    xlims=None,
    quantiles_from_sim=False,
    name_exp="Herwig Pseudo-Data",
    name_sim="MC Simulation Pythia",
    show_sim=True,
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    pull_lims=(-5, 5),
    density=False,
    add_chi2=True,
    exp_weights=None,
    add_legend=False,
    colors=None,
    legend_loc=None,
    ypad=1,
    atlas_label:   str  = "Simulation",
    atlas_subtext: str  = "Work in progress",
    atlas_second:  str  = None,
    add_atlas:     bool = True,
):
    num_ratios = int(np.max(ratio_idx) + 1)
    # one extra panel at the bottom, dedicated to the pull of every
    # non-data curve relative to data (aggregated across all curves,
    # regardless of which ratio_idx group they belong to)
    total_panels = num_ratios + 1
    pull_panel_idx = num_ratios  # last panel

    exp, sim, exp_weights, _, weights_list, bins = _compute_bins(
        exp, sim, xlims, qlims, discrete, log_bins, num_bins,
        logx, exp_weights, None, weights_list,
    )

    fig, main_ax, ratio_axes = _make_main_ratio_axes(figsize, n_ratios=total_panels)
    for ax in ratio_axes[:num_ratios]:
        ax.axhline(1.0, color="gray", lw=1.0, zorder=0)
    pull_ax = ratio_axes[pull_panel_idx]
    pull_ax.axhline(0.0, color="gray", lw=1.0, zorder=0)

    if exp_weights is None:
        exp_weights = np.ones_like(exp)

    y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
    sum_w2s = binned_statistic(exp, exp_weights ** 2, "sum", bins=bins)[0]
    if density:
        norm_exp = y_exp.sum()
        y_exp = y_exp / norm_exp
        sum_w2s = sum_w2s / (norm_exp ** 2)
    err_exp = np.sqrt(sum_w2s)

    y_sim = np.histogram(sim, bins=bins)[0]
    err_sim = np.sqrt(y_sim)
    if density:
        norm_sim = y_sim.sum()
        err_sim = err_sim / norm_sim
        y_sim = y_sim / norm_sim

    y_rews, err_rews = [], []
    for ws, vs in zip(weights_list, variance_list):
        assert ws.ndim > 1
        all_y = np.apply_along_axis(
            lambda w: np.histogram(sim, bins=bins, weights=w)[0], 1, ws
        )
        all_sum_w2s = np.apply_along_axis(
            lambda w: binned_statistic(sim, w ** 2, "sum", bins=bins)[0], 1, ws
        )
        if density:
            all_y = all_y / norm_sim
            all_sum_w2s = all_sum_w2s / (norm_sim ** 2)

        central = np.quantile(all_y, 0.5, axis=0)
        var_across = np.var(all_y, axis=0)
        mean_within = np.mean(all_sum_w2s, axis=0)
        err = np.sqrt(var_across + mean_within)

        y_rews.append(central)
        err_rews.append(err)

    ys = (y_exp, y_sim, *y_rews)
    errs = (err_exp, err_sim, *err_rews)
    labels = (name_exp, name_sim, *names_list)

    colors = colors or ["#3B4CC0", "#B22222", "#009826", "C1", "C3", "#13b2b7"]
    denom = ys[denom_idx]

    ratio_idcs = [np.arange(len(ratio_idx))] * 2 + [[i] for i in ratio_idx]

    for i, (y, err, label, color) in enumerate(zip(ys, errs, labels, colors)):

        if (i == 1) and not show_sim:
            continue

        ratio_scale = denom.sum() / y.sum() if density else 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom > 0, y / denom * ratio_scale, np.nan)
            ratio_err = np.where(denom > 0, err / denom * ratio_scale, np.nan)

        pull = _compute_pull(y, err, y_exp, err_exp, density=False) if i > 0 else None
        # note: y_exp/y_sim/y_rew are already density-normalized above when
        # density=True, so pass density=False to _compute_pull here

        if i == 0:
            _draw_data(main_ax, bins, y, err, label, zorder=6, color="black")
            for ir in range(num_ratios):
                if ir in ratio_idcs[i]:
                    _draw_ratio_data(ratio_axes[ir], bins, ratio, ratio_err, color="black")

        elif i == 1:
            _draw_mc_filled(main_ax, bins, y, err, color, label, zorder=2)
            for ir in range(num_ratios):
                if ir in ratio_idcs[i]:
                    _draw_ratio_curve(ratio_axes[ir], bins, ratio, ratio_err, color, zorder=2)
            _draw_pull_data_marker(pull_ax, bins, pull, color=color)

        else:
            _draw_curve_line(main_ax, bins, y, err, color, label, zorder=4)
            for ir in range(num_ratios):
                if ir in ratio_idcs[i]:
                    _draw_ratio_curve(ratio_axes[ir], bins, ratio, ratio_err, color, zorder=3)
            _draw_pull_data_marker(pull_ax, bins, pull, color=color)

    if logx:
        main_ax.semilogx()
        for ax in ratio_axes:
            ax.semilogx()
    if logy:
        main_ax.semilogy()

    main_ax.set_ylim(
        main_ax.get_ylim()[0],
        main_ax.get_ylim()[1] ** ypad if logy else ypad * main_ax.get_ylim()[1],
    )
    main_ax.set_xlim(bins[0], bins[-1])
    main_ax.set_ylabel(_bin_width_label(bins, log_bins, discrete, density), fontsize=AXIS_LABEL_FONTSIZE)

    for ir in range(num_ratios):
        ratio_axes[ir].set_ylim(*ratio_lims)
        ratio_axes[ir].set_xlim(bins[0], bins[-1])
        ratio_axes[ir].set_ylabel(
            rf"$\frac{{\mathrm{{{ratio_names[ir]}}}}}{{\mathrm{{Data}}}}$",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

    pull_ax.set_ylim(*pull_lims)
    pull_ax.set_xlim(bins[0], bins[-1])
    pull_ax.set_ylabel("Pull", fontsize=AXIS_LABEL_FONTSIZE)

    ratio_axes[-1].set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)

    if add_legend:
        _legend_with_data_first(
            main_ax, name_exp, loc=legend_loc or "upper right", **LEGEND_KWARGS
        )

    if title is not None:
        main_ax.set_title(title, loc="right", fontsize=12)

    if add_atlas:
        add_atlas_label(
            main_ax,
            atlas_label=atlas_label,
            subtext=atlas_subtext,
            atlas_second=atlas_second,
        )

    return fig, (main_ax, ratio_axes)