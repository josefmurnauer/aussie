"""
A/B response-matrix unfolding, with error propagation, for comparing
against the unregularized pseudo-inverse approach (compute_tunfold_result
in src.utils.tunfold) and against OmniFold/AUSSIE's unbinned approach.

Given the joint (reco, truth) sim histogram M[i,j] (i=reco bin, j=truth
bin, weighted by sim event weights):

    A[i,j] = M[i,j] / sum_i M[i,j] = P(reco=i | truth=j)
             -- column-normalized. Forward folding: reco = A @ truth.
             Unfolding via A requires INVERSION (pinv), which is exactly
             the unregularized "naive TUnfold" approach already
             implemented in compute_tunfold_result.

    B[j,i] = M[i,j] / sum_j M[i,j] = P(truth=j | reco=i)
             -- row-normalized (over truth, for fixed reco bin).
             Direct one-shot correction: truth_hat = B @ reco_data.
             No inversion needed -- automatically non-negative for
             non-negative reco_data, since B's entries are probabilities.
             This is exactly the zeroth iteration of the classic
             D'Agostini-style "Bayesian"/matrix correction: well-posed
             by construction, but biased toward the sim's truth PRIOR
             in bins with significant migration ambiguity.

NOTE: B is NOT A.T in general -- both come from the same M, but
normalized along different axes.

The "star" variants (A*, B*) are built from a SIM POPULATION reweighted
by a converged OmniFold/AUSSIE unf run's final per-event weight -- i.e.
ONLY the response-matrix construction changes; the observed pseudodata
(reco_data, its variance) is IDENTICAL across A/B/A*/B*. Comparing A to
A* answers: "was A's instability mostly a prior/migration-shape
mismatch (fixed once sim already resembles data), or is it intrinsic to
the detector's migrations for this observable (persists even given a
near-perfect prior)?"
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.tunfold import make_response_bins


@dataclass
class ABResult:
    weight: np.ndarray
    logvar: np.ndarray
    x_hat: np.ndarray                 # central truth-level histogram (pre-clip)
    truth_edges: np.ndarray
    cov: np.ndarray
    corr: np.ndarray
    cond_number: float
    n_negative_bins: int
    matrix_kind: str                  # "A" or "B"
    reweighted: bool                  # False for A/B, True for A*/B*
    reco_edges: Optional[np.ndarray] = field(default=None)


def build_joint_histogram(
    x_sim_v, z_sim_v, sim_w, obs_x, obs_z, num_bins_truth, reco_bin_factor,
    truth_edges=None, reco_edges=None,
):
    """Build the joint (reco, truth) histogram M[i,j] from sim events,
    weighted by `sim_w`. Bin edges may be supplied explicitly so that
    A/B and A*/B* share IDENTICAL binning despite using different sim
    weights for the joint histogram itself -- if not supplied, they are
    derived from x_sim_v/z_sim_v via make_response_bins."""
    num_bins_reco = int(round(num_bins_truth * reco_bin_factor))

    if truth_edges is None:
        truth_edges = make_response_bins(z_sim_v, obs_z, num_bins_truth)
    if reco_edges is None:
        reco_edges = make_response_bins(x_sim_v, obs_x, num_bins_reco)

    M, _, _ = np.histogram2d(
        x_sim_v, z_sim_v, bins=[reco_edges, truth_edges], weights=sim_w
    )
    return M, reco_edges, truth_edges


def matrices_from_joint(M):
    """
    A[i,j] = P(reco=i | truth=j)   shape (n_reco, n_truth)
    B[j,i] = P(truth=j | reco=i)   shape (n_truth, n_reco)
    """
    col_sums = M.sum(axis=0)  # sum over reco, per truth bin
    A = np.divide(M, col_sums, out=np.zeros_like(M), where=col_sums > 0)

    row_sums = M.sum(axis=1, keepdims=True)  # sum over truth, per reco bin
    P_truth_given_reco = np.divide(
        M, row_sums, out=np.zeros_like(M), where=row_sums > 0
    )  # shape (n_reco, n_truth)
    B = P_truth_given_reco.T  # shape (n_truth, n_reco)

    return A, B


def _condition_number(matrix):
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")


def propagate_and_convert(
    reco_op,
    reco_data,
    var_reco,
    z_sim_v,
    truth_edges,
    sim_w_raw,
    display_rescale,
    clip_negative,
    weight_floor,
    matrix_kind,
    reweighted,
    cond_number,
):
    """
    Apply a (n_truth x n_reco) linear operator `reco_op` to the observed
    reco-level histogram (rescaled by `display_rescale`, a single global
    constant calibrating the result to Sim's RAW total -- consistent
    with the Classifier/AUSSIE plotting convention, since these
    estimators are otherwise naturally calibrated to DATA's total) with
    UNCORRELATED reco-level statistical uncertainty `var_reco` (diagonal
    V_reco):

        truth_hat = reco_op @ reco_data
        cov       = reco_op @ diag(var_reco) @ reco_op.T

    For matrix_kind="A", pass reco_op = pinv(A) (unregularized
    inversion). For matrix_kind="B", pass reco_op = B directly (no
    inversion -- automatically non-negative for non-negative reco_data).

    The resulting truth-level histogram is converted into a per-sim-
    event weight/log-variance, ALWAYS calibrated against the RAW
    (un-reweighted) sim population `sim_w_raw` -- both in the
    numerator-matching (sim_truth_hist) and the final weight -- so
    A/B/A*/B* are all expressed as absolute weights over the SAME
    underlying raw sim events, directly comparable to each other and to
    Classifier/AUSSIE/TUnfold curves in the same plot, regardless of
    which weights were used to build the response matrix itself.
    """
    n_truth_bins = len(truth_edges) - 1

    reco_data_scaled = reco_data * display_rescale
    var_reco_scaled = var_reco * display_rescale ** 2

    x_hat = reco_op @ reco_data_scaled
    cov = reco_op @ np.diag(var_reco_scaled) @ reco_op.T

    n_negative_bins = int((x_hat < 0).sum())

    var_x = np.clip(np.diag(cov), 0, None)
    std_x = np.sqrt(var_x)
    denom = np.outer(std_x, std_x)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, np.where(std_x > 0, 1.0, 0.0))

    x_hat_central = x_hat.copy()
    x_hat_clipped = np.clip(x_hat, 0, None) if clip_negative else x_hat

    sim_truth_hist, _ = np.histogram(z_sim_v, bins=truth_edges, weights=sim_w_raw)
    scale = np.divide(
        x_hat_clipped, sim_truth_hist, out=np.zeros_like(x_hat_clipped),
        where=sim_truth_hist > 0,
    )

    bin_idx = np.digitize(z_sim_v, truth_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_truth_bins)

    per_event_scale = np.zeros_like(z_sim_v, dtype=np.float64)
    per_event_scale[valid] = scale[bin_idx[valid]]
    weight = (per_event_scale * sim_w_raw).astype(np.float64)
    weight = np.clip(weight, weight_floor, None)

    sum_w2_bin = np.zeros(n_truth_bins, dtype=np.float64)
    np.add.at(sum_w2_bin, bin_idx[valid], weight[valid] ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(var_x, sum_w2_bin, out=np.zeros_like(var_x), where=sum_w2_bin > 0)
        logvar_bin = np.where(ratio > 0, 0.5 * np.log(ratio), 0.0)

    per_event_logvar = np.zeros_like(z_sim_v, dtype=np.float64)
    per_event_logvar[valid] = logvar_bin[bin_idx[valid]]

    return ABResult(
        weight=weight.astype(np.float32),
        logvar=per_event_logvar.astype(np.float32),
        x_hat=x_hat_central,
        truth_edges=truth_edges,
        cov=cov,
        corr=corr,
        cond_number=cond_number,
        n_negative_bins=n_negative_bins,
        matrix_kind=matrix_kind,
        reweighted=reweighted,
    )


def compute_ab_results(
    x_sim_v,
    x_dat_v,
    z_sim_v,
    sim_w_raw,
    sim_w_star,
    exp_w,
    obs_x,
    obs_z,
    num_bins_truth,
    reco_bin_factor=2,
    clip_negative=True,
    weight_floor=1e-12,
):
    """
    Compute all four (A, B, A*, B*) unfolded estimates for one
    observable, sharing IDENTICAL bin edges (so they remain directly
    comparable). A/B use `sim_w_raw` for the response matrix; A*/B* use
    `sim_w_star` (a converged OmniFold/AUSSIE run's FINAL absolute
    per-event weight -- see load_omnifold_final_weight) for the
    response matrix ONLY. Pseudodata (x_dat_v, exp_w) is identical
    across all four. Per-event weight/logvar conversion is always
    calibrated against `sim_w_raw` for all four (see
    propagate_and_convert).

    Returns
    -------
    results : dict[str, ABResult]         keys: "A", "B", "A_star", "B_star"
    joint_histograms : dict[str, ndarray]  keys: "A", "B", "A_star", "B_star"
                                            (raw M matrix used to build that entry --
                                            "A"/"B" share the same M, as do "A_star"/"B_star")
    """
    num_bins_reco = int(round(num_bins_truth * reco_bin_factor))

    truth_edges = make_response_bins(z_sim_v, obs_z, num_bins_truth)
    reco_edges = make_response_bins(
        np.hstack([x_sim_v, x_dat_v]), obs_x, num_bins_reco
    )

    exp_w_use = exp_w if exp_w is not None else np.ones_like(x_dat_v)
    reco_data, _ = np.histogram(x_dat_v, bins=reco_edges, weights=exp_w_use)
    var_reco, _ = np.histogram(x_dat_v, bins=reco_edges, weights=exp_w_use ** 2)

    display_rescale = sim_w_raw.sum() / reco_data.sum() if reco_data.sum() > 0 else 1.0

    results = {}
    joint_histograms = {}

    for tag, sim_w_for_matrix, reweighted in (
        ("A", sim_w_raw, False),
        ("B", sim_w_raw, False),
        ("A_star", sim_w_star, True),
        ("B_star", sim_w_star, True),
    ):
        M, _, _ = build_joint_histogram(
            x_sim_v, z_sim_v, sim_w_for_matrix, obs_x, obs_z,
            num_bins_truth, reco_bin_factor,
            truth_edges=truth_edges, reco_edges=reco_edges,
        )
        joint_histograms[tag] = M
        A, B = matrices_from_joint(M)

        if tag.startswith("A"):
            cond = _condition_number(A)
            reco_op = np.linalg.pinv(A)
            kind = "A"
        else:
            cond = _condition_number(B)
            reco_op = B
            kind = "B"

        result = propagate_and_convert(
            reco_op, reco_data, var_reco, z_sim_v, truth_edges,
            sim_w_raw, display_rescale, clip_negative, weight_floor,
            matrix_kind=kind, reweighted=reweighted, cond_number=cond,
        )
        result.reco_edges = reco_edges
        results[tag] = result

    return results, joint_histograms


def load_omnifold_final_weight(omnifold_unf_dir):
    """
    Load the FINAL absolute per-sim-event weight from a converged
    OmniFold/AUSSIE unf run's predictions_test.npz. This is
    np.exp(lw_z_sim), ensemble-averaged. IMPORTANT: lw_z_sim ALREADY
    includes the base MC sample weight (added at evaluate() time via
    `+ lw_sample`), so this must be used DIRECTLY as the sim weight when
    building a response matrix -- do NOT multiply it again by the raw
    MC sample weight, or the base weight will be double-counted.
    """
    record = np.load(os.path.join(omnifold_unf_dir, "predictions_test.npz"))
    lw_z_sim = record["lw_z_sim"].mean(0)
    return np.exp(lw_z_sim)