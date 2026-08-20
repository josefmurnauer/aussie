import torch


def _draw_training_bootstrap_weights(n_data, n_replicas, gen, distribution="poisson", lognormal_sigma=1.0):
    """
    Draw (n_data, n_replicas) i.i.d. multiplicative bootstrap weights
    for TRAINING-time replica generation, mean exactly 1.0 by
    construction, matching the same distributional choice available for
    evaluation-time resampling (see
    src.utils.uncertainty._draw_bootstrap_multipliers).

      - "poisson":   w ~ Poisson(lambda=1). Classic Poisson bootstrap.
      - "lognormal": w ~ LogNormal(mu, sigma), mu = -sigma^2/2 so that
                     E[w] = 1 exactly.
    """
    if distribution == "poisson":
        return torch.poisson(torch.ones(n_data, n_replicas), generator=gen)
    elif distribution == "lognormal":
        mu = -0.5 * lognormal_sigma ** 2
        normal_draws = torch.normal(
            mean=mu, std=lognormal_sigma,
            size=(n_data, n_replicas), generator=gen,
        )
        return torch.exp(normal_draws)
    else:
        raise ValueError(f"Unknown bootstrap distribution '{distribution}'")


def attach_replica_bootstrap(
    dset, n_replicas: int, seed: int, log=None,
    distribution: str = "poisson", lognormal_sigma: float = 1.0,
):
    """
    Attach per-event REPLICA bootstrap weights (linear, shape (N, K)) to
    a raw (pre-split) dataset, for use in a K-way ensembled classifier
    training run.

    Multiplicities are drawn independently per (event, replica) pair
    for DATA-labeled events ONLY (labels == 1); sim-labeled events
    (labels == 0) get weight 1 for every replica.

    `distribution` ("poisson" default, or "lognormal") and
    `lognormal_sigma` control the training-time bootstrap distribution,
    matching the equivalent evaluation-time knobs in
    src.utils.uncertainty.bootstrap_histogram_covariance -- IMPORTANT:
    these are SEPARATE settings from `uncertainty.distribution`/
    `uncertainty.lognormal_sigma` in the config; setting the latter does
    NOT automatically change what's used here. Set BOTH consistently if
    you want the same distributional family used everywhere.

    Requires dset.labels to be set (label==1 identifies the "data"
    population; label==0 identifies "sim").
    """
    assert dset.labels is not None, (
        "attach_replica_bootstrap requires dset.labels to identify the "
        "data (label==1) vs. sim (label==0) populations"
    )

    n = len(dset)
    is_data = (dset.labels == 1)
    n_data = int(is_data.sum().item())

    gen = torch.Generator(device="cpu").manual_seed(seed)
    draws = _draw_training_bootstrap_weights(
        n_data, n_replicas, gen, distribution=distribution, lognormal_sigma=lognormal_sigma,
    )

    weights = torch.ones((n, n_replicas), dtype=torch.float32)
    weights[is_data] = draws.float()

    dset.replica_weights = weights

    if log is not None:
        log.info(
            f"Attached replica bootstrap weights: n_replicas={n_replicas}, "
            f"seed={seed}, n_data_events={n_data}, distribution={distribution}"
            f"{f' (sigma={lognormal_sigma})' if distribution == 'lognormal' else ''}, "
            f"mean multiplier={draws.mean().item():.4f} (target: 1.0)."
        )

    return dset