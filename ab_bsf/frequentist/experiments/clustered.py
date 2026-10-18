"""
Cluster-randomised designs: when the unit of randomisation (e.g. a coin, a
market, a day) differs from the unit of analysis (a user or session).
"""

from typing import Dict, Tuple

import numpy as np


def design_effect(cluster_size: float, icc: float, cluster_size_cv: float = 0.0) -> float:
    """
    Variance inflation from clustering.

    DE = 1 + (m - 1) * ICC                          (equal cluster sizes)
    DE = 1 + ((cv^2 + 1) * m - 1) * ICC             (unequal sizes, Eldridge et al. 2006)

    Parameters
    ----------
    cluster_size : mean cluster size m.
    icc : intra-cluster correlation.
    cluster_size_cv : coefficient of variation of the cluster sizes.
    """
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if not 0 <= icc <= 1:
        raise ValueError("icc must be in [0, 1]")
    return float(1 + ((cluster_size_cv**2 + 1) * cluster_size - 1) * icc)


def clustered_sample_size(
    n_control: int,
    n_treatment: int,
    cluster_size: float,
    icc: float,
    cluster_size_cv: float = 0.0,
) -> Tuple[int, int, Dict[str, float]]:
    """
    Inflate individually-randomised sample sizes (e.g. from any design class's
    ``sample_size_given_power``) for cluster randomisation.

    Returns
    -------
    (n_control, n_treatment, info) where sizes are rounded up to whole clusters and
    ``info`` has the design effect and the number of clusters per arm.
    """
    de = design_effect(cluster_size, icc, cluster_size_cv)
    k_c = int(np.ceil(n_control * de / cluster_size))
    k_t = int(np.ceil(n_treatment * de / cluster_size))
    info = {
        "design_effect": de,
        "n_clusters_control": k_c,
        "n_clusters_treatment": k_t,
        "unadjusted_n_control": n_control,
        "unadjusted_n_treatment": n_treatment,
    }
    return int(np.ceil(k_c * cluster_size)), int(np.ceil(k_t * cluster_size)), info
