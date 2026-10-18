"""Multiple-comparison corrections for results dicts (several arms or many KPIs)."""

from typing import Dict, Literal

from statsmodels.stats.multitest import multipletests

__all__ = ["adjust_p_values"]


def adjust_p_values(
    results: Dict[str, Dict[str, float]],
    p_value_key: str = "p_values",
    method: Literal["bonferroni", "holm", "fdr_bh"] = "holm",
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Add ``<p_value_key>_adjusted`` and ``reject`` to every comparison in a results
    dict (e.g. from ``t_test``). Use "holm" to control the family-wise error rate
    across treatment arms, "fdr_bh" for exploratory KPIs.
    """
    keys = list(results)
    reject, adj, _, _ = multipletests([results[k][p_value_key] for k in keys], alpha=alpha, method=method)
    return {k: {**results[k], f"{p_value_key}_adjusted": float(a), "reject": bool(r)} for k, a, r in zip(keys, adj, reject)}
