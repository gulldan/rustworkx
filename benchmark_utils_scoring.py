# ruff: noqa: E501
import logging
from typing import Any

import numpy as np

from metrics_config import ORDERED_METRIC_PROPERTIES

logger = logging.getLogger(__name__)


def normalize_metric_value(raw_value: Any, metric_prop: dict[str, Any]) -> float:
    """Normalizes a metric value to a [0, 1] scale where 1 is better.

    Handles None, NaN, and applies logic based on 'higher_is_better'
    and known properties of certain metrics (e.g., ARI, modularity).

    Args:
        raw_value: The raw metric value.
        metric_prop: A dictionary containing properties of the metric,
            including 'key' (str, e.g., "_ari"), 'higher_is_better' (bool),
            and potentially others.

    Returns:
        A float representing the normalized score (0.0 to 1.0).
        Returns 0.0 for invalid inputs or if normalization is not applicable.
    """
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return 0.0

    val_f: float
    try:
        val_f = float(raw_value)
    except (ValueError, TypeError):
        logger.warning(
            f"OverallScore: Could not convert raw_value '{raw_value}' to float for metric '{metric_prop.get('key', 'unknown')}'. Treating as 0."
        )
        return 0.0

    higher_is_better: bool | None = metric_prop.get("higher_is_better")
    metric_key_suffix: str | None = metric_prop.get("key")

    if higher_is_better is None or metric_key_suffix is None:
        logger.warning(
            f"OverallScore: 'higher_is_better' or 'key' missing in metric_prop for value '{raw_value}'. Treating as 0."
        )
        return 0.0

    norm_score: float = 0.0

    if higher_is_better:
        if metric_key_suffix == "_ari":  # Range: -1 to 1
            norm_score = (val_f + 1.0) / 2.0
        elif metric_key_suffix == "_modularity":  # Typical range approx -0.5 to 1
            # Normalize from [-0.5, 1.0] to [0, 1.0]
            # Values below -0.5 map to 0, values above 1.0 map to 1.0
            norm_score = max(0.0, min(1.0, (val_f + 0.5) / 1.5 if val_f >= -0.5 else 0.0))
        elif val_f > 0:  # For metrics like NMI, FMI, Purity, GCR, PCP (typically 0-1)
            if val_f <= 1.0:
                norm_score = val_f
            else:  # For metrics like Surprise, Significance, Avg Int Deg (can be > 1)
                # Simple cap at 1.0. More sophisticated normalization might be needed
                # if their typical ranges vary wildly and a simple cap is too crude.
                # For now, assume values > 1 are "very good".
                norm_score = 1.0
        else:  # val_f <= 0 for a "higher_is_better" metric (undesirable)
            norm_score = 0.0
    else:  # lower_is_better
        if metric_key_suffix == "_vi":  # Range: >= 0. Lower is better. 0 is best.
            # 1 / (1 + x) maps [0, inf) to (0, 1], with 0 -> 1.
            norm_score = 1.0 / (1.0 + val_f) if val_f >= 0 else 0.0
        elif metric_key_suffix in ["_conductance", "_cut_ratio"]:  # Range: [0,1]. Lower is better.
            norm_score = 1.0 - val_f if 0.0 <= val_f <= 1.0 else 0.0
        # Add other "lower_is_better" metrics here if specific normalization is needed
        else:  # Default for other "lower is better" (if any)
            norm_score = 1.0 / (1.0 + val_f) if val_f >= 0 else 0.0  # Similar to VI

    return max(0.0, min(1.0, norm_score))  # Ensure result is strictly [0,1]


def calculate_overall_score(metrics_dict: dict[str, Any], algo_prefix: str) -> float:
    """Calculates an overall quality score from various normalized metrics.

    The score is an average of normalized values of metrics defined in
    `ORDERED_METRIC_PROPERTIES` (from `metrics_config.py`) that are
    designated as quality metrics (i.e., not 'info', 'perf', or 'structure' types,
    and have 'higher_is_better' defined).

    Args:
        metrics_dict: A dictionary containing raw metric values, keyed by
            `algo_prefix` + `metric_key_suffix` (e.g., "nx_louvain_ari").
        algo_prefix: The prefix string identifying the algorithm run
            (e.g., "nx_louvain").

    Returns:
        A float representing the overall score (0.0 to 1.0). Returns 0.0
        if no valid metrics are found or if `ORDERED_METRIC_PROPERTIES` is empty.
    """
    if not ORDERED_METRIC_PROPERTIES:
        logger.error("OverallScore: ORDERED_METRIC_PROPERTIES is empty. Cannot calculate score.")
        return 0.0

    normalized_scores: list[float] = []
    # metrics_used_for_score: list[dict[str, Any]] = [] # For debugging

    for metric_prop in ORDERED_METRIC_PROPERTIES:
        metric_key_suffix: str | None = metric_prop.get("key")
        metric_type: str | None = metric_prop.get("type")
        higher_is_better: bool | None = metric_prop.get("higher_is_better")

        # Skip non-quality metrics or those without a clear preference
        if metric_type in ["info", "perf", "structure"] or higher_is_better is None:
            continue
        if not metric_key_suffix:  # Should not happen if config is correct
            continue

        full_metric_key: str = f"{algo_prefix}{metric_key_suffix}"
        raw_value: Any = metrics_dict.get(full_metric_key)

        normalized_score: float = normalize_metric_value(raw_value, metric_prop)
        normalized_scores.append(normalized_score)
        # metrics_used_for_score.append({ # For debugging
        #     "key": full_metric_key,
        #     "raw": raw_value,
        #     "norm": normalized_score,
        #     "hib": higher_is_better
        # })

    if not normalized_scores:
        logger.warning(f"OverallScore: No valid quality metrics found for {algo_prefix}. Returning 0.0.")
        return 0.0

    overall_score_value: float = sum(normalized_scores) / len(normalized_scores)

    # log_msg_details = f"OverallScore for {algo_prefix}: {overall_score_value:.4f} (avg of {len(normalized_scores)} metrics). Metrics considered: {metrics_used_for_score}" # Debugging
    log_msg_details = f"OverallScore for {algo_prefix}: {overall_score_value:.4f} (avg of {len(normalized_scores)} metrics)."
    logger.info(log_msg_details)

    return overall_score_value
