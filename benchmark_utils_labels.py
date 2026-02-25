# ruff: noqa: E501
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix

logger = logging.getLogger(__name__)


def _extract_label_mapping(
    communities: list[set[Any] | list[Any]],
    true_labels: dict[Any, Any] | list[Any] | np.ndarray,
    node_map: dict[Any, int] | None = None,
    algorithm_marker: str | None = None,
) -> tuple[list[Any], list[int], int]:
    """Builds aligned true/prediction label arrays for common nodes."""
    reverse_node_map: dict[int, Any] | None = None
    if node_map is not None:
        reverse_node_map = {v: k for k, v in node_map.items()}

    node_to_pred: dict[Any, int] = {}
    for i, comm in enumerate(communities):
        if not comm:
            continue
        for node in comm:
            mapped_node: Any = reverse_node_map.get(node, node) if reverse_node_map else node
            node_to_pred[mapped_node] = i

    node_to_true: dict[Any, Any]
    if isinstance(true_labels, dict):
        node_to_true = true_labels
    elif isinstance(true_labels, list | np.ndarray):
        node_to_true = {i: v for i, v in enumerate(true_labels)}
    else:
        logger.warning(
            f"({algorithm_marker or 'label_extraction'}) Invalid true_labels type: {type(true_labels)}. Returning no overlap."
        )
        return [], [], 0

    common_nodes: list[Any] = sorted(list(set(node_to_pred) & set(node_to_true)))
    if not common_nodes:
        logger.warning(
            f"({algorithm_marker or 'label_extraction'}) No common nodes between predicted and true labels."
        )
        return [], [], 0

    y_true: list[Any] = [node_to_true[n] for n in common_nodes]
    y_pred: list[int] = [node_to_pred[n] for n in common_nodes]
    return y_true, y_pred, len(common_nodes)


def compare_with_true_labels(
    communities: list[set[Any] | list[Any]],
    true_labels: dict[Any, Any] | list[Any] | np.ndarray,
    node_map: dict[Any, int] | None = None,
    algorithm_marker: str | None = None,
) -> tuple[float, float, float, float, float, float, float, float, float, float, int]:
    """Compares predicted communities with true labels using various metrics.

    Calculates Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
    Homogeneity, Completeness, V-Measure, Fowlkes-Mallows Index (FMI),
    Variation of Information (VI), pairwise precision, recall, and F1-score.

    Args:
        communities: A list of predicted communities. Each community is a
            list or set of node IDs (can be int, str, etc.).
        true_labels: Ground truth labels. Can be a dictionary mapping node ID
            to cluster ID, or a list/numpy array where the index is the
            node ID and the value is the cluster ID.
        node_map: This argument is not currently used. Communities should
            already contain original node IDs.
        algorithm_marker: An optional string marker for logging purposes,
            identifying the algorithm that produced the communities.

    Returns:
        A tuple containing the following metrics in order:
            - ARI (float)
            - NMI (float)
            - Homogeneity (float, currently returns 0 as placeholder)
            - Completeness (float, currently returns 0 as placeholder)
            - V-Measure (float)
            - FMI (float)
            - VI (float)
            - Pairwise Precision (float)
            - Pairwise Recall (float)
            - Pairwise F1-score (float)
            - Number of common nodes used for calculation (int)
        Returns a tuple of zeros if no common nodes are found or if an
        error occurs during processing.
    """
    y_true, y_pred, common_nodes_count = _extract_label_mapping(
        communities, true_labels, node_map=node_map, algorithm_marker=algorithm_marker
    )
    if common_nodes_count == 0:
        nan_val = float("nan")
        return (nan_val,) * 10 + (0,)

    nan_val = float("nan")

    def safe_metric(func: Callable[[list[Any], list[int]], Any], default: float = nan_val) -> float:
        try:
            return float(func(y_true, y_pred))
        except Exception as e:
            logger.debug(
                f"({algorithm_marker or 'compare_with_true_labels'}) Error in safe_metric for {func.__name__}: {e}. Returning default {default}."
            )
            return default

    ari: float = safe_metric(adjusted_rand_score)
    nmi: float = safe_metric(normalized_mutual_info_score)
    v_measure: float = safe_metric(v_measure_score)
    fmi: float = safe_metric(fowlkes_mallows_score)

    homogeneity: float = safe_metric(homogeneity_score)
    completeness: float = safe_metric(completeness_score)

    pw_precision: float = nan_val
    pw_recall: float = nan_val
    pw_f1: float = nan_val
    try:
        if len(set(y_true)) > 1 or len(set(y_pred)) > 1:  # Avoid division by zero if only one cluster
            tn, fp, fn, tp = pair_confusion_matrix(y_true, y_pred).ravel()
            pw_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            pw_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            pw_f1 = (
                2 * pw_precision * pw_recall / (pw_precision + pw_recall)
                if (pw_precision + pw_recall) > 0
                else 0.0
            )
        elif len(y_true) > 0:  # All in one cluster, perfect match if y_pred also one cluster
            pw_precision = 1.0
            pw_recall = 1.0
            pw_f1 = 1.0
    except Exception as e_pair:
        logger.debug(
            f"({algorithm_marker or 'compare_with_true_labels'}) Error in pairwise metrics: {e_pair}. Returning 0s."
        )
        pw_precision = pw_recall = pw_f1 = nan_val

    vi: float = nan_val
    try:
        from sklearn.metrics import mutual_info_score

        # Check if y_true and y_pred can be processed by entropy function
        if not y_true or not y_pred:  # Should be caught by common_nodes check, but defensive
            raise ValueError("y_true or y_pred is empty for VI calculation")

        labels_for_entropy_true = np.array(y_true)
        labels_for_entropy_pred = np.array(y_pred)

        # Ensure labels are non-negative integers for np.bincount
        if not (
            np.issubdtype(labels_for_entropy_true.dtype, np.integer)
            and np.all(labels_for_entropy_true >= 0)
            and np.issubdtype(labels_for_entropy_pred.dtype, np.integer)
            and np.all(labels_for_entropy_pred >= 0)
        ):
            # If not, attempt to factorize them if they are not already suitable
            # This might happen if cluster IDs are strings or negative numbers
            _, labels_for_entropy_true = np.unique(labels_for_entropy_true, return_inverse=True)
            _, labels_for_entropy_pred = np.unique(labels_for_entropy_pred, return_inverse=True)

        def entropy(labels_array: np.ndarray) -> float:
            if labels_array.size == 0:
                return 0.0
            probs: np.ndarray = np.bincount(labels_array) / len(labels_array)
            return -np.sum([p * np.log(p) for p in probs if p > 0])

        h_true: float = entropy(labels_for_entropy_true)
        h_pred: float = entropy(labels_for_entropy_pred)
        mi: float = mutual_info_score(y_true, y_pred)  # Original y_true, y_pred are fine for MI
        vi = h_true + h_pred - 2 * mi
        if vi < 0:
            vi = 0.0  # VI should be non-negative
    except Exception as e_vi:
        logger.debug(
            f"({algorithm_marker or 'compare_with_true_labels'}) Error in VI calculation: {e_vi}. Returning NaN."
        )
        vi = nan_val

    return (
        ari,
        nmi,
        homogeneity,
        completeness,
        v_measure,
        fmi,
        vi,
        pw_precision,
        pw_recall,
        pw_f1,
        common_nodes_count,
    )


def calculate_purity(
    communities: list[list[Any] | set[Any]],
    true_labels: dict[Any, Any] | list[Any] | np.ndarray,
    node_map: dict[Any, int] | None = None,
) -> tuple[float, int]:
    """Calculates Purity score and the number of nodes used.

    Purity = (1/N) * sum_k max_j |c_k intersect t_j|, where N is the
    total number of data points (common nodes), c_k is a predicted cluster,
    and t_j is a true cluster.

    Args:
        communities: List of predicted communities (each a list/set of node IDs).
            Node IDs can be original or RustWorkX integer indices if `node_map` is provided.
        true_labels: Ground truth labels. Dict mapping node ID to true cluster label,
            or list/numpy array of true labels (for 0-N indexed integer nodes).
        node_map: Optional dict mapping original node ID to RustWorkX int index.
            Used if `communities` contain RustWorkX indices.

    Returns:
        A tuple containing:
            - float: Purity score (0 to 1), or `np.nan` if calculation fails.
            - int: Number of nodes common to predicted communities and true labels
              that were used in the calculation.
    """
    y_true, y_pred, common_nodes_count = _extract_label_mapping(
        communities, true_labels, node_map=node_map, algorithm_marker="calculate_purity"
    )
    if not y_true:
        return np.nan, common_nodes_count

    cm = contingency_matrix(y_true, y_pred, sparse=False)
    if cm.size == 0:
        return np.nan, 0

    cm_arr = np.asarray(cm, dtype=np.float64)
    total_points = float(cm_arr.sum())
    if total_points == 0:
        return np.nan, 0

    # Purity: (1/N) * sum over predicted clusters of the max overlap with any true cluster
    max_over_true_per_pred = cm_arr.max(axis=0)  # shape: (num_pred,)
    purity_score = float(max_over_true_per_pred.sum() / total_points)

    return purity_score, common_nodes_count


def calculate_cluster_matching_metrics(
    predicted_communities_input: list[list[Any] | set[Any]],
    true_labels: dict[Any, int] | list[int],
    node_map: dict[Any, int] | None = None,
    jaccard_threshold: float = 0.5,
) -> tuple[float, float]:
    """Calculates Ground Truth Cluster Recall (GCR) and Predicted Cluster Precision (PCP).

    Aims to always return numeric values (no NaN) so downstream tables stay populated,
    even for degenerate cases with single clusters.
    """
    y_true, y_pred, common_nodes_count = _extract_label_mapping(
        predicted_communities_input,
        true_labels,
        node_map=node_map,
        algorithm_marker="calculate_cluster_matching_metrics",
    )
    if common_nodes_count == 0:
        return 0.0, 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = contingency_matrix(y_true, y_pred, sparse=False).astype(float)
    if cm.size == 0:
        return 0.0, 0.0

    cm_arr = np.asarray(cm, dtype=np.float64)
    true_sizes = cm_arr.sum(axis=1).reshape(-1, 1)
    pred_sizes = cm_arr.sum(axis=0).reshape(1, -1)

    # Handle single-cluster edge cases explicitly to avoid NaNs.
    if cm_arr.shape[0] == 1 and cm_arr.shape[1] == 1:
        return 1.0, 1.0
    if cm_arr.shape[0] == 1:
        # One ground-truth cluster vs multiple predicted clusters.
        inter = cm_arr[0, :]
        union = true_sizes[0, 0] + pred_sizes[0, :] - inter
        jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
        gcr_val = 1.0 if np.any(jacc > jaccard_threshold) else 0.0
        pcp_val = float((jacc > jaccard_threshold).mean()) if jacc.size > 0 else 0.0
        return gcr_val, pcp_val
    if cm_arr.shape[1] == 1:
        # Multiple ground-truth clusters vs one predicted cluster.
        inter = cm_arr[:, 0]
        union = true_sizes[:, 0] + pred_sizes[0, 0] - inter
        jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
        gcr_val = float((jacc > jaccard_threshold).mean()) if jacc.size > 0 else 0.0
        pcp_val = 1.0 if np.any(jacc > jaccard_threshold) else 0.0
        return gcr_val, pcp_val

    union = true_sizes + pred_sizes - cm_arr
    jaccard = np.divide(cm_arr, union, out=np.zeros_like(cm_arr), where=union != 0)

    max_j_true = jaccard.max(axis=1)
    max_j_pred = jaccard.max(axis=0)

    gcr = float((max_j_true > jaccard_threshold).mean()) if max_j_true.size > 0 else 0.0
    pcp = float((max_j_pred > jaccard_threshold).mean()) if max_j_pred.size > 0 else 0.0

    logger.info(
        f"PCP/GCR DEBUG: pred_comms={cm.shape[1]}, true_comms={cm.shape[0]}, pcp={pcp:.3f}, gcr={gcr:.3f}, j_thresh={jaccard_threshold}"
    )

    return gcr, pcp
