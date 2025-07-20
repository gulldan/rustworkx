import logging
import math
import tracemalloc
from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
import rustworkx as rx

# Import from metrics_config.py
from metrics_config import ORDERED_METRIC_PROPERTIES
from scipy.stats import hypergeom  # Added for custom significance

# Vectorized metrics from scikit-learn
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    v_measure_score,
)

# Contingency matrix for fast purity & Jaccard computations
from sklearn.metrics.cluster import contingency_matrix

# Setup logger for this module
logger = logging.getLogger(__name__)
# Basic configuration (optional here, could be in main script)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Helper Function for Parallel ASPL Calculation ---
# Must be defined at top level for multiprocessing pickling

# --- End Helper Function ---

def convert_nx_to_rx(nx_graph: nx.Graph) -> tuple[rx.PyGraph, dict[Any, int]]:
    """Converts a NetworkX graph to a RustWorkX graph.

    Ensures that edge weights are positive, defaulting to 1.0. If 'distance'
    is present in edge attributes, it's converted to similarity.

    Args:
        nx_graph: The NetworkX graph to convert.

    Returns:
        A tuple containing:
            - rx.PyGraph: The converted RustWorkX graph.
            - dict[Any, int]: A mapping from original node IDs to RustWorkX
              integer node indices.
    """
    node_map: dict[Any, int] = {}
    rx_graph: rx.PyGraph = rx.PyGraph()

    for node in nx_graph.nodes():
        rx_idx: int = rx_graph.add_node(node)
        node_map[node] = rx_idx

    for u, v, data in nx_graph.edges(data=True):
        weight: float = 1.0
        if "weight" in data:
            weight = float(data["weight"])
        elif "distance" in data:
            distance: float = float(data.get("distance", 2.0))
            similarity: float = max(1e-9, 1.0 - distance / 2.0)
            weight = similarity

        final_weight: float = float(max(weight, 1e-9))
        rx_graph.add_edge(node_map[u], node_map[v], final_weight)

    return rx_graph, node_map


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
    node_to_pred: dict[Any, int] = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_pred[node] = i

    node_to_true: dict[Any, Any]
    if isinstance(true_labels, dict):
        node_to_true = true_labels
    elif isinstance(true_labels, list | np.ndarray):
        node_to_true = {i: v for i, v in enumerate(true_labels)}
    else:
        logger.warning(
            f"({algorithm_marker or 'compare_with_true_labels'}) Invalid true_labels type: {type(true_labels)}. Returning zero metrics."
        )
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    common_nodes: list[Any] = sorted(list(set(node_to_pred) & set(node_to_true)))
    if not common_nodes:
        logger.warning(
            f"({algorithm_marker or 'compare_with_true_labels'}) No common nodes between predicted and true labels. Returning zero metrics."
        )
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    y_true: list[Any] = [node_to_true[n] for n in common_nodes]
    y_pred: list[int] = [node_to_pred[n] for n in common_nodes]

    def safe_metric(func: Callable[[list[Any], list[int]], Any], default: float = 0.0) -> float:
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

    homogeneity: float = (
        0.0  # Placeholder, sklearn.metrics.homogeneity_score requires labels_true, labels_pred
    )
    completeness: float = 0.0  # Placeholder, sklearn.metrics.completeness_score

    pw_precision: float = 0.0
    pw_recall: float = 0.0
    pw_f1: float = 0.0
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
        pw_precision = pw_recall = pw_f1 = 0.0

    # Purity is calculated by calculate_purity function now.
    # This function's purity calculation was removed earlier based on context.

    vi: float = 0.0
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
            f"({algorithm_marker or 'compare_with_true_labels'}) Error in VI calculation: {e_vi}. Returning 0."
        )
        vi = 0.0

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
        len(common_nodes),
    )


def measure_memory(func: Callable[..., Any]) -> Callable[..., tuple[Any, float]]:
    """Decorator to measure peak memory usage of a function.

    Uses `tracemalloc` to track memory allocation.

    Args:
        func: The function to measure.

    Returns:
        A wrapper function that, when called, executes the original function
        and returns a tuple containing the original function's result and
        the peak memory usage in megabytes (MB).
    """

    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        tracemalloc.start()
        result: Any = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()  # current, peak
        tracemalloc.stop()
        return result, peak / (1024 * 1024)  # Convert to MB

    return wrapper


def format_memory(memory_mb: float) -> str:
    """Formats memory usage with adaptive units (KB, MB, GB).

    Displays the memory value with a maximum of 3 significant digits.

    Args:
        memory_mb: Memory usage in megabytes (MB).

    Returns:
        A string representing the formatted memory usage with an
        appropriate unit (KB, MB, or GB).
    """
    if memory_mb < 0.1:
        memory_kb: float = memory_mb * 1024
        value: float = (
            round(memory_kb, 3 - int(math.floor(math.log10(abs(memory_kb)))) - 1)
            if memory_kb >= 1 and memory_kb != 0
            else memory_kb
        )
        return f"{value:.3g} KB"
    elif memory_mb < 1000:
        value = (
            round(memory_mb, 3 - int(math.floor(math.log10(abs(memory_mb)))) - 1)
            if memory_mb >= 1 and memory_mb != 0
            else memory_mb
        )
        return f"{value:.3g} MB"
    else:
        memory_gb: float = memory_mb / 1024
        value = (
            round(memory_gb, 3 - int(math.floor(math.log10(abs(memory_gb)))) - 1)
            if memory_gb >= 1 and memory_gb != 0
            else memory_gb
        )
        return f"{value:.3g} GB"


def format_time(seconds: float) -> str:
    """Formats time duration with adaptive units (µs, ms, s).

    Displays the time value with a maximum of 3 significant digits.

    Args:
        seconds: Time duration in seconds.

    Returns:
        A string representing the formatted time duration with an
        appropriate unit (µs, ms, or s).
    """
    if seconds == 0:
        return "0 s"

    microseconds: float = seconds * 1000000
    value: float

    if microseconds < 1000:  # μs range
        value = (
            round(microseconds, 3 - int(math.floor(math.log10(abs(microseconds)))) - 1)
            if microseconds >= 1 and microseconds != 0
            else microseconds
        )
        return f"{value:.3g} μs"
    elif microseconds < 1000000:  # ms range
        milliseconds: float = microseconds / 1000
        value = (
            round(milliseconds, 3 - int(math.floor(math.log10(abs(milliseconds)))) - 1)
            if milliseconds >= 1 and milliseconds != 0
            else milliseconds
        )
        return f"{value:.3g} ms"
    else:  # s range
        value = (
            round(seconds, 3 - int(math.floor(math.log10(abs(seconds)))) - 1)
            if seconds >= 1 and seconds != 0
            else seconds
        )
        return f"{value:.3g} s"


def rx_modularity_calculation(
    rx_graph: rx.PyGraph | rx.PyDiGraph,
    communities: list[list[int]],
    weight_fn: Callable[[Any], float] | None,
) -> float:
    """Calculates modularity for a graph and communities using `rustworkx.community.modularity`.

    Args:
        rx_graph: The RustWorkX graph (PyGraph or PyDiGraph).
        communities: A list of communities, where each community is a list of
            node indices (RustWorkX internal IDs).
        weight_fn: An optional callable that takes an edge object and returns
            its weight as a float. If None, the graph is treated as unweighted
            by `rustworkx.community.modularity` or it uses a default.

    Returns:
        The calculated modularity score as a float. Returns `float("nan")`
        if an error occurs during calculation or if the modularity function
        is not found (e.g., due to version issues).
    """
    try:
        # rustworkx.community.modularity should handle weight_fn=None correctly
        # for unweighted graphs or graphs where edge data are directly weights.
        return rx.community.modularity(rx_graph, communities, weight_fn=weight_fn) # type: ignore
    except AttributeError:
        logger.warning(
            "Warning: Function 'modularity' not found in rustworkx.community or type error with arguments."
        )
        return float("nan")
    except Exception as e:
        logger.warning(f"Warning: Error during rx modularity calculation: {e}")
        return float("nan")


def calculate_custom_significance(nx_graph: nx.Graph, communities_list_of_sets: list[set[Any]]) -> float:
    """Calculates community significance based on Traag et al. (2015).

    Significance is defined as sum_c (-log p_c), where p_c is the p-value
    of observing at least m_c (internal) edges in community c, given its
    size n_c. The p-value is calculated using the hypergeometric distribution.

    Args:
        nx_graph: The NetworkX graph object.
        communities_list_of_sets: A list of communities, where each community
            is a set of node IDs.

    Returns:
        The calculated significance score as a float. Returns `np.nan` or `0.0`
        under certain conditions (e.g., graph too small, no edges, calculation errors).
    """
    N: int = nx_graph.number_of_nodes()
    M_graph_edges: int = nx_graph.number_of_edges()
    
    # Removed early skip for large graphs. Instead use vectorized evaluation for performance.

    if N < 2:
        logger.info(f"Custom significance: Graph has < 2 nodes (N={N}). Returning NaN.")
        return np.nan

    total_possible_edges_in_graph: int = N * (N - 1) // 2

    if M_graph_edges == 0:
        logger.info("Custom significance: Graph has 0 edges. Returning 0.0.")
        return 0.0

    if total_possible_edges_in_graph == 0:  # Should be caught by N < 2, but defensive
        logger.warning(
            f"Custom significance: Total possible edges in graph is 0 despite N={N}>=2. Returning NaN."
        )
        return np.nan

    if M_graph_edges > total_possible_edges_in_graph:
        logger.warning(
            f"Custom significance: Graph edges {M_graph_edges} > total possible edges {total_possible_edges_in_graph}. Invalid. Returning NaN."
        )
        return np.nan

    # Vectorized computation across communities
    k_arr = []  # m_c - 1
    n_arr = []  # possible_edges_in_c

    for community_nodes_set in communities_list_of_sets:
        n_c = len(community_nodes_set)
        if n_c < 2:
            continue
        valid_nodes = [node for node in community_nodes_set if node in nx_graph]
        if len(valid_nodes) < 2:
            continue

        subgraph_c = nx_graph.subgraph(valid_nodes)
        m_c = subgraph_c.number_of_edges()
        possible_edges_in_c = n_c * (n_c - 1) // 2
        if possible_edges_in_c == 0:
            continue

        k_arr.append(m_c - 1)
        n_arr.append(possible_edges_in_c)

    if not k_arr:
        return 0.0

    k_vec = np.array(k_arr)
    n_vec = np.array(n_arr)
    M_sf = total_possible_edges_in_graph
    K_sf = M_graph_edges

    p_vals = hypergeom.sf(k_vec, M_sf, n_vec, K_sf)

    # Handle 0, NaN, etc.
    p_vals = np.clip(p_vals, 1e-300, 1.0)
    neg_log_p = -np.log(p_vals)

    # Cap very high values to 708 (approx -log(min double))
    neg_log_p = np.minimum(neg_log_p, 708.0)

    return float(np.sum(neg_log_p))


# Precompute global triangles function for efficiency
def _global_triangles_cache(nx_graph: nx.Graph) -> dict[Any, int]:
    """Returns a cache of triangle counts per node for the given graph.

    Uses LRU cache implicitly by attaching attribute to graph object."""
    cache_key = "_triangles_cache"
    if hasattr(nx_graph, cache_key):
        return getattr(nx_graph, cache_key)  # type: ignore[attr-defined]
    tri_dict = nx.triangles(nx_graph)
    setattr(nx_graph, cache_key, tri_dict)
    return tri_dict


def calculate_internal_metrics(
    nx_graph: nx.Graph,
    communities_list_of_sets: list[set[Any]],
    node_map: dict[Any, int] | None = None,
    algorithm_marker: str = "algo",
) -> tuple[float, float, float, float, float, float, float]:
    """Compute internal metrics without cdlib (faster, no external deps).

    Metrics: conductance, internal_density, avg_internal_degree,
    triangle participation ratio (TPR), cut_ratio, surprise (NaN placeholder), significance.
    """

    if not communities_list_of_sets or nx_graph.number_of_nodes() == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    total_edges_graph = nx_graph.number_of_edges()
    triangle_cache = _global_triangles_cache(nx_graph)

    conductance_vals: list[float] = []
    int_density_vals: list[float] = []
    avg_int_deg_vals: list[float] = []
    tpr_vals: list[float] = []
    cut_ratio_vals: list[float] = []

    for comm in communities_list_of_sets:
        n_c = len(comm)
        if n_c < 1:
            continue

        subgraph = nx_graph.subgraph(comm)
        m_c = subgraph.number_of_edges()

        # Internal density
        possible_edges = n_c * (n_c - 1) / 2
        int_density_vals.append(float(m_c / possible_edges) if possible_edges > 0 else 0.0)

        # Average internal degree
        avg_int_deg_vals.append(float(2 * m_c / n_c))

        # Cut edges
        cut_edges = 0
        for u in comm:
            for v in nx_graph.neighbors(u):
                if v not in comm:
                    cut_edges += 1
        cut_edges = cut_edges / 2  # undirected counted twice

        # Conductance
        denom = 2 * m_c + cut_edges
        conductance_vals.append(float(cut_edges / denom) if denom > 0 else 0.0)

        # Cut ratio
        cut_ratio_vals.append(float(cut_edges / total_edges_graph) if total_edges_graph > 0 else 0.0)

        # TPR (using global triangles as approximation)
        nodes_with_triangles = sum(1 for n in comm if triangle_cache.get(n, 0) > 0)
        tpr_vals.append(float(nodes_with_triangles / n_c))

    # Aggregate (mean). If list empty -> NaN
    def _mean_safe(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else np.nan

    conductance = _mean_safe(conductance_vals)
    internal_density = _mean_safe(int_density_vals)
    avg_internal_degree = _mean_safe(avg_int_deg_vals)
    tpr = _mean_safe(tpr_vals)
    cut_ratio = _mean_safe(cut_ratio_vals)

    surprise = np.nan  # Placeholder (complex to compute efficiently)

    significance_raw = calculate_custom_significance(nx_graph, communities_list_of_sets)
    significance = significance_raw if not (np.isnan(significance_raw) or np.isinf(significance_raw)) else np.nan

    logger.info(
        f"({algorithm_marker}) internal metrics computed: cond={conductance:.3f}, dens={internal_density:.3f}, avg_int_deg={avg_internal_degree:.3f}, tpr={tpr:.3f}, cut_ratio={cut_ratio:.3f}, significance={significance:.3f}"
    )

    return (
        conductance,
        internal_density,
        avg_internal_degree,
        tpr,
        cut_ratio,
        surprise,
        significance,
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
    # --- Vectorized implementation ---
    # Map predicted communities to node -> cluster_id
    node_to_pred: dict[Any, int] = {}
    for idx, comm in enumerate(communities):
        if not comm:
            continue
        for n in comm:
            if node_map is not None:
                n = {v: k for k, v in node_map.items()}.get(n, n)  # map back to original if needed
            node_to_pred[n] = idx

    if not node_to_pred:
        logger.warning("Warning (calculate_purity): Empty predicted communities after processing.")
        return np.nan, 0

    # Ground-truth mapping
    if isinstance(true_labels, dict):
        node_to_true = true_labels
    else:  # list[int]
        node_to_true = {i: v for i, v in enumerate(true_labels)}  # type: ignore[arg-type]

    common_nodes = [n for n in node_to_pred if n in node_to_true]
    if not common_nodes:
        logger.warning("Warning (calculate_purity): No common nodes with ground truth for purity.")
        return np.nan, 0

    y_true = np.array([node_to_true[n] for n in common_nodes])
    y_pred = np.array([node_to_pred[n] for n in common_nodes])

    cm = contingency_matrix(y_true, y_pred, sparse=False)
    if cm.size == 0:
        return np.nan, 0

    cm_arr = np.asarray(cm, dtype=np.float64)
    true_sizes = cm_arr.sum(axis=1).reshape(-1, 1)
    pred_sizes = cm_arr.sum(axis=0).reshape(1, -1)
    union = true_sizes + pred_sizes - cm_arr
    jaccard = np.divide(cm_arr, union, out=np.zeros_like(cm_arr), where=union != 0)

    max_j_true = jaccard.max(axis=1)
    max_j_pred = jaccard.max(axis=0)

    gcr = float((max_j_true > 0.5).mean())
    pcp = float((max_j_pred > 0.5).mean())

    logger.info(
        f"PCP/GCR DEBUG: pred_comms={cm.shape[1]}, true_comms={cm.shape[0]}, pcp={pcp:.3f}, gcr={gcr:.3f}, j_thresh=0.5"
    )

    return gcr, len(common_nodes)


def calculate_cluster_matching_metrics(
    predicted_communities_input: list[list[Any] | set[Any]],
    true_labels: dict[Any, int] | list[int],
    node_map: dict[Any, int] | None = None,
    jaccard_threshold: float = 0.5,
) -> tuple[float, float]:
    """Calculates Ground Truth Cluster Recall (GCR) and Predicted Cluster Precision (PCP).

    GCR: Percentage of ground truth clusters well-matched by predicted clusters.
    PCP: Percentage of predicted clusters well-matched by ground truth clusters.
    A 'good match' is defined by Jaccard Index > `jaccard_threshold`.

    Args:
        predicted_communities_input: List of predicted communities. Each community
            is a list or set of node IDs (original, or RX int indices if `node_map`
            is given).
        true_labels: Dict mapping node ID to true cluster ID, or a list of
            true cluster IDs for 0-N indexed integer nodes.
        node_map: Optional dict mapping original node ID to RX int index.
        jaccard_threshold: Minimum Jaccard Index for a match (default 0.5).

    Returns:
        A tuple (gcr, pcp), where gcr is Ground Truth Recall and pcp is
        Predicted Cluster Precision. Both are floats between 0.0 and 1.0.
        Returns (0.0, 0.0) if inputs are empty or issues occur.
    """
    # --- Vectorized implementation ---
    # Map predicted communities to node -> cluster_id
    node_to_pred: dict[Any, int] = {}
    for idx, comm in enumerate(predicted_communities_input):
        if not comm:
            continue
        for n in comm:
            if node_map is not None:
                n = {v: k for k, v in node_map.items()}.get(n, n)  # map back to original if needed
            node_to_pred[n] = idx

    if not node_to_pred:
        logger.info("PCP/GCR: Predicted communities empty after processing. Returning 0,0.")
        return 0.0, 0.0

    # Ground-truth mapping
    if isinstance(true_labels, dict):
        node_to_true = true_labels
    else:  # list[int]
        node_to_true = {i: v for i, v in enumerate(true_labels)}  # type: ignore[arg-type]

    common_nodes = [n for n in node_to_pred if n in node_to_true]
    if not common_nodes:
        return 0.0, 0.0

    y_true = np.array([node_to_true[n] for n in common_nodes])
    y_pred = np.array([node_to_pred[n] for n in common_nodes])

    cm = contingency_matrix(y_true, y_pred, sparse=False).astype(float)
    if cm.size == 0:
        return 0.0, 0.0

    cm_arr = np.asarray(cm, dtype=np.float64)
    true_sizes = cm_arr.sum(axis=1).reshape(-1, 1)
    pred_sizes = cm_arr.sum(axis=0).reshape(1, -1)
    union = true_sizes + pred_sizes - cm_arr
    jaccard = np.divide(cm_arr, union, out=np.zeros_like(cm_arr), where=union != 0)

    max_j_true = jaccard.max(axis=1)
    max_j_pred = jaccard.max(axis=0)

    gcr = float((max_j_true > jaccard_threshold).mean())
    pcp = float((max_j_pred > jaccard_threshold).mean())

    logger.info(
        f"PCP/GCR DEBUG: pred_comms={cm.shape[1]}, true_comms={cm.shape[0]}, pcp={pcp:.3f}, gcr={gcr:.3f}, j_thresh={jaccard_threshold}"
    )

    return gcr, pcp


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
