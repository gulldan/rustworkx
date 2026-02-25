# ruff: noqa: E501
import logging
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import hypergeom

logger = logging.getLogger(__name__)


def calculate_custom_significance(nx_graph: nx.Graph, communities_list_of_sets: list[set[Any]]) -> float:
    """Calculates canonical global Significance for a partition (Aldecoa–Marín Surprise base e).

    This computes the tail probability that at least the observed number of
    intra-community edges would appear under a random edge placement model,
    using the hypergeometric distribution, and returns -ln(p_value).

    Specifically, let:
      - N_nodes = number of nodes in the (undirected, simple) graph
      - M_total = C(N_nodes, 2) = total possible edges
      - M_edges = actual number of edges in the graph
      - N_within = sum over communities of C(|C_i|, 2) = possible intra-community pairs
      - m_within = sum over communities of observed intra-community edges

    Then p_value = sf(m_within - 1; M_total, N_within, M_edges) and
    Significance = -ln(p_value). Surprise (base-10) = -log10(p_value).

    Args:
        nx_graph: The NetworkX graph object (should be simple undirected for counting).
        communities_list_of_sets: A list of communities (sets of node IDs).

    Returns:
        Significance value (natural log). Returns np.nan on invalid inputs, or 0.0
        when no intra-community structure is possible (e.g., no edges).
    """
    N_nodes: int = nx_graph.number_of_nodes()
    M_edges: int = nx_graph.number_of_edges()

    if N_nodes < 2:
        return np.nan

    M_total: int = N_nodes * (N_nodes - 1) // 2
    if M_total == 0:
        return np.nan

    if M_edges == 0:
        # No edges: p-value = 1 for m_within = 0, significance 0
        return 0.0

    if M_edges > M_total:
        # Invalid simple-graph case
        return np.nan

    # Compute global possible-within pairs and observed-within edges
    N_within: int = 0
    m_within: int = 0

    for comm_nodes in communities_list_of_sets:
        n_c = len(comm_nodes)
        if n_c < 2:
            continue
        N_within += n_c * (n_c - 1) // 2
        # Count actual internal edges for this community
        sub = nx_graph.subgraph(comm_nodes)
        m_within += sub.number_of_edges()

    # If no within-pair is possible, significance is 0
    if N_within == 0:
        return 0.0

    # Hypergeometric tail probability: P[X >= m_within]
    # scipy.stats.hypergeom.sf(k, M, n, N): population M, successes n, draws N, tail >= k+1
    k_param = max(0, m_within - 1)
    p_val = float(hypergeom.sf(k_param, M_total, N_within, M_edges))
    # Numerical guards
    if not np.isfinite(p_val) or p_val <= 0.0:
        p_val = 1e-300
    elif p_val > 1.0:
        p_val = 1.0

    significance_ln = -float(np.log(p_val))
    # Cap extremely large values to avoid inf downstream (approx -ln(min double))
    if not np.isfinite(significance_ln):
        significance_ln = 708.0
    else:
        significance_ln = min(significance_ln, 708.0)

    return significance_ln


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

    # Use a simple undirected graph for internal metrics to avoid unsupported ops on directed/multi graphs
    if nx_graph.is_directed() or isinstance(nx_graph, (nx.MultiGraph, nx.MultiDiGraph)):
        G_u = nx.Graph()
        G_u.add_nodes_from(nx_graph.nodes())
        try:
            # Collapses multiedges automatically when adding to simple Graph
            G_u.add_edges_from(nx_graph.to_undirected().edges())
        except Exception:
            # Fallback: iterate edges without data
            G_u.add_edges_from([(u, v) for u, v in nx_graph.edges()])
    else:
        G_u = nx_graph if isinstance(nx_graph, nx.Graph) else nx.Graph(nx_graph)

    total_edges_graph = G_u.number_of_edges()
    try:
        triangle_cache = _global_triangles_cache(G_u)
    except Exception:
        triangle_cache = {}

    conductance_vals: list[float] = []
    int_density_vals: list[float] = []
    avg_int_deg_vals: list[float] = []
    tpr_vals: list[float] = []
    cut_ratio_vals: list[float] = []

    for comm in communities_list_of_sets:
        n_c = len(comm)
        if n_c < 1:
            continue

        subgraph = G_u.subgraph(comm)
        m_c = subgraph.number_of_edges()

        # Internal density
        possible_edges = n_c * (n_c - 1) / 2
        int_density_vals.append(float(m_c / possible_edges) if possible_edges > 0 else 0.0)

        # Average internal degree
        avg_int_deg_vals.append(float(2 * m_c / n_c))

        # Cut edges
        cut_edges = 0
        for u in comm:
            for v in G_u.neighbors(u):
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

    # Canonical global Significance/Surprise (Aldecoa–Marín): -ln p and -log10 p
    significance_raw = calculate_custom_significance(G_u, communities_list_of_sets)
    significance = significance_raw if np.isfinite(significance_raw) else np.nan
    surprise = (significance_raw / np.log(10.0)) if np.isfinite(significance_raw) else np.nan

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
