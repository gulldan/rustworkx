# ruff: noqa: E501
import logging
import time
from typing import Any

import networkx as nx
import numpy as np

from benchmark_utils import format_time
from dataset_loaders_shared import SimpleLoaderReturn

logger = logging.getLogger(__name__)


def load_lfr(n: int = 250, mu: float = 0.1, name: str = "LFR Benchmark") -> SimpleLoaderReturn:
    """Generate a synthetic LFR benchmark graph with ground truth communities.

    Uses `nx.LFR_benchmark_graph`. Node attribute "community" (a frozenset)
    is used to derive integer community labels (min element of the set).

    Args:
        n: Number of nodes.
        mu: Mixing parameter (fraction of inter-community edges for a node).
        name: Name for logging.

    Returns:
        A tuple containing:
            - nx.Graph: The LFR benchmark graph.
            - list[int]: True community labels.
            - bool: True (ground truth available).
    """
    tau1: int = 3  # Power law exponent for degree distribution
    tau2: int = 1.5  # Power law exponent for community size distribution
    avg_degree: int = 10
    min_comm: int = 50
    # Ensure max_community is at least min_community and reasonable for n
    max_comm: int = max(min_comm, n // 5 if n // 5 >= min_comm else min_comm + 1)
    if n < min_comm:  # Adjust if n is too small for min_community
        min_comm = max(1, n // 2 if n > 1 else 1)
        max_comm = n
        logger.warning(
            f"LFR: n ({n}) < min_community (50). Adjusted min_community to {min_comm}, max_community to {max_comm}."
        )

    def _generate_lfr_graph(
        n_local: int, mu_local: float, min_c: int, max_c: int, avg_deg_local: int
    ) -> nx.Graph:
        return nx.LFR_benchmark_graph(
            n_local,
            tau1,
            tau2,
            mu_local,
            average_degree=avg_deg_local,
            min_community=min_c,
            max_community=max_c,
            seed=42,
        )

    logger.info(
        f"Generating {name} graph (n={n}, mu={mu}, tau1={tau1}, tau2={tau2}, avg_deg={avg_degree}, min_comm={min_comm}, max_comm={max_comm})..."
    )
    start_time: float = time.time()

    try:
        G: nx.Graph = _generate_lfr_graph(n, mu, min_comm, max_comm, avg_degree)
    except nx.ExceededMaxIterations:
        # Retry with more permissive community sizes/degree for small graphs
        min_comm_retry = max(5, n // 10)
        max_comm_retry = max(min_comm_retry + 1, n // 2)
        avg_degree_retry = max(4, min(8, n - 1))
        logger.warning(
            f"LFR: initial generation failed (ExceededMaxIterations). Retrying with "
            f"min_comm={min_comm_retry}, max_comm={max_comm_retry}, avg_deg={avg_degree_retry}"
        )
        G = _generate_lfr_graph(n, mu, min_comm_retry, max_comm_retry, avg_degree_retry)

    gen_time: float = time.time() - start_time
    logger.info(
        f"Generated {name} ({len(G.nodes())} nodes, {len(G.edges())} edges) in {format_time(gen_time)}"
    )

    # Robustly extract community labels (NetworkX LFR uses list/frozenset; handle both).
    true_labels: list[int] = []
    for node_id in G.nodes():
        community_attr = G.nodes[node_id].get("community")

        def _extract_label(obj: Any) -> int:
            # Handle nested container (e.g., list of frozensets) or flat container of ints.
            if obj is None:
                return -1
            if isinstance(obj, (set, frozenset, list, tuple)):
                if len(obj) == 0:
                    return -1
                first = next(iter(obj))
                if isinstance(first, (set, frozenset, list, tuple)):
                    return min(first) if len(first) > 0 else -1
                try:
                    return int(min(obj))
                except Exception:
                    try:
                        return int(first)
                    except Exception:
                        return -1
            try:
                return int(obj)
            except Exception:
                return -1

        label_val = _extract_label(community_attr)
        if label_val == -1:
            logger.debug(
                f"LFR: node {node_id} missing/invalid community attr {community_attr}; assigned -1."
            )
        true_labels.append(label_val)

    # If community parsing failed (e.g., all -1), fall back to connected components as GT
    if len(set(true_labels)) <= 1:
        logger.warning(
            "LFR: Parsed ground-truth has <=1 cluster. Falling back to connected components as GT."
        )
        comp_labels: dict[Any, int] = {}
        for comp_id, comp_nodes in enumerate(nx.connected_components(G)):
            for n in comp_nodes:
                comp_labels[n] = comp_id
        true_labels = [comp_labels.get(n, 0) for n in G.nodes()]

    for u, v in G.edges():  # Ensure weights for LFR if not added by default
        if "weight" not in G.edges[u, v]:
            G.edges[u, v]["weight"] = 1.0

    return G, true_labels, True


def load_large_synthetic() -> SimpleLoaderReturn:
    """Generate a large synthetic graph with many communities.

    Creates a graph with 15 predefined communities of varying sizes.
    Intra-community connection probability is higher than inter-community.

    Returns:
        A tuple containing:
            - nx.Graph: The SBM graph.
            - list[int]: True community labels.
            - bool: True (ground truth available).
    """
    logger.info("Generating large synthetic SBM graph...")
    num_nodes_total: int = 1500
    sizes: list[int] = [50, 80, 100, 120, 70, 90, 110, 60, 150, 130, 100, 80, 120, 140, 100]

    # Ensure total size matches num_nodes_total if it was specified, otherwise sum sizes.
    if sum(sizes) != num_nodes_total:
        logger.warning(
            f"Sum of SBM community sizes ({sum(sizes)}) does not match target total nodes ({num_nodes_total}). Using sum of sizes."
        )
        # num_nodes_total = sum(sizes) # This line is not needed as sizes define the graph

    p_in: float = 0.1  # Intra-community connection probability
    p_out: float = 0.001  # Inter-community connection probability

    probs: np.ndarray = np.full((len(sizes), len(sizes)), p_out)
    np.fill_diagonal(probs, p_in)

    G: nx.Graph = nx.stochastic_block_model(sizes, probs, seed=42)

    true_labels: list[int] = []
    for i, size_val in enumerate(sizes):
        true_labels.extend([i] * size_val)

    logger.info(
        f"Generated SBM graph with {len(G.nodes())} nodes, {len(G.edges())} edges, and {len(sizes)} communities."
    )
    for u, v in G.edges():  # Ensure weights
        if "weight" not in G.edges[u, v]:
            G.edges[u, v]["weight"] = 1.0
    return G, true_labels, True
