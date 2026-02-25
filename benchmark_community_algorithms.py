# ruff: noqa: E501
import logging
from collections.abc import Callable
from typing import Any

import cdlib
import cdlib.algorithms
import networkx as nx
import rustworkx as rx
from benchmark_utils import measure_memory


logger = logging.getLogger(__name__)

try:
    if not hasattr(rx, "PyGraph"):
        logger.warning("rx.PyGraph not found, checking for alternative imports...")
        rx_attrs = dir(rx)
        logger.info(f"Available rustworkx attributes: {', '.join(rx_attrs)}")
except Exception as e:
    logger.error(f"Error checking rustworkx attributes: {e}", exc_info=True)


@measure_memory
def run_nx_algorithm(graph: nx.Graph, resolution: float = 1.0) -> list[set]:
    """Runs the NetworkX Louvain community detection algorithm.

    Ensures that edges have a 'weight' attribute (defaults to 1.0).
    Returns communities as a list of sets of node IDs.

    Args:
        graph (nx.Graph): The input NetworkX graph.
        resolution (float, optional): Resolution parameter for Louvain. Defaults to 1.0.

    Returns:
        List[set]: A list of sets, where each set contains the node IDs
                   of a detected community. Returns an empty list if an error
                   occurs or no communities are found.
    """
    try:
        if graph.number_of_edges() == 0:
            logger.warning(
                "  Graph has no edges, NetworkX Louvain may fail or return trivial communities."
            )
            return [{node} for node in graph.nodes()]

        for u, v, data in graph.edges(data=True):
            if "weight" not in data:
                graph.edges[u, v]["weight"] = 1.0

        communities_gen: list[set] = nx.community.louvain_communities(
            graph, weight="weight", seed=42, resolution=resolution
        )

        return [set(c) for c in communities_gen if c] if communities_gen else []
    except Exception as e:
        logger.error(f"  Error in NetworkX Louvain (resolution={resolution}): {e}", exc_info=True)
        return []


@measure_memory
def run_rx_algorithm(
    graph: rx.PyGraph, resolution: float = 1.0, adjacency: list[list[int]] | None = None
) -> list[list[int]]:
    """Runs the RustWorkX Louvain community detection algorithm.

    Args:
        graph (rx.PyGraph): The input RustWorkX graph.
        resolution (float, optional): Resolution parameter for Louvain. Defaults to 1.0.
        adjacency (Optional[list[list[int]]]): Pre-built adjacency list for exact NetworkX matching.
            If provided, uses NX's neighbor order for deterministic tie-breaking.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains the
                         node indices (RustWorkX internal IDs) of a detected community.
                         Returns an empty list if an error occurs.
    """

    try:
        communities: list[list[int]] = rx.community.louvain_communities(
            graph, seed=42, resolution=resolution, adjacency=adjacency
        )  # type: ignore
        return communities
    except Exception as e:
        logger.error(f"  Error in RustWorkX Louvain (resolution={resolution}): {e}", exc_info=True)
        return []


@measure_memory
def run_cdlib_algorithm(
    graph: nx.Graph, algorithm_name: str, resolution: float = 1.0
) -> list[list[Any]]:
    """Runs a specified cdlib community detection algorithm.

    Currently supports 'leiden' and 'infomap'.

    Args:
        graph (nx.Graph): The input NetworkX graph.
        algorithm_name (str): The name of the cdlib algorithm to run
                              (e.g., "leiden", "infomap").
        resolution (float, optional): Resolution parameter, primarily for Leiden.
                                      Not used by Infomap in this setup. Defaults to 1.0.

    Returns:
        List[List[Any]]: A list of lists, where each inner list contains the
                         node IDs of a detected community. Returns an empty list
                         if an error occurs or the algorithm is unknown.

    Raises:
        ValueError: If an unsupported `algorithm_name` is provided.
    """
    coms_result: list[list[Any]] = []
    try:
        if algorithm_name == "leiden":
            # cdlib wraps leidenalg/igraph and does not expose a seed parameter.
            # Seed igraph's RNG explicitly for deterministic benchmark runs and
            # inject `seed` into cdlib's internal leidenalg.find_partition call.
            original_find_partition: Callable[..., Any] | None = None
            try:
                import random
                import igraph as ig
                import leidenalg

                ig.set_random_number_generator(random.Random(42))
                if hasattr(leidenalg, "set_rng_seed"):
                    leidenalg.set_rng_seed(42)
                original_find_partition = getattr(leidenalg, "find_partition", None)
            except Exception as seed_exc:
                logger.debug(f"  Unable to set leidenalg RNG seed via cdlib wrapper: {seed_exc}")

            def _seeded_find_partition(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("seed", 42)
                return original_find_partition(*args, **kwargs)  # type: ignore[misc]

            if original_find_partition is not None:
                setattr(leidenalg, "find_partition", _seeded_find_partition)

            try:
                coms_obj: cdlib.classes.node_clustering.NodeClustering = cdlib.algorithms.leiden(
                    graph, weights="weight"
                )
            finally:
                if original_find_partition is not None:
                    setattr(leidenalg, "find_partition", original_find_partition)

            if hasattr(coms_obj, "communities") and isinstance(coms_obj.communities, list):
                coms_result = coms_obj.communities
        elif algorithm_name == "infomap":
            coms_obj: cdlib.classes.node_clustering.NodeClustering = cdlib.algorithms.infomap(
                graph, flags="--seed 42"
            )
            if hasattr(coms_obj, "communities") and isinstance(coms_obj.communities, list):
                coms_result = coms_obj.communities
            elif isinstance(coms_obj, list):  # Infomap might directly return a list
                coms_result = coms_obj
        else:
            raise ValueError(f"Unknown cdlib algorithm: {algorithm_name}")
    except Exception as e:
        resolution_info: str = f"resolution={resolution}" if algorithm_name == "leiden" else "N/A"
        logger.error(f"  Error in cdlib {algorithm_name} ({resolution_info}): {e}", exc_info=True)
        return []

    return [list(c) for c in coms_result if c] if coms_result else []


@measure_memory
def run_rx_leiden_algorithm(
    graph: rx.PyGraph, resolution: float = 1.0, adjacency: list[list[int]] | None = None
) -> list[list[int]]:
    """Runs the RustWorkX Leiden community detection algorithm.

    Args:
        graph (rx.PyGraph): The input RustWorkX graph.
        resolution (float, optional): Resolution parameter for Leiden. Defaults to 1.0.
        adjacency (Optional[list[list[int]]]): Pre-built adjacency list to align
            node-neighbor iteration order with NetworkX benchmark graph.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains the
                         node indices (RustWorkX internal IDs) of a detected community.
                         Returns an empty list if an error occurs.
    """

    try:
        communities: list[list[int]] = rx.community.leiden_communities(  # type: ignore
            graph,
            resolution=resolution,
            seed=42,
            adjacency=adjacency,
        )
        return communities
    except Exception as e:
        logger.error(f"  Error in RustWorkX Leiden (resolution={resolution}): {e}", exc_info=True)
        return []


@measure_memory
def run_rx_lpa_algorithm(
    graph: rx.PyGraph, weight: str | None, seed: int | None, adjacency: list[list[int]] | None = None
) -> list[list[int]]:
    """Runs the RustWorkX Label Propagation Algorithm (LPA).

    Args:
        graph (rx.PyGraph): The input RustWorkX graph.
        weight (str | None): The name of the edge attribute to use as weight.
        seed (Optional[int]): A seed for the random number generator used by LPA.
        adjacency (Optional[list[list[int]]]): Pre-built adjacency list for exact NetworkX matching.
            If provided, must preserve the same neighbor order as NetworkX's G[node].keys().

    Returns:
        List[List[int]]: A list of lists, where each inner list contains the
                         node indices (RustWorkX internal IDs) of a detected community.
                         Returns an empty list if an error occurs.
    """
    try:
        # Match NetworkX behavior: run until convergence (no max_iterations cap).
        communities = rx.community.asyn_lpa_communities(
            graph, weight=weight, seed=seed, adjacency=adjacency
        )  # type: ignore
        return communities
    except Exception as e:
        logger.error(f"  Error in RustWorkX LPA (weight={weight}): {e}", exc_info=True)
        return []


@measure_memory
def run_rx_lpa_strongest_algorithm(
    graph: rx.PyGraph, weight: str | None, seed: int | None
) -> list[list[int]]:
    """Runs the RustWorkX strongest-edge Label Propagation variant."""
    try:
        communities: list[list[int]] = rx.community.asyn_lpa_communities_strongest(
            graph, weight=weight, seed=seed
        )  # type: ignore
        return communities
    except Exception as e:
        logger.error(f"  Error in RustWorkX strongest-edge LPA (weight={weight}): {e}", exc_info=True)
        return []


@measure_memory
def run_nx_lpa_algorithm(graph: nx.Graph, seed: int | None) -> list[set]:
    """Runs the NetworkX Asynchronous Label Propagation Algorithm (LPA).

    Ensures communities are returned as a list of sets of node IDs.

    Args:
        graph (nx.Graph): The input NetworkX graph.
        seed (Optional[int]): A seed for the random number generator.

    Returns:
        List[set]: A list of sets, where each set contains the node IDs
                   of a detected community. Returns an empty list if an error
                   occurs.
    """
    try:
        # NX has no max_iterations; on large graphs it can be slow. For consistency with RX cap,
        # we accept full convergence here to compare quality; skips handled elsewhere for huge graphs.
        communities_generator = nx.community.asyn_lpa_communities(graph, weight="weight", seed=seed)
        return [set(c) for c in communities_generator if c]
    except Exception as e:
        logger.error(f"  Error in NetworkX LPA: {e}", exc_info=True)
        return []


@measure_memory
def run_nx_cliques_algorithm(graph: nx.Graph) -> list[list[Any]]:
    """Runs NetworkX maximal-clique enumeration."""
    try:
        return [list(c) for c in nx.find_cliques(graph) if c]
    except Exception as e:
        logger.error(f"  Error in NetworkX maximal cliques: {e}", exc_info=True)
        return []


@measure_memory
def run_rx_cliques_algorithm(graph: rx.PyGraph) -> list[list[int]]:
    """Runs rustworkx maximal-clique enumeration."""
    try:
        communities: list[list[int]] = rx.community.find_maximal_cliques(graph)  # type: ignore
        return communities
    except Exception as e:
        logger.error(f"  Error in RustWorkX maximal cliques: {e}", exc_info=True)
        return []


@measure_memory
def run_nx_cpm_algorithm(graph: nx.Graph, k: int = 3) -> list[list[Any]]:
    """Runs NetworkX clique-percolation communities (CPM)."""
    try:
        communities = nx.community.k_clique_communities(graph, k)
        return [list(c) for c in communities if c]
    except Exception as e:
        logger.error(f"  Error in NetworkX CPM (k={k}): {e}", exc_info=True)
        return []


@measure_memory
def run_rx_cpm_algorithm(graph: rx.PyGraph, k: int = 3) -> list[list[int]]:
    """Runs rustworkx clique-percolation communities (CPM)."""
    try:
        communities: list[list[int]] = rx.community.cpm_communities(graph, k)  # type: ignore
        return communities
    except Exception as e:
        logger.error(f"  Error in RustWorkX CPM (k={k}): {e}", exc_info=True)
        return []


@measure_memory
def run_leidenalg_algorithm(graph: nx.Graph) -> list[list[Any]]:
    """Runs the original leidenalg Leiden algorithm (by V.A. Traag).

    This is the reference C++ implementation from https://github.com/vtraag/leidenalg

    Args:
        graph (nx.Graph): The input NetworkX graph.

    Returns:
        List[List[Any]]: A list of lists, where each inner list contains the
                         node IDs of a detected community. Returns an empty list
                         if an error occurs.
    """
    try:
        import igraph as ig
        import leidenalg

        # Convert NetworkX to igraph
        nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        idx_to_node = {i: n for n, i in node_to_idx.items()}

        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
        weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]

        g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
        g_ig.es["weight"] = weights

        # Run Leiden with ModularityVertexPartition (same as RX Leiden default)
        partition = leidenalg.find_partition(
            g_ig, leidenalg.ModularityVertexPartition, weights="weight", seed=42
        )

        # Convert back to original node IDs
        communities: list[list[Any]] = []
        for comm_indices in partition:
            comm = [idx_to_node[i] for i in comm_indices]
            communities.append(comm)

        return communities
    except ImportError as ie:
        logger.warning(f"  leidenalg or igraph not installed: {ie}")
        return []
    except Exception as e:
        logger.error(f"  Error in leidenalg: {e}", exc_info=True)
        return []
