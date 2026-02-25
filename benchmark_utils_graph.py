# ruff: noqa: E501
import logging
from typing import Any

import networkx as nx
import rustworkx as rx

logger = logging.getLogger(__name__)


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
        # Mirror NetworkX semantics: use the 'weight' attribute if present,
        # otherwise default to 1.0. Do not transform or clamp values so that
        # RX and NX see identical weights.
        weight = data.get("weight", 1.0)
        rx_graph.add_edge(node_map[u], node_map[v], float(weight))

    return rx_graph, node_map


def rx_modularity_calculation(
    rx_graph: rx.PyGraph | rx.PyDiGraph,
    communities: list[list[int]],
) -> float:
    """Calculates modularity for a graph and communities using `rustworkx.community.modularity`.

    Args:
        rx_graph: The RustWorkX graph (PyGraph or PyDiGraph).
        communities: A list of communities, where each community is a list of
            node indices (RustWorkX internal IDs).

    Returns:
        The calculated modularity score as a float. Returns `float("nan")`
        if an error occurs during calculation or if the modularity function
        is not found (e.g., due to version issues).
    """
    try:
        return rx.community.modularity(rx_graph, communities)  # type: ignore
    except AttributeError:
        logger.warning(
            "Warning: Function 'modularity' not found in rustworkx.community or type error with arguments."
        )
        return float("nan")
    except Exception as e:
        logger.warning(f"Warning: Error during rx modularity calculation: {e}")
        return float("nan")
