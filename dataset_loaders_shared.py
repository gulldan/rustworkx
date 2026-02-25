# ruff: noqa: E501
import logging
import os
from collections.abc import Callable
from typing import Any

import networkx as nx
import polars as pl

logger = logging.getLogger(__name__)

SimpleLoaderReturn = tuple[nx.Graph, list[int] | dict[Any, int], bool]
DictLoaderReturn = dict[str, nx.Graph | dict[Any, int] | dict[str, Any] | str | None]


def _ensure_default_edge_weight(graph: nx.Graph, default_weight: float = 1.0) -> None:
    """Assigns default weights to edges missing a weight attribute."""
    for u, v, data in graph.edges(data=True):
        if "weight" not in data:
            data["weight"] = default_weight


def _load_gml_with_ground_truth(
    file_path: str,
    node_label_fn: Callable[[dict[str, Any]], Any],
    *,
    to_graph_edges_fn: Callable[[str], nx.Graph] | None = None,
) -> SimpleLoaderReturn:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")

    load_gml = to_graph_edges_fn or (lambda path: nx.read_gml(path, label="id"))
    raw_graph: nx.Graph = load_gml(file_path)
    graph: nx.Graph = raw_graph
    true_labels: list[Any] = [node_label_fn(raw_graph.nodes[node]) for node in raw_graph.nodes()]
    _ensure_default_edge_weight(graph)
    return graph, true_labels, True


def _add_weighted_edges_from_df(
    df: pl.DataFrame,
    graph: nx.Graph,
    source_col: str = "src",
    dst_col: str = "dst",
    *,
    chunk_size: int = 1_000_000,
    progress_every: int | None = None,
    progress_prefix: str = "",
) -> None:
    """Adds weighted edges from dataframe to graph with optional chunked progress logging."""
    total_rows = len(df)
    if total_rows > chunk_size:
        for i in range(0, total_rows, chunk_size):
            chunk = df.slice(i, min(chunk_size, total_rows - i))
            graph.add_weighted_edges_from(chunk.select([source_col, dst_col, "weight"]).rows())
            if progress_every and (i // chunk_size + 1) % progress_every == 0:
                logger.info(f"{progress_prefix}Processed {i + len(chunk)} / {total_rows} edges...")
    else:
        graph.add_weighted_edges_from(df.select([source_col, dst_col, "weight"]).rows())
