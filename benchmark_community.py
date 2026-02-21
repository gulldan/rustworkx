#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Benchmark script comparing rustworkx and networkx Louvain community detection
on various graph datasets with ground truth communities.
"""

import logging  # Add logging import
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

# cdlib imports
import cdlib
import cdlib.algorithms  # Added for Leiden/Infomap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Import shared configurations from benchmark_config_data.py
from benchmark_config_data import (
    ALGORITHMS_CONFIG_STRUCTURE,
    DATASETS,
    LARGE_EDGES_THRESHOLD,
    LARGE_GRAPH_THRESHOLD,
    RESOLUTIONS_TO_TEST,
    SKIPPED_DATASETS,
)

# Import functions from the new modules
from benchmark_utils import (
    calculate_cluster_matching_metrics,
    calculate_internal_metrics,
    calculate_overall_score,
    calculate_purity,
    compare_with_true_labels,
    convert_nx_to_rx,
    format_memory,
    format_time,
    measure_memory,
    rx_modularity_calculation,
)

# Import constants from metrics_config.py
from metrics_config import JACCARD_THRESHOLDS_TO_TEST, ORDERED_METRIC_PROPERTIES
from plotting import (
    create_comparison_chart,
    generate_results_table,
    generate_results_table_matplotlib,
    plot_gcr_pcp_vs_jaccard_threshold,  # Added new import
)

import rustworkx as rx

# Setup logger for this module
logger = logging.getLogger(__name__)
# Basic configuration will be done in main or run_benchmark to avoid multiple basicConfig calls.

# Verify available rustworkx attributes (No print here, but if there was, it would be logger.warning)
try:
    if not hasattr(rx, "PyGraph"):
        logger.warning("rx.PyGraph not found, checking for alternative imports...")
        rx_attrs = dir(rx)
        logger.info(f"Available rustworkx attributes: {', '.join(rx_attrs)}")
except Exception as e:
    logger.error(f"Error checking rustworkx attributes: {e}", exc_info=True)


# Helper to get runner function from its name string (defined globally or in this module)
# This is needed because we store runner names as strings in ALGORITHMS_BENCHMARK_CONFIG
# to avoid issues with function definitions order or pickling if these were direct function objects
# in a more complex setup. For this script, direct assignment would also work but this is cleaner.
def _get_runner_function_by_name(name: str) -> Callable[..., Any]:
    """Resolves a string name to an actual callable function.

    Handles special cases like cdlib_leiden which requires a lambda wrapper.
    Searches for other functions in the global scope of this module.

    Args:
        name (str): The string name of the runner function to retrieve.

    Returns:
        Callable[..., Any]: The callable function corresponding to the given name.

    Raises:
        ValueError: If the runner function name is not found or the resolved
                    object is not callable.
    """
    # Special handling for cdlib_leiden which needs a wrapper
    if name == "run_cdlib_leiden":
        return lambda g, **args: run_cdlib_algorithm(
            g, "leiden", **args
        )  # args not used by infomap, Leiden doesn't take resolution here

    # Other functions are directly available in the global scope of this module
    # or will be once all function definitions are processed.
    runner_func: Callable[..., Any] | None = globals().get(name)
    if runner_func is None or not callable(runner_func):
        raise ValueError(f"Runner function '{name}' not found or not callable in benchmark_community.py")
    return runner_func


# --- End Centralized Algorithm Configuration ---


def _generate_exact_match_markdown_summary(benchmark_results: list[dict[str, Any]]) -> str:
    """Build a compact exact-match summary section for the Markdown report."""
    exact_metric_props = [
        metric_prop
        for metric_prop in ORDERED_METRIC_PROPERTIES
        if isinstance(metric_prop.get("key"), str) and metric_prop["key"].startswith("_exact_match_")
    ]
    if not exact_metric_props:
        return ""

    valid_results = [
        res
        for res in benchmark_results
        if isinstance(res, dict) and res.get("status") != "skipped_placeholder"
    ]
    if not valid_results:
        return ""

    lines: list[str] = [
        "## Exact-Match Summary",
        "",
        "| Pair | Yes | No | N/A | SKIPPED |",
        "|-|-:|-:|-:|-:|",
    ]
    mismatch_lines: list[str] = []

    for metric_prop in exact_metric_props:
        metric_key = metric_prop["key"]
        metric_name = metric_prop.get("name", metric_key)

        yes_count = 0
        no_count = 0
        na_count = 0
        skipped_count = 0
        mismatches_for_metric: list[str] = []

        for dataset_result in valid_results:
            dataset_name = dataset_result.get("dataset", "unknown")
            for algo_cfg in ALGORITHMS_CONFIG_STRUCTURE:
                algo_prefix = algo_cfg["prefix"]
                algo_display_name = algo_cfg["name"]

                # Mirror row-inclusion logic from Markdown table generation:
                # if no metrics were recorded for this algorithm on this dataset,
                # do not include it in summary counts.
                if not any(k.startswith(algo_prefix) for k in dataset_result.keys()):
                    continue

                elapsed_val = dataset_result.get(f"{algo_prefix}_elapsed", -2)
                raw_val = dataset_result.get(f"{algo_prefix}{metric_key}")

                if elapsed_val == -1:
                    skipped_count += 1
                    continue

                if isinstance(raw_val, bool):
                    if raw_val:
                        yes_count += 1
                    else:
                        no_count += 1
                        mismatches_for_metric.append(f"`{dataset_name}` / `{algo_display_name}`")
                    continue

                if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                    na_count += 1
                else:
                    na_count += 1

        lines.append(
            f"| {metric_name} | {yes_count} | {no_count} | {na_count} | {skipped_count} |"
        )

        if mismatches_for_metric:
            mismatch_lines.append(f"### {metric_name}")
            mismatch_lines.extend(f"- {item}" for item in mismatches_for_metric)
            mismatch_lines.append("")

    if mismatch_lines:
        lines.append("")
        lines.append("## Exact-Match Mismatches")
        lines.extend(mismatch_lines)

    return "\n".join(lines).strip()


def run_benchmark() -> None:
    """Runs the full community detection benchmark.

    This function orchestrates the entire benchmarking process including:
    - Setting up logging.
    - Defining datasets and their loading functions.
    - Iterating through datasets, running algorithms, and collecting results.
    - Generating comparison charts, markdown tables, and other plots.
    """
    # Configure logging here if this is the main entry point for standalone runs
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", force=True
    )

    logger.info("=" * 80)
    logger.info("RUSTWORKX vs NETWORKX COMMUNITY DETECTION BENCHMARK")
    logger.info("=" * 80)
    logger.info("\nThis benchmark will compare RustWorkX and NetworkX implementations of")
    logger.info("the Louvain community detection algorithm on various datasets.")
    logger.info("Results will be saved as high-quality image files.\n")

    base_result_folder: str = "results"
    timestamp: str = datetime.now().strftime("%Y%m%d")
    result_folder: str = os.path.join(base_result_folder, timestamp)

    if not os.path.exists(base_result_folder):
        os.makedirs(base_result_folder)
    os.makedirs(result_folder, exist_ok=True)

    logger.info(f"Results will be saved to '{result_folder}'")

    plt.switch_backend("Agg")

    benchmark_results: list[dict[str, Any]] = []

    for dataset_name, load_func in DATASETS.items():
        logger.info(f"\n{'-'.ljust(40, '-')}")
        logger.info(f"Processing {dataset_name} dataset...")
        logger.info(f"{'-'.ljust(40, '-')}")

        if dataset_name in SKIPPED_DATASETS:
            logger.warning(f"🚧 Skipping {dataset_name} as it is marked for skipping (placeholder).")
            benchmark_results.append(
                {
                    "dataset": dataset_name,
                    "nodes": 0,
                    "edges": 0,
                    "status": "skipped_placeholder",
                }
            )
            continue

        try:
            result_data: dict[str, Any] | None = run_benchmark_on_dataset(
                dataset_name, load_func, result_folder
            )
            if result_data:
                benchmark_results.append(result_data)
            logger.info(f"Completed {dataset_name} dataset. Continuing to next dataset...")
        except FileNotFoundError as fnf_error:
            logger.error(f"❌ Skipping {dataset_name}: Dataset file not found - {fnf_error}")
            benchmark_results.append(
                {
                    "dataset": dataset_name,
                    "nodes": 0,
                    "edges": 0,
                    "error": f"File not found: {fnf_error}",
                }
            )
        except Exception as e:
            logger.exception(
                f"❌ Error processing {dataset_name} dataset: {str(e)}"
            )  # logger.exception includes traceback

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING FINAL PERFORMANCE COMPARISON CHART")
    logger.info("=" * 80)
    try:
        chart_path: str = os.path.join(result_folder, "community_detection_comparison.png")
        create_comparison_chart(benchmark_results, output_file=chart_path)
    except Exception as e:
        logger.error(f"Error generating comparison chart: {str(e)}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE - VISUALIZATION FILES GENERATED:")
    logger.info("=" * 80)
    png_files: list[str] = sorted([f for f in os.listdir(result_folder) if f.endswith(".png")])
    if png_files:
        for i, png_file in enumerate(png_files):
            file_path: str = os.path.join(result_folder, png_file)
            file_size: float = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"{i + 1}. {png_file} ({file_size:.2f} MB)")
        logger.info(f"\nYou can find them in the '{result_folder}' directory.")
    else:
        logger.warning("No visualization files were generated. Please check for errors.")

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING RESULTS TABLE (MARKDOWN)")
    logger.info("=" * 80)
    table_string: str = generate_results_table(benchmark_results)
    exact_summary: str = _generate_exact_match_markdown_summary(benchmark_results)
    if exact_summary:
        table_string = f"{exact_summary}\n\n{table_string}"
    logger.info(f"Markdown Table:\n{table_string}")  # Log the table string itself
    table_filename: str = os.path.join(result_folder, "benchmark_results_table.md")
    try:
        with open(table_filename, "w") as f:
            f.write(table_string)
        logger.info(f"\nResults table saved to: {table_filename}")
    except Exception as e:
        logger.error(f"Error saving results table: {e}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING RESULTS TABLE IMAGE (PNG)")
    logger.info("=" * 80)
    table_image_filename: str = os.path.join(result_folder, "benchmark_results_table.png")
    try:
        generate_results_table_matplotlib(benchmark_results, output_file=table_image_filename)
    except Exception as e:
        logger.error(f"Error generating results table image: {str(e)}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING GCR/PCP vs. JACCARD THRESHOLD PLOTS")
    logger.info("=" * 80)
    try:
        plot_gcr_pcp_vs_jaccard_threshold(benchmark_results, result_folder)
    except Exception as e:
        logger.error(f"Error generating GCR/PCP vs. Jaccard Threshold plots: {str(e)}", exc_info=True)

    logger.info("\nBenchmark completed successfully!")


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

    def weight_fn(edge: Any) -> float:
        return float(edge)

    try:
        communities: list[list[int]] = rx.community.louvain_communities(
            graph, weight_fn=weight_fn, seed=42, resolution=resolution, adjacency=adjacency
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
            coms_obj: cdlib.classes.node_clustering.NodeClustering = cdlib.algorithms.leiden(
                graph, weights="weight"
            )
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

    def weight_fn(edge: Any) -> float:
        if isinstance(edge, int | float):
            return float(edge)
        elif isinstance(edge, dict) and "weight" in edge:
            return float(edge["weight"])
        return 1.0

    try:
        communities: list[list[int]] = rx.community.leiden_communities(  # type: ignore
            graph, weight_fn=weight_fn, resolution=resolution, seed=42
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


def _process_communities(
    communities_raw: list[list[Any]] | list[set[Any]],
    is_rx_algo: bool,
    node_map: dict[Any, int],
    needs_relabeling_for_cdlib: bool,
    cdlib_node_map_reverse: dict[int, Any] | None,
) -> list[set[Any]]:
    """Processes raw communities from algorithms into a list of sets of original node IDs.

    Args:
        communities_raw (Union[List[List[Any]], List[set[Any]]]):
            The raw output from a community detection algorithm. This can be
            a list of lists (e.g., from RustWorkX algorithms containing integer indices)
            or a list of sets (e.g., from NetworkX algorithms containing original node IDs
            or relabeled integer IDs).
        is_rx_algo (bool): True if the `communities_raw` came from a RustWorkX algorithm
                           (meaning they are lists of rx_graph internal integer indices).
        node_map (Dict[Any, int]): A mapping from original node IDs to RustWorkX internal
                                   integer indices. Used if `is_rx_algo` is True.
        needs_relabeling_for_cdlib (bool): True if the graph processed by a non-RX (cdlib/NX)
                                           algorithm was relabeled to integer nodes.
        cdlib_node_map_reverse (Optional[Dict[int, Any]]):
            A mapping from relabeled integer node IDs back to original node IDs.
            Used if `is_rx_algo` is False and `needs_relabeling_for_cdlib` is True.

    Returns:
        List[set[Any]]: A list of sets, where each set contains the original
                        node IDs of a detected community. Filters out empty communities.
    """
    processed_communities: list[set[Any]] = []
    if communities_raw:  # If communities were returned (not empty list from error)
        if is_rx_algo:
            reverse_node_map: dict[int, Any] = {
                v: k for k, v in node_map.items()
            }  # rx_idx -> original_id
            processed_communities = [
                {reverse_node_map[node_idx] for node_idx in comm if node_idx in reverse_node_map}
                for comm in communities_raw
                if comm  # Filter inner empty comms
            ]
        elif needs_relabeling_for_cdlib and cdlib_node_map_reverse:
            # communities_raw from NX/cdlib on relabeled graph are sets/lists of integer IDs
            processed_communities = [
                {
                    cdlib_node_map_reverse[node_idx]
                    for node_idx in comm
                    if node_idx in cdlib_node_map_reverse
                }
                for comm in communities_raw
                if comm  # Filter inner empty comms
            ]
        else:  # NX/cdlib on original graph (nodes were already ints or not cdlib)
            processed_communities = [
                set(c) for c in communities_raw if c
            ]  # Ensure sets and filter empty

        processed_communities = [
            c for c in processed_communities if c
        ]  # Final filter for any outer empty sets
    return processed_communities


def _calculate_and_store_metrics_for_run(
    current_algo_prefix: str,
    processed_communities: list[set[Any]],
    communities_raw: list[list[Any]] | list[set[Any]],  # For rx_modularity
    results_dict: dict[str, Any],
    nx_graph_original_ids: nx.Graph,
    rx_graph: rx.PyGraph,  # For rx_modularity
    node_map: dict[Any, int],  # For rx_modularity (unused directly here, but good context)
    is_rx_algo: bool,
    dataset_gt: dict[Any, int] | list[int] | None,
    has_ground_truth: bool,
    rx_weight_fn_for_modularity: Callable[[Any], float] | None = None,  # Specific for RX modularity
) -> None:
    """Calculates and stores all relevant metrics for a single algorithm run.

    This includes performance (num_comms), internal (modularity, conductance, etc.),
    and external (ARI, NMI, purity, GCR/PCP, etc. if ground truth exists) metrics,
    as well as an overall score. Updates `results_dict` in place.

    Args:
        current_algo_prefix (str): A unique string prefix for the algorithm configuration
                                   being run (e.g., "nx_louvain_res0p5").
        processed_communities (List[set[Any]]): A list of sets of original node IDs
                                                representing the detected communities.
        communities_raw (Union[List[List[Any]], List[set[Any]]]):
            Raw community output from the algorithm. For RustWorkX algorithms, this is
            a list of lists of internal rx_graph node indices, used for rx_modularity.
            For other algorithms, this might be the same as `processed_communities`
            or the relabeled integer communities.
        results_dict (Dict[str, Any]): The dictionary where results are stored.
                                       Keys are typically `current_algo_prefix` + `_metric_name`.
        nx_graph_original_ids (nx.Graph): The NetworkX graph with original node IDs.
                                          Used for NetworkX-based metric calculations.
        rx_graph (rx.PyGraph): The RustWorkX graph. Used for RustWorkX modularity.
        node_map (Dict[Any, int]): Mapping from original node IDs to rx_graph indices.
        is_rx_algo (bool): True if the algorithm was a RustWorkX algorithm.
        dataset_gt (Optional[Union[Dict[Any, int], List[int]]]):
            Ground truth community labels. Can be a dictionary mapping node ID to
            community ID, or a list of community IDs (if nodes are 0-indexed integers).
            None if no ground truth is available.
        has_ground_truth (bool): True if ground truth is available for this dataset.
        rx_weight_fn_for_modularity (Optional[Callable[[Any], float]], optional):
            A specific weight function to be used for RustWorkX modularity calculation,
            especially for algorithms like LPA where the main runner's weight function
            might differ. Defaults to None.
    """
    results_dict[f"{current_algo_prefix}_num_comms"] = len(processed_communities)

    if not processed_communities:
        # Overall score calculation will handle NaNs if no communities are found.
        # Set singleton count to 0 if no communities
        results_dict[f"{current_algo_prefix}_num_singleton_comms"] = 0
        pass
    else:
        # Calculate percentage of communities with size < 5
        num_small_comms = sum(1 for comm in processed_communities if len(comm) < 5)
        total_num_comms = len(processed_communities)
        if total_num_comms > 0:
            percentage_small_comms = (num_small_comms / total_num_comms) * 100.0
        else:
            percentage_small_comms = 0.0  # If no communities, 0% are small
        results_dict[f"{current_algo_prefix}_num_singleton_comms"] = percentage_small_comms

    # Modularity
    try:
        if processed_communities:  # Only calculate if there are communities
            if is_rx_algo:
                # communities_raw would be list of lists of RX indices
                # We need to ensure communities_raw is List[List[int]]
                # The type checker complains because it could be List[Set[Any]].
                # We can be sure it's List[List[int]] if is_rx_algo is True.
                rx_comms = (
                    communities_raw
                    if isinstance(communities_raw[0], list)
                    else [list(c) for c in communities_raw]
                )
                results_dict[f"{current_algo_prefix}_modularity"] = rx_modularity_calculation(
                    rx_graph,
                    rx_comms,
                    weight_fn=rx_weight_fn_for_modularity,  # type: ignore
                )
            else:
                # processed_communities are list of sets of original node IDs
                results_dict[f"{current_algo_prefix}_modularity"] = nx.community.modularity(
                    nx_graph_original_ids, processed_communities, weight="weight"
                )
    except Exception as mod_e:
        logger.warning(f"Modularity calculation for {current_algo_prefix} failed: {mod_e}")
        results_dict[f"{current_algo_prefix}_modularity"] = float("nan")

    # Internal Metrics
    try:
        if processed_communities:
            internal_metrics_tuple: tuple[float, ...] = calculate_internal_metrics(
                nx_graph_original_ids, processed_communities, algorithm_marker=current_algo_prefix
            )
        else:  # No communities, fill with NaNs
            internal_metrics_tuple = (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )

        (
            results_dict[f"{current_algo_prefix}_conductance"],
            results_dict[f"{current_algo_prefix}_internal_density"],
            results_dict[f"{current_algo_prefix}_avg_internal_degree"],
            results_dict[f"{current_algo_prefix}_tpr"],
            results_dict[f"{current_algo_prefix}_cut_ratio"],
            results_dict[f"{current_algo_prefix}_surprise"],
            results_dict[f"{current_algo_prefix}_significance"],
        ) = internal_metrics_tuple
    except Exception as int_met_e:
        logger.warning(f"Internal metrics for {current_algo_prefix} failed: {int_met_e}")
        # Ensure all internal metrics are NaN if calculation fails
        for m_key in [
            "_conductance",
            "_internal_density",
            "_avg_internal_degree",
            "_tpr",
            "_cut_ratio",
            "_surprise",
            "_significance",
        ]:
            results_dict[f"{current_algo_prefix}{m_key}"] = float("nan")

    # External Metrics (if ground truth available)
    if has_ground_truth and dataset_gt is not None:
        try:
            if processed_communities:
                (
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
                ) = compare_with_true_labels(
                    processed_communities, dataset_gt, None, current_algo_prefix
                )  # type: ignore

                purity_score, purity_nodes = calculate_purity(processed_communities, dataset_gt)  # type: ignore

                # Store GCR/PCP for different Jaccard thresholds
                gcr_pcp_metrics: dict[str, float] = {}
                for jt in JACCARD_THRESHOLDS_TO_TEST:
                    jt_str: str = str(jt).replace(".", "p")
                    gcr_val, pcp_val = calculate_cluster_matching_metrics(
                        processed_communities, dataset_gt, jaccard_threshold=jt
                    )  # type: ignore
                    gcr_pcp_metrics[f"{current_algo_prefix}_gcr_jt{jt_str}"] = (
                        gcr_val if gcr_val is not None else float("nan")
                    )
                    gcr_pcp_metrics[f"{current_algo_prefix}_pcp_jt{jt_str}"] = (
                        pcp_val if pcp_val is not None else float("nan")
                    )
            else:  # No communities, fill with default values for external metrics
                (
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
                ) = (float("nan"),) * 10 + (0,)
                purity_score, purity_nodes = float("nan"), 0
                gcr_pcp_metrics = {}
                for jt in JACCARD_THRESHOLDS_TO_TEST:
                    jt_str = str(jt).replace(".", "p")
                    gcr_pcp_metrics[f"{current_algo_prefix}_gcr_jt{jt_str}"] = float("nan")
                    gcr_pcp_metrics[f"{current_algo_prefix}_pcp_jt{jt_str}"] = float("nan")

            results_dict[f"{current_algo_prefix}_ari"] = ari
            results_dict[f"{current_algo_prefix}_nmi"] = nmi
            results_dict[f"{current_algo_prefix}_homogeneity"] = homogeneity
            results_dict[f"{current_algo_prefix}_completeness"] = completeness
            results_dict[f"{current_algo_prefix}_v_measure"] = v_measure
            results_dict[f"{current_algo_prefix}_fmi"] = fmi
            results_dict[f"{current_algo_prefix}_vi"] = vi
            results_dict[f"{current_algo_prefix}_pw_precision"] = pw_precision
            results_dict[f"{current_algo_prefix}_pw_recall"] = pw_recall
            results_dict[f"{current_algo_prefix}_pw_f1"] = pw_f1
            results_dict[f"{current_algo_prefix}_common_nodes"] = common_nodes_count
            results_dict[f"{current_algo_prefix}_purity"] = purity_score
            results_dict[f"{current_algo_prefix}_purity_nodes"] = purity_nodes
            results_dict.update(gcr_pcp_metrics)

        except Exception as ext_met_e:
            logger.warning(f"External metrics for {current_algo_prefix} failed: {ext_met_e}")
            # Ensure all external metrics are NaN/0 if calculation fails
            for m_key in [
                "_ari",
                "_nmi",
                "_homogeneity",
                "_completeness",
                "_v_measure",
                "_fmi",
                "_vi",
                "_pw_precision",
                "_pw_recall",
                "_pw_f1",
                "_purity",
            ]:
                results_dict[f"{current_algo_prefix}{m_key}"] = float("nan")
            results_dict[f"{current_algo_prefix}_common_nodes"] = 0
            results_dict[f"{current_algo_prefix}_purity_nodes"] = 0
            for jt in JACCARD_THRESHOLDS_TO_TEST:
                jt_str = str(jt).replace(".", "p")
                results_dict[f"{current_algo_prefix}_gcr_jt{jt_str}"] = 0.0
                results_dict[f"{current_algo_prefix}_pcp_jt{jt_str}"] = 0.0

    # Calculate Percentage of Unclustered Nodes
    try:
        total_nodes_in_graph = nx_graph_original_ids.number_of_nodes()
        if total_nodes_in_graph > 0:
            if not processed_communities:  # No communities found
                percentage_unclustered = 100.0
            else:
                clustered_nodes = set()
                for comm in processed_communities:
                    clustered_nodes.update(comm)
                num_unclustered_nodes = total_nodes_in_graph - len(clustered_nodes)
                percentage_unclustered = (num_unclustered_nodes / total_nodes_in_graph) * 100.0
        else:  # Empty graph
            percentage_unclustered = 0.0  # Or float("nan") depending on desired behavior
        results_dict[f"{current_algo_prefix}_unclustered_pct"] = percentage_unclustered
    except Exception as unclust_e:
        logger.warning(
            f"Unclustered percentage calculation for {current_algo_prefix} failed: {unclust_e}"
        )
        results_dict[f"{current_algo_prefix}_unclustered_pct"] = float("nan")

    # Overall Score - calculated regardless of whether communities were found (will use NaNs/0s)
    overall_score: float = calculate_overall_score(results_dict, current_algo_prefix)
    results_dict[f"{current_algo_prefix}_overall_score"] = overall_score


def _run_single_algorithm_config(
    algo_runner_func: Callable[..., tuple[list[list[Any]] | list[set[Any]], float]],
    run_args: dict[str, Any],
    graph_to_run_on: rx.PyGraph | nx.Graph,
    current_algo_prefix: str,
    is_rx_algo: bool,
    results_for_current_dataset: dict[str, Any],
    node_map: dict[Any, int],
    needs_relabeling_for_cdlib: bool,
    cdlib_node_map_reverse: dict[int, Any] | None,
    nx_graph_original_ids: nx.Graph,
    rx_graph: rx.PyGraph,
    dataset_gt: dict[Any, int] | list[int] | None,
    has_ground_truth: bool,
    rx_specific_weight_fn_for_modularity: Callable[[Any], float] | None = None,
) -> list[set[Any]]:
    """Runs a single algorithm configuration, processes its communities, and calculates all metrics.

    This function encapsulates the logic for executing one variation of a community
    detection algorithm (e.g., NetworkX Louvain with a specific resolution).
    It measures execution time and memory, processes the raw community output,
    and then calls `_calculate_and_store_metrics_for_run` to compute and store
    all relevant metrics. The `results_for_current_dataset` dictionary is updated
    in place.

    Args:
        algo_runner_func (Callable[..., Tuple[Union[List[List[Any]], List[set[Any]]], float]]):
            The actual community detection function to run (e.g., `run_nx_algorithm`).
            It's expected to return a tuple: (raw_communities, memory_usage_mb).
        run_args (Dict[str, Any]): Arguments to pass to `algo_runner_func`
                                   (e.g., `{"resolution": 0.5}`).
        graph_to_run_on (Union[rx.PyGraph, nx.Graph]): The graph object
            (either RustWorkX or NetworkX) on which the algorithm will be run.
        current_algo_prefix (str): A unique string prefix for this algorithm
                                   configuration (e.g., "nx_louvain_res0p5").
        is_rx_algo (bool): True if `algo_runner_func` is a RustWorkX algorithm.
        results_for_current_dataset (Dict[str, Any]): The main results dictionary
                                                      for the current dataset, which
                                                      this function will update.
        node_map (Dict[Any, int]): Mapping from original node IDs to RustWorkX
                                   internal indices. Used if `is_rx_algo` is True.
        needs_relabeling_for_cdlib (bool): True if `graph_to_run_on` (for non-RX algos)
                                           has been relabeled to integer nodes.
        cdlib_node_map_reverse (Optional[Dict[int, Any]]): Mapping from relabeled
                                                            integer nodes back to original IDs.
        nx_graph_original_ids (nx.Graph): The NetworkX graph with original node IDs.
        rx_graph (rx.PyGraph): The RustWorkX graph.
        dataset_gt (Optional[Union[Dict[Any, int], List[int]]]): Ground truth labels.
        has_ground_truth (bool): True if ground truth is available.
        rx_specific_weight_fn_for_modularity (Optional[Callable[[Any], float]], optional):
            Specific weight function for RX modularity, if needed. Defaults to None.
    """
    logger.info(f"Running {current_algo_prefix}...")

    start_time: float = time.time()
    communities_raw: list[list[Any]] | list[set[Any]]
    mem_usage: float
    communities_raw, mem_usage = algo_runner_func(graph_to_run_on, **run_args)

    results_for_current_dataset[f"{current_algo_prefix}_elapsed"] = time.time() - start_time
    results_for_current_dataset[f"{current_algo_prefix}_memory"] = mem_usage

    processed_communities: list[set[Any]] = _process_communities(
        communities_raw, is_rx_algo, node_map, needs_relabeling_for_cdlib, cdlib_node_map_reverse
    )

    num_pred_comms: int = len(processed_communities)
    results_for_current_dataset[f"{current_algo_prefix}_num_comms"] = (
        num_pred_comms  # Stored here, also in _calculate_and_store
    )

    logger.info(
        f"{current_algo_prefix}: {num_pred_comms} communities "
        f"in {format_time(results_for_current_dataset[f'{current_algo_prefix}_elapsed'])} "
        f"using {format_memory(results_for_current_dataset[f'{current_algo_prefix}_memory'])}"
    )

    actual_rx_weight_fn_modularity: Callable[[Any], float] | None = None
    if is_rx_algo:

        def _rx_lpa_weighted_edge_weight(edge: Any) -> float:
            return float(edge)

        def _rx_unweighted_edge_weight(edge: Any) -> float:
            return 1.0

        def _rx_generic_edge_weight(edge: Any) -> float:
            return float(edge)

        if current_algo_prefix.startswith("rx_lpa_weighted"):
            actual_rx_weight_fn_modularity = _rx_lpa_weighted_edge_weight
        elif current_algo_prefix.startswith("rx_lpa_unweighted"):
            actual_rx_weight_fn_modularity = _rx_unweighted_edge_weight
        elif current_algo_prefix.startswith("rx_louvain") or current_algo_prefix.startswith("rx_leiden"):
            actual_rx_weight_fn_modularity = _rx_generic_edge_weight
        # Add other rx algos here if needed

    _calculate_and_store_metrics_for_run(
        current_algo_prefix=current_algo_prefix,
        processed_communities=processed_communities,
        communities_raw=communities_raw,
        results_dict=results_for_current_dataset,
        nx_graph_original_ids=nx_graph_original_ids,
        rx_graph=rx_graph,
        node_map=node_map,
        is_rx_algo=is_rx_algo,
        dataset_gt=dataset_gt,
        has_ground_truth=has_ground_truth,
        rx_weight_fn_for_modularity=actual_rx_weight_fn_modularity,
    )
    return processed_communities


def _partition_signature(partition: list[set[Any]]) -> set[frozenset[Any]]:
    """Convert partition to an order-independent signature."""
    return {frozenset(comm) for comm in partition if comm}


def _exact_partition_match(partition_a: list[set[Any]], partition_b: list[set[Any]]) -> bool:
    """Check exact set-wise equality of two partitions."""
    return _partition_signature(partition_a) == _partition_signature(partition_b)


def _get_partition_if_ran(
    results_dict: dict[str, Any],
    partitions_by_algo: dict[str, list[set[Any]]],
    algo_prefix: str,
) -> list[set[Any]] | None:
    elapsed_key = f"{algo_prefix}_elapsed"
    elapsed_val = results_dict.get(elapsed_key, -2)
    if elapsed_val in (-2, -1):
        return None
    return partitions_by_algo.get(algo_prefix)


def _set_exact_match_metric(
    results_dict: dict[str, Any],
    algo_prefix: str,
    metric_suffix: str,
    value: bool | None,
) -> None:
    if value is None:
        results_dict[f"{algo_prefix}{metric_suffix}"] = float("nan")
    else:
        results_dict[f"{algo_prefix}{metric_suffix}"] = bool(value)


def _calculate_and_store_exact_match_metrics(
    results_dict: dict[str, Any],
    partitions_by_algo: dict[str, list[set[Any]]],
) -> None:
    """Store exact partition-match diagnostics for key cross-library pairs."""
    # Louvain: compare against NetworkX Louvain at each configured resolution.
    for res_val in RESOLUTIONS_TO_TEST:
        res_key = str(res_val).replace(".", "p")
        nx_prefix = f"nx_louvain_res{res_key}"
        rx_prefix = f"rx_louvain_res{res_key}"
        metric = "_exact_match_nx_louvain"

        nx_part = _get_partition_if_ran(results_dict, partitions_by_algo, nx_prefix)
        if nx_part is not None:
            _set_exact_match_metric(results_dict, nx_prefix, metric, True)
            rx_part = _get_partition_if_ran(results_dict, partitions_by_algo, rx_prefix)
            if rx_part is not None:
                _set_exact_match_metric(
                    results_dict, rx_prefix, metric, _exact_partition_match(rx_part, nx_part)
                )

    # LPA: compare canonical RX LPA (weighted) against NetworkX LPA.
    # Strongest-edge and unweighted variants are intentionally different methods.
    lpa_metric = "_exact_match_nx_lpa"
    nx_lpa_prefix = "nx_lpa"
    nx_lpa_part = _get_partition_if_ran(results_dict, partitions_by_algo, nx_lpa_prefix)
    if nx_lpa_part is not None:
        _set_exact_match_metric(results_dict, nx_lpa_prefix, lpa_metric, True)
        for algo_prefix in ["rx_lpa_weighted"]:
            algo_part = _get_partition_if_ran(results_dict, partitions_by_algo, algo_prefix)
            if algo_part is not None:
                _set_exact_match_metric(
                    results_dict,
                    algo_prefix,
                    lpa_metric,
                    _exact_partition_match(algo_part, nx_lpa_part),
                )

    # Maximal cliques: compare RX against NetworkX.
    cliques_metric = "_exact_match_nx_cliques"
    nx_cliques_prefix = "nx_cliques"
    nx_cliques_part = _get_partition_if_ran(results_dict, partitions_by_algo, nx_cliques_prefix)
    if nx_cliques_part is not None:
        _set_exact_match_metric(results_dict, nx_cliques_prefix, cliques_metric, True)
        rx_cliques_part = _get_partition_if_ran(results_dict, partitions_by_algo, "rx_cliques")
        if rx_cliques_part is not None:
            _set_exact_match_metric(
                results_dict,
                "rx_cliques",
                cliques_metric,
                _exact_partition_match(rx_cliques_part, nx_cliques_part),
            )

    # CPM (k=3): compare RX against NetworkX.
    cpm_metric = "_exact_match_nx_cpm"
    nx_cpm_prefix = "nx_cpm_k3"
    nx_cpm_part = _get_partition_if_ran(results_dict, partitions_by_algo, nx_cpm_prefix)
    if nx_cpm_part is not None:
        _set_exact_match_metric(results_dict, nx_cpm_prefix, cpm_metric, True)
        rx_cpm_part = _get_partition_if_ran(results_dict, partitions_by_algo, "rx_cpm_k3")
        if rx_cpm_part is not None:
            _set_exact_match_metric(
                results_dict,
                "rx_cpm_k3",
                cpm_metric,
                _exact_partition_match(rx_cpm_part, nx_cpm_part),
            )

    # Leiden: compare against cdlib Leiden and against original leidenalg.
    rx_leiden_prefixes = [f"rx_leiden_res{str(res).replace('.', 'p')}" for res in RESOLUTIONS_TO_TEST]
    leiden_candidates = ["cdlib_leiden", "leidenalg"] + rx_leiden_prefixes

    cdlib_metric = "_exact_match_cdlib_leiden"
    cdlib_part = _get_partition_if_ran(results_dict, partitions_by_algo, "cdlib_leiden")
    if cdlib_part is not None:
        _set_exact_match_metric(results_dict, "cdlib_leiden", cdlib_metric, True)
        for algo_prefix in leiden_candidates:
            if algo_prefix == "cdlib_leiden":
                continue
            algo_part = _get_partition_if_ran(results_dict, partitions_by_algo, algo_prefix)
            if algo_part is not None:
                _set_exact_match_metric(
                    results_dict,
                    algo_prefix,
                    cdlib_metric,
                    _exact_partition_match(algo_part, cdlib_part),
                )

    leidenalg_metric = "_exact_match_leidenalg"
    leidenalg_part = _get_partition_if_ran(results_dict, partitions_by_algo, "leidenalg")
    if leidenalg_part is not None:
        _set_exact_match_metric(results_dict, "leidenalg", leidenalg_metric, True)
        for algo_prefix in leiden_candidates:
            if algo_prefix == "leidenalg":
                continue
            algo_part = _get_partition_if_ran(results_dict, partitions_by_algo, algo_prefix)
            if algo_part is not None:
                _set_exact_match_metric(
                    results_dict,
                    algo_prefix,
                    leidenalg_metric,
                    _exact_partition_match(algo_part, leidenalg_part),
                )


def initialize_results_dict(
    dataset_name: str,
    num_nodes: int = 0,
    num_edges: int = 0,
    has_ground_truth: bool = False,
    num_gt_clusters: int = 0,
    num_nodes_in_gt: int = 0,
) -> dict[str, Any]:
    """Initializes the results dictionary with all necessary keys.

    This function populates a dictionary with predefined keys for dataset information
    (name, nodes, edges, etc.) and placeholders for all metrics that will be
    calculated for each algorithm. It uses `ORDERED_METRIC_PROPERTIES` (from
    `metrics_config.py`) and `ALGORITHMS_CONFIG_STRUCTURE` (from
    `benchmark_config_data.py`) to determine the full set of metric keys.

    Args:
        dataset_name (str): The name of the dataset.
        num_nodes (int, optional): The number of nodes in the dataset's graph. Defaults to 0.
        num_edges (int, optional): The number of edges in the dataset's graph. Defaults to 0.
        has_ground_truth (bool, optional): Whether ground truth communities are available
                                          for this dataset. Defaults to False.
        num_gt_clusters (int, optional): The number of ground truth clusters, if available.
                                         Defaults to 0.
        num_nodes_in_gt (int, optional): The number of nodes covered by the ground truth
                                        clustering, if available. Defaults to 0.

    Returns:
        Dict[str, Any]: An initialized dictionary with keys for dataset info and
                        placeholders (NaN, 0, or None) for all anticipated metrics.
    """
    results: dict[str, Any] = {
        mp["key"]: None
        for mp in ORDERED_METRIC_PROPERTIES
        if mp["type"] == "info"
        and mp["key"]
        not in ["dataset", "nodes", "edges", "num_gt_clusters", "num_nodes_in_gt", "algorithm_name"]
    }
    results.update(
        {
            "dataset": dataset_name,
            "nodes": num_nodes,
            "edges": num_edges,
            "has_ground_truth": has_ground_truth,
            "num_gt_clusters": num_gt_clusters,
            "num_nodes_in_gt": num_nodes_in_gt,
        }
    )

    algo_prefixes_to_initialize: list[str] = [
        algo_conf["prefix"] for algo_conf in ALGORITHMS_CONFIG_STRUCTURE
    ]

    default_values: dict[str, Any] = {
        "perf": 0.0,
        "structure": 0,
        "internal": float("nan"),
        "external": float("nan"),
        "summary": float("nan"),
    }
    specific_key_defaults: dict[str, Any] = {
        "_common_nodes": 0,
        "_num_comms": 0,
        "_num_singleton_comms": 0,
        "_purity_nodes": 0,
        "_unclustered_pct": float("nan"),
        "_exact_match_nx_louvain": float("nan"),
        "_exact_match_nx_lpa": float("nan"),
        "_exact_match_nx_cliques": float("nan"),
        "_exact_match_nx_cpm": float("nan"),
        "_exact_match_cdlib_leiden": float("nan"),
        "_exact_match_leidenalg": float("nan"),
    }
    for jt_key_part in [f"_gcr_jt{str(jt).replace('.', 'p')}" for jt in JACCARD_THRESHOLDS_TO_TEST] + [
        f"_pcp_jt{str(jt).replace('.', 'p')}" for jt in JACCARD_THRESHOLDS_TO_TEST
    ]:
        specific_key_defaults[jt_key_part] = 0.0

    for algo_prefix in algo_prefixes_to_initialize:
        for metric_prop in ORDERED_METRIC_PROPERTIES:
            metric_key_suffix: str = metric_prop["key"]
            metric_type: str = metric_prop["type"]

            if metric_type == "info":
                continue

            full_metric_key: str = f"{algo_prefix}{metric_key_suffix}"

            default_val: Any = specific_key_defaults.get(
                metric_key_suffix, default_values.get(metric_type, float("nan"))
            )
            results[full_metric_key] = default_val

    return results


def run_benchmark_on_dataset(
    dataset_name: str,
    load_func: Callable[..., Any],
    result_folder: str,
) -> dict[str, Any] | None:
    """Runs the benchmark for all configured algorithms on a single dataset.

    This function handles:
    1. Loading the dataset graph and ground truth (if any) using `load_func`.
    2. Initializing a results dictionary for this dataset.
    3. Converting the graph to RustWorkX format and preparing a NetworkX copy.
    4. Handling node relabeling if required for cdlib/NetworkX algorithms.
    5. Iterating through all algorithm configurations defined in `ALGORITHMS_CONFIG_STRUCTURE`.
    6. For each configuration:
        - Skipping if specific conditions are met (e.g., NX LPA on large graphs).
        - Running the algorithm via `_run_single_algorithm_config`.
        - Storing results and handling exceptions.
    7. Performing a diagnostic log for a sample parameterized algorithm.

    Args:
        dataset_name (str): The name of the dataset to benchmark.
        load_func (Callable[..., Any]): The function responsible for loading the
                                                 dataset. It can return a dictionary
                                                 or a tuple (graph, labels, has_gt).
        result_folder (str): The path to the folder where results (like plots)
                             for this dataset might be saved (though this specific
                             function mainly returns the results dictionary).

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing all benchmark results
                                  for the dataset. Returns None or a dictionary
                                  with an "error" key if critical loading errors occur.
    """
    logger.info(f"\n--- Starting {dataset_name} ---")
    results_for_current_dataset: dict[str, Any] = {}
    nx_graph_original_ids: nx.Graph | None = None
    dataset_gt: dict[Any, int] | list[int] | None = None
    data_obj: dict[str, Any]

    try:
        logger.info(f"  Loading dataset: {dataset_name}...")

        # Loaders that take 'dtype_spec' and return a dict
        dict_based_loaders = {
            "Graph Edges GT Clusters",
            "Graph Edges LLM Clusters",
            "Graph Edges No GT",
        }

        if dataset_name in dict_based_loaders:
            data_obj = load_func(dtype_spec={"node": str, "src": str, "dst": str})
        else:
            # This is a tuple-based loader, it doesn't take dtype_spec
            graph, true_labels, has_gt = load_func()

            # Convert to dict format to standardize
            info = {"has_ground_truth": has_gt}
            if has_gt and true_labels:
                # Assuming true_labels is a list where len is number of nodes
                info["num_gt_clusters"] = len(set(true_labels))
                info["num_nodes_in_gt"] = len(true_labels)

            data_obj = {
                "graph": graph,
                "ground_truth": true_labels,
                "info": info,
            }

        nx_graph_original_ids = data_obj.get("graph")
        dataset_gt = data_obj.get("ground_truth")
        dataset_info: dict[str, Any] = data_obj.get("info", {})

        if nx_graph_original_ids is None:
            logger.error(f"  Error: Failed to load graph for {dataset_name}.")
            results_for_current_dataset["error"] = "Failed to load graph."
            results_for_current_dataset["dataset"] = dataset_name  # Ensure name for error reporting
            return results_for_current_dataset

        num_nodes: int = nx_graph_original_ids.number_of_nodes()
        num_edges: int = nx_graph_original_ids.number_of_edges()
        logger.info(f"  Dataset loaded: {num_nodes} nodes, {num_edges} edges.")

        has_ground_truth: bool = dataset_gt is not None and bool(dataset_gt)
        num_gt_clusters: int = dataset_info.get("num_gt_clusters", 0)
        num_nodes_in_gt: int = dataset_info.get("num_nodes_in_gt", 0) if has_ground_truth else 0

        # Align ground-truth labels to original node IDs so external metrics work even when
        # nodes are non-integer (e.g., Davis/Florentine string node IDs). When loaders return
        # a list/array, its order follows the graph node iteration order; convert to a dict
        # keyed by the original node objects to match `processed_communities`.
        if has_ground_truth and isinstance(dataset_gt, (list, np.ndarray)):
            nodes_order = list(nx_graph_original_ids.nodes())
            if len(dataset_gt) == len(nodes_order):
                dataset_gt = {node_id: label for node_id, label in zip(nodes_order, dataset_gt)}

        results_for_current_dataset = initialize_results_dict(
            dataset_name, num_nodes, num_edges, has_ground_truth, num_gt_clusters, num_nodes_in_gt
        )

    except FileNotFoundError as fnf_error:
        logger.error(f"  ❌ Critical Error: Dataset file not found for {dataset_name} - {fnf_error}")
        # Initialize a minimal dict for error reporting if not already done
        if not results_for_current_dataset:
            results_for_current_dataset = {"dataset": dataset_name}
        results_for_current_dataset["error"] = f"File not found: {fnf_error}"
        return results_for_current_dataset
    except Exception as e_load:
        logger.exception(f"  ❌ Critical Error loading {dataset_name}: {str(e_load)}")
        if not results_for_current_dataset:
            results_for_current_dataset = {"dataset": dataset_name}
        results_for_current_dataset["error"] = f"Load error: {str(e_load)}"
        return results_for_current_dataset

    # Ensure nx_graph_original_ids is not None before proceeding
    if nx_graph_original_ids is None:  # Should have been caught by earlier checks
        logger.error(
            f"  Critical error: nx_graph_original_ids is None for {dataset_name} after load attempt."
        )
        if not results_for_current_dataset:
            results_for_current_dataset = {"dataset": dataset_name}
        results_for_current_dataset["error"] = "Graph object is None after loading."
        return results_for_current_dataset

    rx_graph: rx.PyGraph
    node_map: dict[Any, int]
    rx_graph, node_map = convert_nx_to_rx(nx_graph_original_ids)
    graph_for_nx_cdlib: nx.Graph = nx.Graph(nx_graph_original_ids)

    needs_relabeling_for_cdlib: bool = False
    cdlib_node_map_reverse: dict[int, Any] | None = None
    original_nodes_for_nx_cdlib: list[Any] = list(graph_for_nx_cdlib.nodes())

    if graph_for_nx_cdlib.nodes():
        first_node: Any = next(iter(original_nodes_for_nx_cdlib))
        if not isinstance(first_node, int):
            needs_relabeling_for_cdlib = True

    graph_for_nx_cdlib_eff: nx.Graph
    if needs_relabeling_for_cdlib:
        logger.info("Info: Relabeling nodes to integers for cdlib/NX algorithms that require it...")
        graph_for_nx_cdlib_relabeled: nx.Graph = nx.convert_node_labels_to_integers(
            graph_for_nx_cdlib, first_label=0, ordering="default", label_attribute="original_label"
        )
        cdlib_node_map_reverse = {
            i: graph_for_nx_cdlib_relabeled.nodes[i]["original_label"]
            for i in graph_for_nx_cdlib_relabeled.nodes()
        }
        graph_for_nx_cdlib_eff = graph_for_nx_cdlib_relabeled
    else:
        graph_for_nx_cdlib_eff = graph_for_nx_cdlib

    # Build adjacency list from graph_for_nx_cdlib_eff (the graph NX actually runs on)
    # This is critical because even nx.Graph(g) creates a copy with different adjacency order!
    # For RX algorithms, we need to map node IDs to RX indices via node_map.
    nx_adjacency: list[list[int]] = []
    if needs_relabeling_for_cdlib and cdlib_node_map_reverse is not None:
        # When relabeled: adjacency[i] should have neighbors of relabeled node i
        # Convert back to original IDs, then to RX indices
        for relabeled_node in graph_for_nx_cdlib_eff.nodes():
            neighbors_relabeled = list(graph_for_nx_cdlib_eff[relabeled_node].keys())
            neighbors_rx = [node_map[cdlib_node_map_reverse[nbr]] for nbr in neighbors_relabeled]
            nx_adjacency.append(neighbors_rx)
    else:
        # No relabeling: but still use graph_for_nx_cdlib_eff (the copy NX runs on)
        # because nx.Graph() copy can have different adjacency order than original!
        for node in graph_for_nx_cdlib_eff.nodes():
            neighbors = [node_map[nbr] for nbr in graph_for_nx_cdlib_eff[node].keys()]
            nx_adjacency.append(neighbors)

    default_weight: float = 1.0
    for u, v, data in graph_for_nx_cdlib_eff.edges(data=True):
        if "weight" not in data or data["weight"] is None:
            data["weight"] = default_weight

    partitions_by_algo: dict[str, list[set[Any]]] = {}

    for algo_config in ALGORITHMS_CONFIG_STRUCTURE:
        current_algo_prefix: str = algo_config["prefix"]
        runner_name: str = algo_config["runner_name"]
        run_args: dict[str, Any] = algo_config["run_args"].copy()  # Copy to avoid mutation
        is_rx: bool = algo_config["is_rx"]
        algo_display_name: str = algo_config["name"]
        needs_adjacency: bool = algo_config.get("needs_adjacency", False)

        # If algorithm needs adjacency for exact NX matching, add it to run_args
        if needs_adjacency and is_rx:
            run_args["adjacency"] = nx_adjacency

        logger.info(
            f"\n--- Processing {results_for_current_dataset['dataset']} with {algo_display_name} ---"
        )

        max_nodes_for_algo: int | None = algo_config.get("max_nodes")
        max_edges_for_algo: int | None = algo_config.get("max_edges")
        if (max_nodes_for_algo is not None and num_nodes > max_nodes_for_algo) or (
            max_edges_for_algo is not None and num_edges > max_edges_for_algo
        ):
            limit_desc: list[str] = []
            if max_nodes_for_algo is not None:
                limit_desc.append(f"nodes={num_nodes} > {max_nodes_for_algo}")
            if max_edges_for_algo is not None:
                limit_desc.append(f"edges={num_edges} > {max_edges_for_algo}")
            logger.warning(
                f"  Skipping {algo_display_name} due to configured complexity limits ({', '.join(limit_desc)})."
            )
            results_for_current_dataset[f"{current_algo_prefix}_elapsed"] = -1
            _calculate_and_store_metrics_for_run(
                current_algo_prefix=current_algo_prefix,
                processed_communities=[],
                communities_raw=[],
                results_dict=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                rx_weight_fn_for_modularity=None,
            )
            partitions_by_algo[current_algo_prefix] = []
            continue

        # Skip NetworkX algorithms for very large datasets (> 1M edges) as they are too slow
        if not is_rx and num_edges > LARGE_EDGES_THRESHOLD:
            logger.warning(
                f"  Skipping {algo_display_name} for large graph ({num_edges} edges > {LARGE_EDGES_THRESHOLD:,}) - NetworkX too slow, using only RustWorkX"
            )
            results_for_current_dataset[f"{current_algo_prefix}_elapsed"] = -1
            _calculate_and_store_metrics_for_run(
                current_algo_prefix=current_algo_prefix,
                processed_communities=[],
                communities_raw=[],
                results_dict=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                rx_weight_fn_for_modularity=None,
            )
            partitions_by_algo[current_algo_prefix] = []
            continue

        # Skip NetworkX LPA for large node count graphs (original check)
        if current_algo_prefix == "nx_lpa" and num_nodes > LARGE_GRAPH_THRESHOLD:  # type: ignore
            logger.warning(
                f"  Skipping {algo_display_name} for large graph ({num_nodes} nodes > {LARGE_GRAPH_THRESHOLD})"
            )  # type: ignore
            results_for_current_dataset[f"{current_algo_prefix}_elapsed"] = -1
            _calculate_and_store_metrics_for_run(
                current_algo_prefix=current_algo_prefix,
                processed_communities=[],
                communities_raw=[],
                results_dict=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                rx_weight_fn_for_modularity=None,
            )
            partitions_by_algo[current_algo_prefix] = []
            continue

        try:
            algo_runner_func: Callable[..., tuple[list[list[Any]] | list[set[Any]], float]] = (
                _get_runner_function_by_name(runner_name)
            )  # type: ignore

            processed_communities_for_algo = _run_single_algorithm_config(
                algo_runner_func=algo_runner_func,
                run_args=run_args,
                graph_to_run_on=rx_graph if is_rx else graph_for_nx_cdlib_eff,
                current_algo_prefix=current_algo_prefix,
                is_rx_algo=is_rx,
                results_for_current_dataset=results_for_current_dataset,
                node_map=node_map,
                needs_relabeling_for_cdlib=needs_relabeling_for_cdlib,
                cdlib_node_map_reverse=cdlib_node_map_reverse,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
            )
            partitions_by_algo[current_algo_prefix] = processed_communities_for_algo
        except Exception as e:
            logger.exception(f"  Error running {algo_display_name} ({current_algo_prefix}): {str(e)}")
            results_for_current_dataset[f"{current_algo_prefix}_elapsed"] = (
                results_for_current_dataset.get(f"{current_algo_prefix}_elapsed", 0)
            )  # type: ignore
            results_for_current_dataset[f"{current_algo_prefix}_memory"] = (
                results_for_current_dataset.get(f"{current_algo_prefix}_memory", 0)
            )  # type: ignore
            _calculate_and_store_metrics_for_run(
                current_algo_prefix=current_algo_prefix,
                processed_communities=[],
                communities_raw=[],
                results_dict=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                rx_weight_fn_for_modularity=None,
            )
            partitions_by_algo[current_algo_prefix] = []

    _calculate_and_store_exact_match_metrics(results_for_current_dataset, partitions_by_algo)

    if len(RESOLUTIONS_TO_TEST) > 1:
        default_res_for_diag: float = RESOLUTIONS_TO_TEST[1]
        default_res_str: str = str(default_res_for_diag).replace(".", "p")
        test_prefix: str = f"nx_louvain_res{default_res_str}"
        logger.debug(f"\nDEBUG: Sample Parameterized Algo ({test_prefix}) results snippet:")
        for k, v in results_for_current_dataset.items():
            if k.startswith(test_prefix) and any(
                suffix in k for suffix in ["_elapsed", "_memory", "_num_comms", "_ari", "_modularity"]
            ):
                logger.debug(f"  {k}: {v}")
        logger.debug("DEBUG: End of sample snippet.\n")

    logger.info(f"\nFinished benchmark for {dataset_name}.")
    return results_for_current_dataset


def main() -> None:
    """Main function to run the benchmark.

    Initializes logging and calls `run_benchmark()`.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", force=True
    )
    run_benchmark()


if __name__ == "__main__":
    main()
