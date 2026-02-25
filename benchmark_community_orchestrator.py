# ruff: noqa: E501
import argparse
import logging
import os
import shutil
from collections.abc import Callable
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import rustworkx as rx

from benchmark_community_execution import (
    _calculate_and_store_exact_match_metrics,
    _calculate_and_store_metrics_for_run,
    _register_skipped_algorithm_result,
    _run_single_algorithm_config,
    initialize_results_dict,
)
from benchmark_community_helpers import (
    _algorithm_matches_filters,
    _ensure_default_graph_weights,
    _generate_exact_match_markdown_summary,
    _generate_run_to_run_markdown_summary,
    _get_runner_function_by_name,
    _matches_any_filter,
    _normalize_ground_truth_labels,
    _prepare_cdlib_graph_and_adjacency,
)
from benchmark_config_data import (
    ALGORITHMS_CONFIG_STRUCTURE,
    DATASETS,
    LARGE_EDGES_THRESHOLD,
    LARGE_GRAPH_THRESHOLD,
    RESOLUTIONS_TO_TEST,
    SKIPPED_DATASETS,
)
from benchmark_utils import convert_nx_to_rx
from plotting import (
    create_comparison_chart,
    generate_results_table,
    generate_results_table_matplotlib,
    plot_gcr_pcp_vs_jaccard_threshold,
)

logger = logging.getLogger(__name__)


def run_benchmark(
    dataset_filters: list[str] | None = None,
    algorithm_filters: list[str] | None = None,
) -> None:
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

    active_dataset_filters: list[str] = [
        item.strip() for item in (dataset_filters or []) if item.strip()
    ]
    active_algorithm_filters: list[str] = [
        item.strip() for item in (algorithm_filters or []) if item.strip()
    ]

    if active_dataset_filters:
        logger.info(f"Dataset filters active: {active_dataset_filters}")
    if active_algorithm_filters:
        logger.info(f"Algorithm filters active: {active_algorithm_filters}")

    selected_datasets: list[tuple[str, Callable[..., Any]]] = [
        (dataset_name, load_func)
        for dataset_name, load_func in DATASETS.items()
        if _matches_any_filter(dataset_name, active_dataset_filters)
    ]

    if not selected_datasets:
        logger.warning(
            "No datasets match the provided filters. Nothing to run. "
            f"Requested filters: {active_dataset_filters}"
        )
        return

    benchmark_results: list[dict[str, Any]] = []

    for dataset_name, load_func in selected_datasets:
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
                dataset_name,
                load_func,
                result_folder,
                algorithm_filters=active_algorithm_filters or None,
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
    table_filename: str = os.path.join(result_folder, "benchmark_results_table.md")
    previous_table_snapshot: str = os.path.join(result_folder, "benchmark_results_table_prev.md")
    if os.path.isfile(table_filename):
        try:
            shutil.copy2(table_filename, previous_table_snapshot)
        except Exception as exc:
            logger.warning(
                f"Unable to snapshot previous results table from {table_filename} to "
                f"{previous_table_snapshot}: {exc}"
            )

    table_string: str = generate_results_table(benchmark_results)
    exact_summary: str = _generate_exact_match_markdown_summary(benchmark_results)
    regression_summary: str = _generate_run_to_run_markdown_summary(table_string, result_folder)

    report_sections: list[str] = []
    if exact_summary:
        report_sections.append(exact_summary)
    if regression_summary:
        report_sections.append(regression_summary)
    report_sections.append(table_string)
    table_string = "\n\n".join(report_sections)

    logger.info(f"Markdown Table:\n{table_string}")  # Log the table string itself
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


def run_benchmark_on_dataset(
    dataset_name: str,
    load_func: Callable[..., Any],
    result_folder: str,
    algorithm_filters: list[str] | None = None,
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
        algorithm_filters (Optional[List[str]]): Optional algorithm name/prefix
                                                 filters for this run.

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

        dataset_gt = _normalize_ground_truth_labels(dataset_gt, nx_graph_original_ids)

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
    (
        graph_for_nx_cdlib_eff,
        needs_relabeling_for_cdlib,
        cdlib_node_map_reverse,
        nx_adjacency,
    ) = _prepare_cdlib_graph_and_adjacency(graph_for_nx_cdlib, node_map)
    _ensure_default_graph_weights(graph_for_nx_cdlib_eff)

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

        if not _algorithm_matches_filters(algo_config, algorithm_filters):
            logger.info(
                f"  Skipping {algo_display_name} due to active algorithm filters: {algorithm_filters}"
            )
            _register_skipped_algorithm_result(
                current_algo_prefix=current_algo_prefix,
                results_for_current_dataset=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                partitions_by_algo=partitions_by_algo,
            )
            continue

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
            _register_skipped_algorithm_result(
                current_algo_prefix=current_algo_prefix,
                results_for_current_dataset=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                partitions_by_algo=partitions_by_algo,
            )
            continue

        # Skip NetworkX algorithms for very large datasets (> 1M edges) as they are too slow
        if not is_rx and num_edges > LARGE_EDGES_THRESHOLD:
            logger.warning(
                f"  Skipping {algo_display_name} for large graph ({num_edges} edges > {LARGE_EDGES_THRESHOLD:,}) - NetworkX too slow, using only RustWorkX"
            )
            _register_skipped_algorithm_result(
                current_algo_prefix=current_algo_prefix,
                results_for_current_dataset=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                partitions_by_algo=partitions_by_algo,
            )
            continue

        # Skip NetworkX LPA for large node count graphs (original check)
        if current_algo_prefix == "nx_lpa" and num_nodes > LARGE_GRAPH_THRESHOLD:  # type: ignore
            logger.warning(
                f"  Skipping {algo_display_name} for large graph ({num_nodes} nodes > {LARGE_GRAPH_THRESHOLD})"
            )  # type: ignore
            _register_skipped_algorithm_result(
                current_algo_prefix=current_algo_prefix,
                results_for_current_dataset=results_for_current_dataset,
                nx_graph_original_ids=nx_graph_original_ids,
                rx_graph=rx_graph,
                node_map=node_map,
                is_rx_algo=is_rx,
                dataset_gt=dataset_gt,
                has_ground_truth=has_ground_truth,  # type: ignore
                partitions_by_algo=partitions_by_algo,
            )
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
    parser = argparse.ArgumentParser(
        description=(
            "Run community benchmark across configured datasets/algorithms. "
            "Filters are optional and can be repeated."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset filter token (case-insensitive, substring or wildcard). "
            "Can be repeated, e.g. --dataset 'Wiki*' --dataset LFR."
        ),
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        default=[],
        help=(
            "Algorithm filter token (case-insensitive, matches prefix/base/name). "
            "Can be repeated, e.g. --algorithm rx_louvain --algorithm leiden."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", force=True
    )
    run_benchmark(dataset_filters=args.dataset, algorithm_filters=args.algorithm)
