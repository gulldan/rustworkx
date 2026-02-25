# ruff: noqa: E501
import logging
import time
from collections.abc import Callable
from typing import Any

import networkx as nx
import rustworkx as rx

from benchmark_config_data import ALGORITHMS_CONFIG_STRUCTURE, RESOLUTIONS_TO_TEST
from benchmark_utils import (
    calculate_cluster_matching_metrics,
    calculate_internal_metrics,
    calculate_overall_score,
    calculate_purity,
    compare_with_true_labels,
    format_memory,
    format_time,
    rx_modularity_calculation,
)
from metrics_config import JACCARD_THRESHOLDS_TO_TEST, ORDERED_METRIC_PROPERTIES

logger = logging.getLogger(__name__)


def _register_skipped_algorithm_result(
    *,
    current_algo_prefix: str,
    results_for_current_dataset: dict[str, Any],
    nx_graph_original_ids: nx.Graph,
    rx_graph: rx.PyGraph,
    node_map: dict[Any, int],
    is_rx_algo: bool,
    dataset_gt: dict[Any, int] | list[int] | None,
    has_ground_truth: bool,
    partitions_by_algo: dict[str, list[set[Any]]],
) -> None:
    """Stores default metrics for intentionally skipped algorithm runs."""
    results_for_current_dataset[f"{current_algo_prefix}_elapsed"] = -1
    _calculate_and_store_metrics_for_run(
        current_algo_prefix=current_algo_prefix,
        processed_communities=[],
        communities_raw=[],
        results_dict=results_for_current_dataset,
        nx_graph_original_ids=nx_graph_original_ids,
        rx_graph=rx_graph,
        node_map=node_map,
        is_rx_algo=is_rx_algo,
        dataset_gt=dataset_gt,
        has_ground_truth=has_ground_truth,  # type: ignore
    )
    partitions_by_algo[current_algo_prefix] = []


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

    cdlib_metric = "_exact_match_cdlib_leiden"
    cdlib_part = _get_partition_if_ran(results_dict, partitions_by_algo, "cdlib_leiden")
    if cdlib_part is not None:
        _set_exact_match_metric(results_dict, "cdlib_leiden", cdlib_metric, True)
        for algo_prefix in rx_leiden_prefixes:
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
        for algo_prefix in rx_leiden_prefixes:
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
