# ruff: noqa: E501
import fnmatch
import logging
import os
import re
from collections.abc import Callable
from statistics import median
from typing import Any

import networkx as nx
import numpy as np

from benchmark_community_algorithms import (
    run_cdlib_algorithm,
    run_leidenalg_algorithm,
    run_nx_algorithm,
    run_nx_cliques_algorithm,
    run_nx_cpm_algorithm,
    run_nx_lpa_algorithm,
    run_rx_algorithm,
    run_rx_cliques_algorithm,
    run_rx_cpm_algorithm,
    run_rx_leiden_algorithm,
    run_rx_lpa_algorithm,
    run_rx_lpa_strongest_algorithm,
)
from benchmark_config_data import ALGORITHMS_CONFIG_STRUCTURE, RESOLUTIONS_TO_TEST
from benchmark_utils import format_memory
from metrics_config import ORDERED_METRIC_PROPERTIES

logger = logging.getLogger(__name__)

RUNNER_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "run_nx_algorithm": run_nx_algorithm,
    "run_rx_algorithm": run_rx_algorithm,
    "run_cdlib_algorithm": run_cdlib_algorithm,
    "run_rx_leiden_algorithm": run_rx_leiden_algorithm,
    "run_rx_lpa_algorithm": run_rx_lpa_algorithm,
    "run_rx_lpa_strongest_algorithm": run_rx_lpa_strongest_algorithm,
    "run_nx_lpa_algorithm": run_nx_lpa_algorithm,
    "run_nx_cliques_algorithm": run_nx_cliques_algorithm,
    "run_rx_cliques_algorithm": run_rx_cliques_algorithm,
    "run_nx_cpm_algorithm": run_nx_cpm_algorithm,
    "run_rx_cpm_algorithm": run_rx_cpm_algorithm,
    "run_leidenalg_algorithm": run_leidenalg_algorithm,
}


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
    if name == "run_cdlib_leiden":
        return lambda g, **args: run_cdlib_algorithm(g, "leiden", **args)

    runner_func = RUNNER_FUNCTIONS.get(name)
    if runner_func is None:
        raise ValueError(
            f"Runner function '{name}' not found or not callable in benchmark_community.py"
        )
    return runner_func


def _matches_any_filter(value: str, filters: list[str] | None) -> bool:
    """Return True if `value` matches any filter token.

    Matching is case-insensitive and supports either substring matching
    or shell-style wildcards (fnmatch).
    """
    if not filters:
        return True
    value_norm = value.lower()
    for token in filters:
        token_norm = token.strip().lower()
        if not token_norm:
            continue
        if token_norm in value_norm or fnmatch.fnmatch(value_norm, token_norm):
            return True
    return False


def _algorithm_matches_filters(
    algo_config: dict[str, Any],
    algorithm_filters: list[str] | None,
) -> bool:
    """Return True if an algorithm config matches provided filters."""
    if not algorithm_filters:
        return True

    candidates = [
        str(algo_config.get("prefix", "")),
        str(algo_config.get("base_prefix", "")),
        str(algo_config.get("name", "")),
    ]
    return any(_matches_any_filter(candidate, algorithm_filters) for candidate in candidates)


def _exact_match_relevant_prefixes(metric_key: str) -> set[str]:
    """Return algorithm prefixes that should contribute to a given exact-match pair metric."""
    if metric_key == "_exact_match_nx_louvain":
        prefixes: set[str] = set()
        for res_val in RESOLUTIONS_TO_TEST:
            res_key = str(res_val).replace(".", "p")
            prefixes.add(f"nx_louvain_res{res_key}")
            prefixes.add(f"rx_louvain_res{res_key}")
        return prefixes

    if metric_key == "_exact_match_nx_lpa":
        return {"nx_lpa", "rx_lpa_weighted"}

    if metric_key == "_exact_match_cdlib_leiden":
        leiden_prefixes = {
            f"rx_leiden_res{str(res_val).replace('.', 'p')}" for res_val in RESOLUTIONS_TO_TEST
        }
        leiden_prefixes.add("cdlib_leiden")
        return leiden_prefixes

    if metric_key == "_exact_match_leidenalg":
        leiden_prefixes = {
            f"rx_leiden_res{str(res_val).replace('.', 'p')}" for res_val in RESOLUTIONS_TO_TEST
        }
        leiden_prefixes.add("leidenalg")
        return leiden_prefixes

    if metric_key == "_exact_match_nx_cliques":
        return {"nx_cliques", "rx_cliques"}

    if metric_key == "_exact_match_nx_cpm":
        return {"nx_cpm_k3", "rx_cpm_k3"}

    return set()


def _ensure_default_graph_weights(
    graph: nx.Graph,
    default_weight: float = 1.0,
) -> None:
    """Ensures each edge in `graph` has a non-null weight attribute."""
    for u, v, data in graph.edges(data=True):
        if "weight" not in data or data["weight"] is None:
            data["weight"] = default_weight


def _prepare_cdlib_graph_and_adjacency(
    graph_for_nx_cdlib: nx.Graph,
    node_map: dict[Any, int],
) -> tuple[nx.Graph, bool, dict[int, Any] | None, list[list[int]]]:
    """Copies graph, relabels if needed, and builds a neighbor index adjacency list."""
    needs_relabeling_for_cdlib: bool = False
    cdlib_node_map_reverse: dict[int, Any] | None = None
    graph_for_nx_cdlib_eff: nx.Graph = graph_for_nx_cdlib

    if graph_for_nx_cdlib.nodes():
        first_node: Any = next(iter(graph_for_nx_cdlib.nodes()))
        if not isinstance(first_node, int):
            needs_relabeling_for_cdlib = True
            logger.info("Info: Relabeling nodes to integers for cdlib/NX algorithms that require it...")
            graph_for_nx_cdlib_eff = nx.convert_node_labels_to_integers(
                graph_for_nx_cdlib, first_label=0, ordering="default", label_attribute="original_label"
            )
            cdlib_node_map_reverse = {
                i: graph_for_nx_cdlib_eff.nodes[i]["original_label"]
                for i in graph_for_nx_cdlib_eff.nodes()
            }

    nx_adjacency: list[list[int]] = []
    for node in graph_for_nx_cdlib_eff.nodes():
        if needs_relabeling_for_cdlib and cdlib_node_map_reverse is not None:
            neighbors = [
                node_map[cdlib_node_map_reverse[nbr]] for nbr in graph_for_nx_cdlib_eff[node].keys()
            ]
        else:
            neighbors = [node_map[nbr] for nbr in graph_for_nx_cdlib_eff[node].keys()]
        nx_adjacency.append(neighbors)

    return graph_for_nx_cdlib_eff, needs_relabeling_for_cdlib, cdlib_node_map_reverse, nx_adjacency


def _normalize_ground_truth_labels(
    dataset_gt: dict[Any, int] | list[int] | np.ndarray | None,
    nx_graph_original_ids: nx.Graph,
) -> dict[Any, int] | list[int] | np.ndarray | None:
    """Aligns list/array-style ground-truth labels to original node ids."""
    if isinstance(dataset_gt, list | np.ndarray):
        nodes_order = list(nx_graph_original_ids.nodes())
        if len(dataset_gt) == len(nodes_order):
            return {node_id: label for node_id, label in zip(nodes_order, dataset_gt)}
    return dataset_gt


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
        relevant_prefixes = _exact_match_relevant_prefixes(metric_key)

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
                if relevant_prefixes and algo_prefix not in relevant_prefixes:
                    continue

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

        lines.append(f"| {metric_name} | {yes_count} | {no_count} | {na_count} | {skipped_count} |")

        if mismatches_for_metric:
            mismatch_lines.append(f"### {metric_name}")
            mismatch_lines.extend(f"- {item}" for item in mismatches_for_metric)
            mismatch_lines.append("")

    if mismatch_lines:
        lines.append("")
        lines.append("## Exact-Match Mismatches")
        lines.extend(mismatch_lines)

    return "\n".join(lines).strip()


def _split_markdown_row(line: str) -> list[str]:
    """Split a Markdown table row into stripped cell values."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _strip_html(value: str) -> str:
    """Remove simple HTML tags from Markdown cell values."""
    return re.sub(r"<[^>]+>", "", value).strip()


def _parse_markdown_results_table(
    markdown_text: str,
) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    """Parse the benchmark Markdown results table into keyed rows.

    Returns:
        (headers, rows_by_dataset_algo)
    """
    lines = markdown_text.splitlines()
    header_idx = next((idx for idx, line in enumerate(lines) if line.startswith("| Dataset |")), -1)
    if header_idx < 0 or header_idx + 1 >= len(lines):
        return [], {}

    headers = _split_markdown_row(lines[header_idx])
    rows: dict[tuple[str, str], dict[str, str]] = {}

    for line in lines[header_idx + 2 :]:
        if not line.startswith("|"):
            break
        cells = _split_markdown_row(line)
        if len(cells) != len(headers):
            continue
        row = {header: cell for header, cell in zip(headers, cells)}
        dataset = row.get("Dataset")
        algorithm = row.get("Algorithm")
        if dataset and algorithm:
            rows[(dataset, algorithm)] = row

    return headers, rows


def _parse_time_cell_to_seconds(cell_value: str) -> float | None:
    """Parse formatted time strings (e.g. '12.3 ms') into seconds."""
    clean = _strip_html(cell_value)
    if clean in {"", "N/A", "SKIPPED"}:
        return None

    match = re.match(r"^([0-9.eE+\-]+)\s*([^\s]+)$", clean)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return value
    if unit == "ms":
        return value * 1e-3
    if unit in {"μs", "µs", "us"}:
        return value * 1e-6
    return None


def _parse_memory_cell_to_mb(cell_value: str) -> float | None:
    """Parse formatted memory strings (e.g. '88.1 KB') into MB."""
    clean = _strip_html(cell_value)
    if clean in {"", "N/A", "SKIPPED"}:
        return None

    match = re.match(r"^([0-9.eE+\-]+)\s*([^\s]+)$", clean)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    if unit == "MB":
        return value
    if unit == "KB":
        return value / 1024.0
    if unit == "GB":
        return value * 1024.0
    if unit == "B":
        return value / (1024.0 * 1024.0)
    return None


def _format_ratio_change(ratio: float | None, better_label: str, worse_label: str) -> str:
    """Format ratio change text relative to previous run."""
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return "N/A"
    if ratio >= 1.0:
        return f"{ratio:.2f}x {better_label}"
    return f"{(1.0 / ratio):.2f}x {worse_label}"


def _find_previous_results_table_path(current_result_folder: str) -> str | None:
    """Find the most recent previous results table for run-to-run comparison."""
    local_previous_path = os.path.join(current_result_folder, "benchmark_results_table_prev.md")
    if os.path.isfile(local_previous_path):
        return local_previous_path

    current_folder = os.path.basename(os.path.normpath(current_result_folder))
    results_root = os.path.dirname(os.path.normpath(current_result_folder))
    if not os.path.isdir(results_root):
        return None

    candidate_dirs: list[str] = []
    for entry in os.listdir(results_root):
        full_dir = os.path.join(results_root, entry)
        if not os.path.isdir(full_dir) or entry == current_folder:
            continue
        table_path = os.path.join(full_dir, "benchmark_results_table.md")
        if not os.path.isfile(table_path):
            continue
        if current_folder.isdigit() and entry.isdigit():
            if entry < current_folder:
                candidate_dirs.append(entry)
        else:
            candidate_dirs.append(entry)

    if not candidate_dirs:
        return None

    previous_dir = sorted(candidate_dirs)[-1]
    return os.path.join(results_root, previous_dir, "benchmark_results_table.md")


def _generate_run_to_run_markdown_summary(
    current_table_markdown: str, current_result_folder: str
) -> str:
    """Generate a compact run-to-run regression summary across all algorithms.

    The summary compares the current table against the latest previous results table.
    'Changed locations' counts datasets where non-performance row fields changed
    (excluding Time/Memory and Exact columns).
    """
    previous_table_path = _find_previous_results_table_path(current_result_folder)
    if previous_table_path is None:
        return ""

    try:
        with open(previous_table_path) as prev_file:
            previous_table_markdown = prev_file.read()
    except Exception as exc:
        logger.warning(f"Unable to load previous results table at {previous_table_path}: {exc}")
        return ""

    curr_headers, curr_rows = _parse_markdown_results_table(current_table_markdown)
    prev_headers, prev_rows = _parse_markdown_results_table(previous_table_markdown)
    if not curr_headers or not curr_rows or not prev_headers or not prev_rows:
        return ""

    common_headers = [header for header in curr_headers if header in set(prev_headers)]
    exact_columns = [header for header in common_headers if header.startswith("Exact ")]
    compare_columns = [
        header
        for header in common_headers
        if header not in {"Time", "Memory"} and not header.startswith("Exact ")
    ]

    algorithm_order = [algo_cfg["name"] for algo_cfg in ALGORITHMS_CONFIG_STRUCTURE]
    available_algorithms = {algorithm for _, algorithm in curr_rows.keys()}
    ordered_algorithms = [name for name in algorithm_order if name in available_algorithms]

    if os.path.basename(previous_table_path) == "benchmark_results_table_prev.md":
        previous_run_id = f"{os.path.basename(os.path.normpath(current_result_folder))} (prev run)"
    else:
        previous_run_id = os.path.basename(os.path.dirname(previous_table_path))
    current_run_id = os.path.basename(os.path.normpath(current_result_folder))

    lines: list[str] = [
        "## Run-to-Run Regression Summary",
        "",
        f"Comparison: `{previous_run_id}` -> `{current_run_id}`.",
        "Changed locations = datasets where non-performance output changed "
        "(Time/Memory and Exact columns excluded).",
        "",
        "| Algorithm | Exact (Y/N/NA) | Speed vs Prev | Memory vs Prev | Current Memory (median) | Changed Locations |",
        "|-|-:|-:|-:|-:|-:|",
    ]

    for algorithm_name in ordered_algorithms:
        curr_by_dataset = {
            dataset: row
            for (dataset, algo_name), row in curr_rows.items()
            if algo_name == algorithm_name
        }
        prev_by_dataset = {
            dataset: row
            for (dataset, algo_name), row in prev_rows.items()
            if algo_name == algorithm_name
        }

        comparable_datasets = sorted(set(curr_by_dataset) & set(prev_by_dataset))

        speed_ratios: list[float] = []
        memory_ratios: list[float] = []
        current_memory_values_mb: list[float] = []
        changed_locations = 0

        for dataset, curr_row in curr_by_dataset.items():
            curr_mem = _parse_memory_cell_to_mb(curr_row.get("Memory", ""))
            if curr_mem is not None:
                current_memory_values_mb.append(curr_mem)

            prev_row = prev_by_dataset.get(dataset)
            if prev_row is None:
                continue

            curr_time = _parse_time_cell_to_seconds(curr_row.get("Time", ""))
            prev_time = _parse_time_cell_to_seconds(prev_row.get("Time", ""))
            if curr_time is not None and prev_time is not None and curr_time > 0:
                speed_ratios.append(prev_time / curr_time)

            prev_mem = _parse_memory_cell_to_mb(prev_row.get("Memory", ""))
            if curr_mem is not None and prev_mem is not None and curr_mem > 0:
                memory_ratios.append(prev_mem / curr_mem)

            if any(
                _strip_html(curr_row.get(col, "")) != _strip_html(prev_row.get(col, ""))
                for col in compare_columns
            ):
                changed_locations += 1

        relevant_exact_cols = [
            col
            for col in exact_columns
            if any(_strip_html(row.get(col, "")) in {"Yes", "No"} for row in curr_by_dataset.values())
        ]

        if relevant_exact_cols:
            exact_yes = 0
            exact_no = 0
            exact_na = 0
            for row in curr_by_dataset.values():
                for col in relevant_exact_cols:
                    val = _strip_html(row.get(col, ""))
                    if val == "Yes":
                        exact_yes += 1
                    elif val == "No":
                        exact_no += 1
                    else:
                        exact_na += 1
            exact_text = f"{exact_yes}/{exact_no}/{exact_na}"
        else:
            exact_text = "N/A"

        speed_text = _format_ratio_change(
            median(speed_ratios) if speed_ratios else None, "faster", "slower"
        )
        memory_text = _format_ratio_change(
            median(memory_ratios) if memory_ratios else None, "less", "more"
        )
        current_memory_text = (
            format_memory(median(current_memory_values_mb)) if current_memory_values_mb else "N/A"
        )
        changed_text = (
            f"{changed_locations}/{len(comparable_datasets)}" if comparable_datasets else "N/A"
        )

        lines.append(
            f"| {algorithm_name} | {exact_text} | {speed_text} | {memory_text} | {current_memory_text} | {changed_text} |"
        )

    return "\n".join(lines).strip()
