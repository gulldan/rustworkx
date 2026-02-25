# ruff: noqa: E501
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from benchmark_config_data import ALGORITHMS_CONFIG_STRUCTURE
from benchmark_utils import format_bool, format_memory, format_time
from metrics_config import JACCARD_THRESHOLDS_TO_TEST, ORDERED_METRIC_PROPERTIES

logger = logging.getLogger(__name__)

BenchmarkResultItem = dict[str, Any]
BenchmarkResultsList = list[BenchmarkResultItem]
MetricProperty = dict[str, Any]

TABLE_COLUMN_KEYS: list[str] = [m["key"] for m in ORDERED_METRIC_PROPERTIES]
TABLE_COLUMN_NAMES: list[str] = [m["name"] for m in ORDERED_METRIC_PROPERTIES]
METRIC_KEY_TO_PROPERTIES: dict[str, MetricProperty] = {m["key"]: m for m in ORDERED_METRIC_PROPERTIES}

EXACT_MATCH_MPL_HEADER_LABELS: dict[str, str] = {
    "_exact_match_nx_louvain": "Exact\nRX vs NX\nLouvain",
    "_exact_match_nx_lpa": "Exact\nRX vs NX\nLPA",
    "_exact_match_cdlib_leiden": "Exact\nRX vs\ncdlib Leiden",
    "_exact_match_leidenalg": "Exact\nRX vs\nleidenalg",
    "_exact_match_nx_cliques": "Exact\nRX vs NX\nCliques",
    "_exact_match_nx_cpm": "Exact\nRX vs NX\nCPM",
}
TABLE_COLUMN_NAMES_MPL: list[str] = [
    EXACT_MATCH_MPL_HEADER_LABELS.get(col_key, col_name)
    for col_key, col_name in zip(TABLE_COLUMN_KEYS, TABLE_COLUMN_NAMES)
]

ALGORITHMS_METADATA: list[dict[str, Any]] = ALGORITHMS_CONFIG_STRUCTURE

FORMATTER_FUNCTIONS: dict[str, Callable[[Any], str]] = {
    "format_bool": format_bool,
    "format_time": format_time,
    "format_memory": format_memory,
}

__all__ = ["JACCARD_THRESHOLDS_TO_TEST"]


def _get_algo_run_elapsed_and_presence(
    dataset_result: BenchmarkResultItem, algo_prefix: str
) -> tuple[bool, Any]:
    """Returns whether an algorithm row exists and its raw elapsed value."""
    has_algo_entry: bool = any(key.startswith(algo_prefix) for key in dataset_result.keys())
    if not has_algo_entry:
        return False, None

    elapsed_key_for_algo: str = f"{algo_prefix}_elapsed"
    return True, dataset_result.get(
        elapsed_key_for_algo,
        dataset_result.get(f"{algo_prefix}{METRIC_KEY_TO_PROPERTIES['_elapsed']['key']}"),
    )


def _coerce_float_metric_value(raw_value: Any) -> float | None:
    """Converts metric values to float when possible."""
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _format_metric_cell_value(
    raw_metric_val: Any,
    metric_property: MetricProperty,
    *,
    raw_elapsed_val: Any,
    is_algo_run_skipped: bool,
    num_gt_clusters: int = 0,
) -> str:
    """Formats a metric cell consistently for both markdown and matplotlib outputs."""
    metric_key: str = metric_property["key"]
    formatted_cell_value: str = "N/A"

    if metric_key.startswith("_gcr_jt") or metric_key.startswith("_pcp_jt"):
        if num_gt_clusters <= 1:
            return "N/A"

    if is_algo_run_skipped and metric_key not in ["_elapsed", "_memory"]:
        return "SKIPPED"

    if raw_metric_val is None:
        return formatted_cell_value

    if isinstance(raw_metric_val, str) and raw_metric_val == "SKIPPED":
        return "SKIPPED"

    formatter_name: str | None = metric_property.get("format_func")
    formatter_func: Callable[[Any], str] | None = (
        FORMATTER_FUNCTIONS.get(formatter_name) if isinstance(formatter_name, str) else None
    )
    if formatter_func:
        return formatter_func(raw_metric_val)

    float_metric_val: float | None = _coerce_float_metric_value(raw_metric_val)
    if float_metric_val is not None and np.isnan(float_metric_val):
        return "N/A"

    if metric_key == "_elapsed" and raw_elapsed_val == -1:
        return "SKIPPED"

    if float_metric_val is not None:
        if abs(float_metric_val) > 1e4 or (abs(float_metric_val) < 0.001 and float_metric_val != 0):
            return f"{float_metric_val:.2e}"
        return f"{float_metric_val:.3f}"

    return str(raw_metric_val)


def style_exact_match_markdown_cell(metric_key: str, value_text: str) -> str:
    """Add color emphasis for exact-match Yes/No values in Markdown tables."""
    if not metric_key.startswith("_exact_match_"):
        return value_text
    if value_text == "Yes":
        return (
            '<span style="background-color:#BAFFBA;color:#000;'
            'padding:0 4px;border-radius:2px;"><strong>Yes</strong></span>'
        )
    if value_text == "No":
        return (
            '<span style="background-color:#FFBABA;color:#000;'
            'padding:0 4px;border-radius:2px;"><strong>No</strong></span>'
        )
    return value_text


def get_valid_benchmark_results(
    benchmark_results_input: BenchmarkResultsList | None,
) -> BenchmarkResultsList:
    """Filters out placeholder or completely failed dataset results from the raw benchmark_results list.

    A result is considered critically failed if it has an "error" field
    and no algorithm within it was successfully attempted or completed
    (indicated by an elapsed time >= 0). Placeholders with
    status "skipped_placeholder" are also removed.

    Args:
        benchmark_results_input: The raw list of benchmark result items.
            Each item is a dictionary containing metrics for a dataset.

    Returns:
        A new list containing only the valid benchmark result items.
        Returns an empty list if input is None or empty.
    """
    if not benchmark_results_input:
        return []

    valid_results_output: BenchmarkResultsList = []
    for res_item in benchmark_results_input:
        # Skip if it's a placeholder status
        if res_item.get("status") == "skipped_placeholder":
            continue

        # Determine if this dataset entry is critically errored (no algorithm attempted or completed)
        # An algorithm is considered "attempted" if its_elapsed key exists and is >= 0 (0 or positive time)
        # -1 means intentionally skipped, -2 (or other defaults) might mean not even initialized.
        has_any_successful_algo: bool = False
        for algo_meta_item in ALGORITHMS_METADATA:
            # Construct the elapsed time key as stored by benchmark_community.py: algo_prefix + metric_key_base
            # Example: "nx_louvain_res0p5" + "_elapsed" -> "nx_louvain_res0p5_elapsed"
            elapsed_key_for_algo: str = f"{algo_meta_item['prefix']}_elapsed"
            if res_item.get(elapsed_key_for_algo, -2) >= 0:  # -2 is a default if key not found
                has_any_successful_algo = True
                break

        # If there's an error string AND no algorithm was successfully run or attempted, then skip this dataset entry.
        if "error" in res_item and not has_any_successful_algo:
            continue

        valid_results_output.append(res_item)
    return valid_results_output
