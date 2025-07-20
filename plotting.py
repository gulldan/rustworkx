import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Any
from collections.abc import Callable, Sequence  # Added for type hinting

# from benchmark_utils import format_memory, format_time # Now imported by metrics_config
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from collections import defaultdict
import logging  # Add logging import

# Import constants from metrics_config.py
from metrics_config import ORDERED_METRIC_PROPERTIES, JACCARD_THRESHOLDS_TO_TEST
from benchmark_utils import (
    format_memory,
    format_time,
)  # It seems plotting.py still uses these directly for chart axis formatting

# Import ALGORITHMS_CONFIG_STRUCTURE from benchmark_config_data.py
# This is the new single source of truth for algorithm metadata (name, prefix, color, run_args, etc.).
# Also import RESOLUTIONS_TO_TEST and GRID_COLOR as they are used for plotting.
from benchmark_config_data import ALGORITHMS_CONFIG_STRUCTURE, GRID_COLOR

# Setup logger for this module
logger = logging.getLogger(__name__)
# BasicConfig should be handled by the main script (benchmark_community.py)

# Define types for benchmark data structures for clarity
BenchmarkResultItem = dict[str, Any]  # A dictionary representing results for one dataset
BenchmarkResultsList = list[BenchmarkResultItem]  # A list of such dictionaries
MetricProperty = dict[str, Any]  # A dictionary describing a metric (from ORDERED_METRIC_PROPERTIES)


# Extract keys and names for convenience, maintaining order for table columns
TABLE_COLUMN_KEYS: list[str] = [m["key"] for m in ORDERED_METRIC_PROPERTIES]
TABLE_COLUMN_NAMES: list[str] = [m["name"] for m in ORDERED_METRIC_PROPERTIES]
# Create a dictionary for quick lookup of metric properties by key
METRIC_KEY_TO_PROPERTIES: dict[str, MetricProperty] = {m["key"]: m for m in ORDERED_METRIC_PROPERTIES}

# Use ALGORITHMS_CONFIG_STRUCTURE directly, aliasing if needed for minimal changes within functions.
ALGORITHMS_METADATA: list[dict[str, Any]] = (
    ALGORITHMS_CONFIG_STRUCTURE  # Alias for less refactoring within this file
)

# Make formatter functions easily accessible by string name
FORMATTER_FUNCTIONS: dict[str, Callable[[Any], str]] = {
    "format_time": format_time,
    "format_memory": format_memory,
}


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


def create_comparison_chart(
    benchmark_results: BenchmarkResultsList, output_file: str = "community_detection_comparison.png"
) -> str | None:
    """Creates a comparative bar chart of performance metrics.

    The chart displays selected metrics for different algorithms across datasets.
    Handles NaN values, algorithm skips, and log scales for time/memory.
    Metrics are selected from `ORDERED_METRIC_PROPERTIES`.

    Args:
        benchmark_results: A list of benchmark result items.
        output_file: The path to save the generated chart image.

    Returns:
        The `output_file` path if the chart is saved successfully,
        otherwise None.
    """
    benchmark_results_valid: BenchmarkResultsList = get_valid_benchmark_results(benchmark_results)
    if not benchmark_results_valid:
        logger.warning("No valid benchmark results to create comparison chart.")
        return None

    datasets: list[str] = [result["dataset"] for result in benchmark_results_valid]
    n_datasets: int = len(datasets)

    # Select metrics for plotting from ORDERED_METRIC_PROPERTIES
    # Prioritize external, internal, and performance. Exclude some for brevity or if they have dedicated plots.
    chart_metrics_selection: list[MetricProperty] = [
        prop
        for prop in ORDERED_METRIC_PROPERTIES
        if prop["type"] in ["external", "internal", "perf"]
        and prop["key"]
        not in [
            "_surprise",
            "_significance",
            "_cut_ratio",
            "_tpr",
            "_avg_internal_degree",  # Often many, can clutter
            "_homogeneity",
            "_completeness",  # Usually covered by V-Measure
            "_gcr",
            "_pcp",  # Base keys for GCR/PCP, specific JTs might be too many
        ]
        and not prop["key"].startswith("_gcr_jt")
        and not prop["key"].startswith("_pred_prec_jt")
    ]
    # Ensure at least elapsed and memory are there if filtered out by above
    if not any(m["key"] == "_elapsed" for m in chart_metrics_selection):
        chart_metrics_selection.append(METRIC_KEY_TO_PROPERTIES["_elapsed"])
    if not any(m["key"] == "_memory" for m in chart_metrics_selection):
        chart_metrics_selection.append(METRIC_KEY_TO_PROPERTIES["_memory"])

    num_metrics_to_plot: int = len(chart_metrics_selection)
    if num_metrics_to_plot == 0:
        logger.warning("No metrics selected for chart plotting.")
        return None

    plt.switch_backend("Agg")
    fig_height_per_metric: float = 7.0
    fig: plt.Figure
    ax_list: np.ndarray[Any, np.dtype[plt.Axes]] | list[plt.Axes]  # Type for axes array or list

    if num_metrics_to_plot == 1:
        fig, ax_one = plt.subplots(1, 1, figsize=(26, fig_height_per_metric), dpi=200)
        ax_list = [ax_one]
    else:
        fig, axes_array = plt.subplots(
            num_metrics_to_plot, 1, figsize=(26, num_metrics_to_plot * fig_height_per_metric), dpi=200
        )
        ax_list = axes_array.flatten() if isinstance(axes_array, np.ndarray) else [axes_array]

    plt.subplots_adjust(hspace=0.9 if num_metrics_to_plot > 1 else 0.1)

    n_algorithms_plot: int = len(ALGORITHMS_METADATA)
    total_bar_space: float = 0.9
    bar_width: float = total_bar_space / n_algorithms_plot
    r_base: np.ndarray[Any, np.dtype[np.float64]] = np.arange(
        n_datasets, dtype=float
    )  # Ensure float for offsets
    offsets: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(
        -total_bar_space / 2 + bar_width / 2, total_bar_space / 2 - bar_width / 2, n_algorithms_plot
    )
    algo_r_positions: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {
        meta["prefix"]: r_base + offset for meta, offset in zip(ALGORITHMS_METADATA, offsets)
    }

    def add_value_labels_to_chart_bars(
        ax_handle: plt.Axes,
        rects_container_list: Sequence[
            plt.Rectangle
        ],  # Changed from list to Sequence for broader compatibility
        data_values_list: list[float | None],
        metric_property_dict: MetricProperty,
        is_log_scale_local: bool,
    ) -> None:
        """Adds text labels above or below bars in a chart.

        Args:
            ax_handle: The Matplotlib Axes object for the subplot.
            rects_container_list: A list of Rectangle objects (the bars).
            data_values_list: The numerical data values corresponding to the bars.
            metric_property_dict: Properties of the metric being plotted.
            is_log_scale_local: True if the y-axis is on a log scale.
        """
        for bar_idx, rect_bar in enumerate(rects_container_list):
            current_data_val = data_values_list[bar_idx]
            if (
                bar_idx >= len(data_values_list)
                or current_data_val is None
                or np.isnan(current_data_val)
            ):
                continue

            y_coordinate: float = rect_bar.get_height()
            x_coordinate: float = rect_bar.get_x() + rect_bar.get_width() / 2.0

            label_y_pos: float = y_coordinate * 1.1 if y_coordinate >= 0 else y_coordinate * 0.9
            if is_log_scale_local and y_coordinate > 1e-9:  # Adjusted positive check for log
                label_y_pos = y_coordinate * 1.5
            elif is_log_scale_local:  # Value is likely 0 or very small, position label near bottom
                label_y_pos = ax_handle.get_ylim()[0] * 2  # Position relative to bottom of y-axis

            text_label: str = "N/A"
            formatter_func_name: str | None = metric_property_dict.get("format_func")
            actual_formatter_func: Callable[[Any], str] | None = (
                FORMATTER_FUNCTIONS.get(formatter_func_name)
                if isinstance(formatter_func_name, str)
                else None
            )

            if metric_property_dict["key"] == "_elapsed" and current_data_val == -1:
                text_label = "SKIPPED"
            elif actual_formatter_func:
                text_label = actual_formatter_func(current_data_val)
            else:
                try:
                    text_label = f"{current_data_val:.2f}"
                except (TypeError, ValueError):
                    text_label = str(current_data_val)

            ax_handle.text(
                x_coordinate,
                label_y_pos,
                text_label,
                ha="center",
                va="bottom" if y_coordinate >= 0 else "top",
                fontsize=4,
                color="#333333",
                rotation=90,
            )

    def style_chart_axis(ax_handle: plt.Axes, metric_property_dict_style: MetricProperty) -> None:
        """Styles the axes, title, and legend of a chart subplot.

        Args:
            ax_handle: The Matplotlib Axes object.
            metric_property_dict_style: Properties of the metric for styling.
        """
        ax_handle.set_ylabel(metric_property_dict_style["name"], fontweight="bold", fontsize=10)

        title_text: str = metric_property_dict_style["name"]
        if metric_property_dict_style.get("higher_is_better") is not None:
            title_text += (
                f" ({'higher' if metric_property_dict_style['higher_is_better'] else 'lower'} is better)"
            )
        # Check if y-axis is log scaled to append to title
        if ax_handle.get_yaxis().get_scale() == "log":
            title_text += " (log scale)"
        ax_handle.set_title(title_text, fontweight="bold", fontsize=12, pad=20)  # Increased pad

        ax_handle.set_xticks(r_base)
        ax_handle.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
        ax_handle.yaxis.grid(True, linestyle="--", linewidth=0.5, color=GRID_COLOR, alpha=0.7)
        ax_handle.set_axisbelow(True)

        legend_handles: list[plt.Rectangle] = [
            plt.Rectangle((0, 0), 1, 1, color=algo_meta_legend["color"])
            for algo_meta_legend in ALGORITHMS_METADATA
        ]
        legend_labels: list[str] = [algo_meta_legend["name"] for algo_meta_legend in ALGORITHMS_METADATA]
        # Adjust legend positioning based on number of algorithms
        legend_ncol: int = min(n_algorithms_plot, 6)  # Max 6 columns for legend
        legend_bbox_y_offset: float = -0.35 - (0.06 * (n_algorithms_plot // legend_ncol))
        ax_handle.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_bbox_y_offset),
            ncol=legend_ncol,
            fontsize=7,
            frameon=True,
            framealpha=0.9,
        )

    for idx, current_metric_prop in enumerate(chart_metrics_selection):
        ax_curr: plt.Axes = ax_list[idx]
        metric_key_current: str = current_metric_prop["key"]
        data_by_algo_curr: dict[str, list[float | None]] = {}  # Store numerical data, None for NaN
        all_metric_vals_for_ylim: list[float] = []
        is_log_scale_metric: bool = metric_key_current in ["_elapsed", "_memory"]

        if is_log_scale_metric:
            ax_curr.set_yscale("log")

        for algo_meta_iter in ALGORITHMS_METADATA:
            algo_prefix_iter: str = algo_meta_iter["prefix"]
            full_metric_key_for_data: str = f"{algo_prefix_iter}{metric_key_current}"

            data_series_for_algo: list[float | None] = []
            for res_data_item in benchmark_results_valid:
                # Special handling for intentionally skipped runs (elapsed == -1)
                if res_data_item.get(f"{algo_prefix_iter}_elapsed") == -1:
                    data_series_for_algo.append(
                        -1.0 if metric_key_current == "_elapsed" else None
                    )  # None for NaN
                else:
                    value: Any = res_data_item.get(full_metric_key_for_data)
                    try:
                        data_series_for_algo.append(float(value) if value is not None else None)
                    except (ValueError, TypeError):
                        data_series_for_algo.append(None)  # Treat conversion errors as NaN

            data_by_algo_curr[algo_prefix_iter] = data_series_for_algo
            # Collect valid values for y-limit calculation, excluding NaNs and our special -1 for skipped elapsed
            all_metric_vals_for_ylim.extend(
                [
                    v
                    for v in data_series_for_algo
                    if v is not None
                    and not np.isnan(v)
                    and not (metric_key_current == "_elapsed" and v == -1)
                ]
            )

        # --- Plotting bars for each algorithm ---
        bar_rect_containers: dict[str, plt.barcontainer.BarContainer] = {}
        for algo_meta_plot in ALGORITHMS_METADATA:
            algo_prefix_plot: str = algo_meta_plot["prefix"]
            series_to_plot: list[float | None] = data_by_algo_curr[algo_prefix_plot]
            bar_positions: np.ndarray[Any, np.dtype[np.float64]] = algo_r_positions[algo_prefix_plot]

            # For log scale, map values: >0 remains, <=0 or NaN becomes small positive for plotting or NaN
            # The -1 for elapsed is a special "SKIPPED" indicator
            if is_log_scale_metric:
                series_adjusted_for_log: list[float] = [
                    max(v, 1e-9)
                    if (v is not None and not np.isnan(v) and v > 0)
                    else (
                        1e-10 if metric_key_current == "_elapsed" and v == -1 else 1e-9
                    )  # Distinguish skipped slightly if needed, or map to baseline
                    for v in series_to_plot
                ]
                rects_drawn: plt.barcontainer.BarContainer = ax_curr.bar(
                    bar_positions,
                    series_adjusted_for_log,
                    width=bar_width,
                    label=algo_meta_plot["name"],
                    color=algo_meta_plot["color"],
                )
            else:
                # For linear scale, plot NaNs as gaps
                series_linear: list[float] = [v if v is not None else np.nan for v in series_to_plot]
                rects_drawn: plt.barcontainer.BarContainer = ax_curr.bar(
                    bar_positions,
                    series_linear,
                    width=bar_width,
                    label=algo_meta_plot["name"],
                    color=algo_meta_plot["color"],
                )
            bar_rect_containers[algo_prefix_plot] = rects_drawn

        # --- Set Y-axis limits ---
        if all_metric_vals_for_ylim:
            y_min_data: float = min(all_metric_vals_for_ylim)
            y_max_data: float = max(all_metric_vals_for_ylim)

            if is_log_scale_metric:
                # Ensure y_min_final_log is positive for log scale
                y_min_final_log: float = max(1e-9, y_min_data / 5 if y_min_data > 1e-8 else 1e-9)
                y_max_final_log: float = y_max_data * 5 if y_max_data > 1e-8 else 1
                # Handle cases where min/max are too close or problematic for log
                if y_min_final_log >= y_max_final_log:
                    y_max_final_log = y_min_final_log * 100  # Ensure range
                ax_curr.set_ylim(y_min_final_log, y_max_final_log)
                # Apply custom formatters for log axes
                if metric_key_current == "_elapsed":
                    ax_curr.yaxis.set_major_formatter(
                        FuncFormatter(lambda x, _: format_time(x if x > 1e-8 else 0))
                    )
                elif metric_key_current == "_memory":
                    ax_curr.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_memory(x)))
            else:  # Linear scale
                data_range: float = y_max_data - y_min_data if (y_max_data - y_min_data) > 1e-6 else 0.1
                padding_linear: float = data_range * 0.1
                y_min_final_linear: float = (
                    min(0, y_min_data - padding_linear)
                    if (current_metric_prop.get("higher_is_better", True) and y_min_data >= 0)
                    else (y_min_data - padding_linear)
                )
                y_max_final_linear: float = y_max_data + padding_linear
                # For metrics like ARI, FMI, etc., ensure y-max is at least 1.0 if data is within [0,1]
                if current_metric_prop.get("higher_is_better") and y_max_data <= 1.0 and y_min_data >= 0:
                    y_max_final_linear = max(y_max_final_linear, 1.0)
                if y_min_final_linear >= y_max_final_linear:
                    y_max_final_linear = y_min_final_linear + 0.1  # Ensure range
                ax_curr.set_ylim(y_min_final_linear, y_max_final_linear)
        else:  # No valid data points, set default reasonable limits
            if is_log_scale_metric:
                ax_curr.set_ylim(1e-9, 1)
            else:
                ax_curr.set_ylim(0, 1)

        # --- Add value labels and style axis ---
        for algo_meta_label in ALGORITHMS_METADATA:
            if algo_meta_label["prefix"] in bar_rect_containers:
                add_value_labels_to_chart_bars(
                    ax_curr,
                    bar_rect_containers[algo_meta_label["prefix"]].patches,
                    data_by_algo_curr[algo_meta_label["prefix"]],
                    current_metric_prop,
                    is_log_scale_metric,
                )
        style_chart_axis(ax_curr, current_metric_prop)  # Style after all data plotting and limit setting

    fig.suptitle(
        "Community Detection Benchmark: Algorithm Comparison", fontsize=22, fontweight="bold", y=1.00
    )
    try:
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0.4)  # Increased pad
        logger.info(f"Comparison chart saved successfully to {output_file}")
    except Exception as e_save:
        logger.error(f"Error saving comparison chart: {e_save}", exc_info=True)
    plt.close(fig)
    return output_file


def generate_results_table(benchmark_results: BenchmarkResultsList) -> str:
    """Generates a Markdown table from benchmark results.

    The table includes results for each algorithm run, accounting for
    parameterized runs. Metrics are ordered and formatted based on
    `ORDERED_METRIC_PROPERTIES`.

    Args:
        benchmark_results: A list of benchmark result items.

    Returns:
        A string containing the Markdown formatted table.
        Returns an error message string if no valid results are available.
    """
    benchmark_results_valid: BenchmarkResultsList = get_valid_benchmark_results(benchmark_results)
    if not benchmark_results_valid:
        logger.warning("No valid benchmark results to display for Markdown table.")
        return "No valid benchmark results to display for Markdown table."

    # Use TABLE_COLUMN_NAMES (derived from ORDERED_METRIC_PROPERTIES) for the header
    header_row_str: str = "| " + " | ".join(TABLE_COLUMN_NAMES) + " |"
    separator_row_str: str = "|-" + "-|-".join(["-" * len(name) for name in TABLE_COLUMN_NAMES]) + "-|"
    markdown_table_rows: list[str] = [header_row_str, separator_row_str]

    for dataset_res_item in benchmark_results_valid:
        # Common dataset information for all algorithm rows of this dataset
        dataset_info_values_map: dict[str, str] = {}
        for metric_prop_info_ds in ORDERED_METRIC_PROPERTIES:
            if metric_prop_info_ds["type"] == "info" and metric_prop_info_ds["key"] != "algorithm_name":
                dataset_info_values_map[metric_prop_info_ds["key"]] = str(
                    dataset_res_item.get(metric_prop_info_ds["key"], "N/A")
                )

        for algo_metadata_item_md in ALGORITHMS_METADATA:
            algo_key_prefix_md: str = algo_metadata_item_md["prefix"]
            algo_name_display_md: str = algo_metadata_item_md["name"]

            # Check if this specific algorithm run has data (e.g., elapsed time recorded)
            elapsed_metric_full_key: str = (
                f"{algo_key_prefix_md}_elapsed"  # Key used in benchmark_community.py
            )

            # If the primary elapsed key is not found, this algo run was likely not even initialized for this dataset
            if (
                elapsed_metric_full_key not in dataset_res_item
                and f"{algo_key_prefix_md}{METRIC_KEY_TO_PROPERTIES['_elapsed']['key']}"
                not in dataset_res_item
            ):
                # Check if ANY key for this algo_prefix exists, otherwise it was truly not run/recorded
                if not any(k.startswith(algo_key_prefix_md) for k in dataset_res_item.keys()):
                    continue  # Skip this algorithm row for this dataset

            raw_elapsed_val_md: Any = dataset_res_item.get(
                elapsed_metric_full_key,
                dataset_res_item.get(
                    f"{algo_key_prefix_md}{METRIC_KEY_TO_PROPERTIES['_elapsed']['key']}"
                ),
            )
            is_skipped_this_algo_run: bool = raw_elapsed_val_md == -1

            current_md_row_cells: list[str] = []
            for metric_prop_md_col in ORDERED_METRIC_PROPERTIES:
                col_key_md: str = metric_prop_md_col["key"]

                if metric_prop_md_col["type"] == "info":
                    if col_key_md == "algorithm_name":
                        current_md_row_cells.append(algo_name_display_md)
                    else:
                        current_md_row_cells.append(dataset_info_values_map.get(col_key_md, "N/A"))
                else:  # Metric column
                    # Construct the full key as stored in benchmark_results
                    # e.g. algo_prefix ('nx_louvain_res0p5') + metric_key_suffix ('_ari') -> 'nx_louvain_res0p5_ari'
                    metric_data_key: str = f"{algo_key_prefix_md}{col_key_md}"
                    raw_metric_val_md: Any = dataset_res_item.get(metric_data_key)

                    formatted_cell_value: str = "N/A"  # Initialize before potential debug print
                    # --- DEBUG PRINT ---
                    if "_pcp_jt" in col_key_md or "_gcr_jt" in col_key_md:
                        # Temporarily format for debug print, actual formatting happens later
                        temp_formatted_debug: str = "N/A"
                        if raw_metric_val_md is not None and not (
                            isinstance(raw_metric_val_md, float) and np.isnan(raw_metric_val_md)
                        ):
                            temp_formatted_debug = f"{float(raw_metric_val_md):.3f}"
                        logger.debug(
                            f"TABLE DEBUG: Algo={algo_name_display_md}, Key='{metric_data_key}', RawValue={raw_metric_val_md}, TempFormattedDebug={temp_formatted_debug}"
                        )
                    # --- END DEBUG PRINT ---

                    if is_skipped_this_algo_run and col_key_md not in ["_elapsed", "_memory"]:
                        formatted_cell_value = "SKIPPED"
                    elif raw_metric_val_md is not None:
                        if isinstance(raw_metric_val_md, str) and raw_metric_val_md == "SKIPPED":
                            formatted_cell_value = "SKIPPED"
                        elif isinstance(raw_metric_val_md, float | int) and np.isnan(
                            float(raw_metric_val_md)
                        ):
                            formatted_cell_value = "N/A"
                        elif (
                            col_key_md == "_elapsed" and raw_elapsed_val_md == -1
                        ):  # Check original elapsed for skip
                            formatted_cell_value = "SKIPPED"
                        else:
                            formatter_name_md: str | None = metric_prop_md_col.get("format_func")
                            actual_formatter_md: Callable[[Any], str] | None = (
                                FORMATTER_FUNCTIONS.get(formatter_name_md)
                                if isinstance(formatter_name_md, str)
                                else None
                            )
                            if actual_formatter_md:
                                formatted_cell_value = actual_formatter_md(raw_metric_val_md)
                            elif isinstance(raw_metric_val_md, float | int):
                                val_fl_md: float = float(raw_metric_val_md)
                                # Scientific notation for very small or very large numbers, else fixed point
                                if abs(val_fl_md) > 1e4 or (abs(val_fl_md) < 0.001 and val_fl_md != 0):
                                    formatted_cell_value = f"{val_fl_md:.2e}"
                                else:
                                    formatted_cell_value = f"{val_fl_md:.3f}"
                            else:
                                formatted_cell_value = str(raw_metric_val_md)
                    current_md_row_cells.append(formatted_cell_value)

            markdown_table_rows.append("| " + " | ".join(current_md_row_cells) + " |")

    return "\n".join(markdown_table_rows)


def get_color_for_cell_mpl(
    value: Any,
    metric_prop: MetricProperty,
    norm_min: float | None,
    norm_max: float | None,
    is_overall_skipped: bool = False,
    is_na_value: bool = False,
) -> str:
    """Determines a hex color string for a Matplotlib table cell.

    Coloring is based on the metric's normalized value, indicating performance
    (e.g., green for good, red for bad). Handles skipped or N/A values.

    Args:
        value: The raw metric value for the cell.
        metric_prop: Properties of the metric being displayed.
        norm_min: The minimum value for this metric in the current context
            (used for normalization).
        norm_max: The maximum value for this metric.
        is_overall_skipped: True if the entire algorithm run for this row was skipped.
        is_na_value: True if the specific cell value is N/A.

    Returns:
        A hex color string (e.g., "#FF0000").
    """
    if is_overall_skipped:
        return "#DCDCDC"  # Lighter Gray for SKIPPED main content
    if (
        is_na_value
        or value is None
        or (isinstance(value, float) and np.isnan(value))
        or norm_min is None
        or norm_max is None
    ):  # norm_min == norm_max removed to allow coloring even if all values are same (will be mid-color)
        return "#FFFFFF"

    higher_is_better: bool | None = metric_prop.get("higher_is_better")
    if higher_is_better is None:
        return "#F5F5F5"

    val_f: float
    try:
        val_f = float(value)
    except (ValueError, TypeError):
        return "#FFFFFF"

    norm_val: float
    if norm_max == norm_min:  # All values for this metric in this context are the same
        norm_val = 0.5  # Mid-point color
    else:
        norm_val = (val_f - norm_min) / (norm_max - norm_min)

    if not higher_is_better:
        norm_val = 1.0 - norm_val

    norm_val = max(0.0, min(1.0, norm_val))

    cmap: mcolors.LinearSegmentedColormap = mcolors.LinearSegmentedColormap.from_list(
        "custom_rg", [(0, "#FFBABA"), (0.5, "#FFFFC8"), (1, "#BAFFBA")]
    )
    return mcolors.to_hex(cmap(norm_val))  # Return hex string


def generate_results_table_matplotlib(
    benchmark_results: BenchmarkResultsList, output_file: str = "results/benchmark_results_table.png"
) -> None:
    """Generates a results table as a PNG image using Matplotlib.

    The table displays detailed benchmark results with cells colored by
    performance. Handles dynamic data presence and formatting.

    Args:
        benchmark_results: A list of benchmark result items.
        output_file: The path to save the generated table image.
    """
    benchmark_results_valid: BenchmarkResultsList = get_valid_benchmark_results(benchmark_results)
    if not benchmark_results_valid:
        logger.warning("No valid benchmark results to create Matplotlib table.")
        return

    table_cell_text_data: list[list[str]] = []
    # Store raw values for coloring: metric_key_suffix -> dataset_name -> list of raw float values
    raw_cell_values_for_color: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Track which algos ran for which datasets: dataset_name -> list of algo_meta dicts
    processed_algo_runs_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for dataset_res_item_mpl in benchmark_results_valid:
        dataset_name_mpl: str = dataset_res_item_mpl["dataset"]
        for algo_meta_mpl in ALGORITHMS_METADATA:
            algo_prefix_key_mpl: str = algo_meta_mpl["prefix"]
            elapsed_key_check: str = f"{algo_prefix_key_mpl}_elapsed"
            # Check if any key for this algo exists (not just _elapsed which might be missing if only info keys are present)
            if any(k.startswith(algo_prefix_key_mpl) for k in dataset_res_item_mpl.keys()):
                # Check if it was at least initialized (elapsed time might be 0 or positive, or -1 for skipped)
                if (
                    dataset_res_item_mpl.get(elapsed_key_check, -2) != -2
                ):  # -2 is our default for "not even in results"
                    processed_algo_runs_map[dataset_name_mpl].append(algo_meta_mpl)

    if not any(processed_algo_runs_map.values()):
        logger.warning("No algorithm data found across any datasets for Matplotlib table.")
        return

    for dataset_res_item_mpl in benchmark_results_valid:
        dataset_name_mpl: str = dataset_res_item_mpl["dataset"]
        current_dataset_algo_metas: list[dict[str, Any]] = processed_algo_runs_map[dataset_name_mpl]
        if not current_dataset_algo_metas:
            continue

        for algo_meta_mpl in current_dataset_algo_metas:
            algo_prefix_key_mpl: str = algo_meta_mpl["prefix"]
            algo_name_display_mpl: str = algo_meta_mpl["name"]
            current_row_text_cells: list[str] = []

            elapsed_key_for_run: str = f"{algo_prefix_key_mpl}_elapsed"
            # Robustly get elapsed value, checking both direct _elapsed and suffixed key
            raw_elapsed_val_for_run: Any = dataset_res_item_mpl.get(
                elapsed_key_for_run,
                dataset_res_item_mpl.get(
                    f"{algo_prefix_key_mpl}{METRIC_KEY_TO_PROPERTIES['_elapsed']['key']}"
                ),
            )
            is_skipped_entire_algo_run: bool = raw_elapsed_val_for_run == -1

            for metric_prop_col_mpl in ORDERED_METRIC_PROPERTIES:
                col_key_suffix_mpl: str = metric_prop_col_mpl["key"]
                text_for_cell: str = "N/A"
                is_cell_na: bool = True  # Assume NA unless a value is found
                is_cell_skipped: bool = False

                if metric_prop_col_mpl["type"] == "info":
                    text_for_cell = (
                        algo_name_display_mpl
                        if col_key_suffix_mpl == "algorithm_name"
                        else str(dataset_res_item_mpl.get(col_key_suffix_mpl, "N/A"))
                    )
                    is_cell_na = text_for_cell == "N/A"
                else:
                    full_metric_data_key: str = f"{algo_prefix_key_mpl}{col_key_suffix_mpl}"
                    raw_metric_val_for_cell: Any = dataset_res_item_mpl.get(full_metric_data_key)

                    if is_skipped_entire_algo_run and col_key_suffix_mpl not in ["_elapsed", "_memory"]:
                        text_for_cell = "SKIPPED"
                        is_cell_skipped = True
                    elif raw_metric_val_for_cell is not None:
                        if (
                            isinstance(raw_metric_val_for_cell, str)
                            and raw_metric_val_for_cell == "SKIPPED"
                        ):
                            text_for_cell = "SKIPPED"
                            is_cell_skipped = True
                        elif isinstance(raw_metric_val_for_cell, float | int) and np.isnan(
                            float(raw_metric_val_for_cell)
                        ):
                            text_for_cell = "N/A"  # Already NA
                        elif (
                            col_key_suffix_mpl == "_elapsed" and raw_elapsed_val_for_run == -1
                        ):  # Check the run's overall skip status
                            text_for_cell = "SKIPPED"
                            is_cell_skipped = True
                        else:
                            is_cell_na = False  # Has a valid value
                            formatter_name_mpl: str | None = metric_prop_col_mpl.get("format_func")
                            actual_formatter_mpl: Callable[[Any], str] | None = (
                                FORMATTER_FUNCTIONS.get(formatter_name_mpl)
                                if isinstance(formatter_name_mpl, str)
                                else None
                            )

                            raw_value_for_color_storage: float | None = None
                            try:  # Attempt to convert to float for storage and formatting
                                raw_value_for_color_storage = float(raw_metric_val_for_cell)
                            except (ValueError, TypeError):  # If not float, use string representation
                                text_for_cell = str(raw_metric_val_for_cell)

                            if raw_value_for_color_storage is not None:  # If it was float-convertible
                                if actual_formatter_mpl:
                                    text_for_cell = actual_formatter_mpl(raw_value_for_color_storage)
                                else:  # Default float formatting
                                    if abs(raw_value_for_color_storage) > 1e4 or (
                                        abs(raw_value_for_color_storage) < 0.001
                                        and raw_value_for_color_storage != 0
                                    ):
                                        text_for_cell = f"{raw_value_for_color_storage:.2e}"
                                    else:
                                        text_for_cell = f"{raw_value_for_color_storage:.3f}"
                                # Store raw float value for color normalization
                                if (
                                    not is_cell_skipped
                                    and metric_prop_col_mpl.get("higher_is_better") is not None
                                ):
                                    raw_cell_values_for_color[col_key_suffix_mpl][
                                        dataset_name_mpl
                                    ].append(raw_value_for_color_storage)
                            # If raw_value_for_color_storage was None (conversion failed), text_for_cell is already str(raw_metric_val_for_cell)
                    # else raw_metric_val_for_cell is None, so text_for_cell remains "N/A"
                current_row_text_cells.append(text_for_cell)
            table_cell_text_data.append(current_row_text_cells)

    if not table_cell_text_data:
        logger.warning("No data rows to build Matplotlib table after processing.")
        return

    metric_dataset_norms: defaultdict[str, dict[str, float | None]] = defaultdict(
        lambda: {"min": None, "max": None}
    )
    for metric_key_norm, datasets_data_norm in raw_cell_values_for_color.items():
        for dataset_name_norm, values_list_norm in datasets_data_norm.items():
            if values_list_norm:  # Ensure list is not empty
                metric_dataset_norms[metric_key_norm + "@" + dataset_name_norm]["min"] = min(
                    values_list_norm
                )
                metric_dataset_norms[metric_key_norm + "@" + dataset_name_norm]["max"] = max(
                    values_list_norm
                )

    best_value_flags: defaultdict[int, defaultdict[int, bool]] = defaultdict(
        lambda: defaultdict(lambda: False)
    )
    data_grouped_by_dataset_then_algo: defaultdict[str, list[list[str]]] = defaultdict(list)
    for original_row_idx, row_content in enumerate(table_cell_text_data):
        # Assuming "dataset" is always the first info column by convention from ORDERED_METRIC_PROPERTIES
        ds_name_group: str = row_content[TABLE_COLUMN_KEYS.index("dataset")]
        data_grouped_by_dataset_then_algo[ds_name_group].append(row_content)

    for dataset_name_group_best, algo_rows_in_ds_best in data_grouped_by_dataset_then_algo.items():
        for col_idx_best, col_metric_key_suffix_best in enumerate(TABLE_COLUMN_KEYS):
            metric_prop_for_best_val: MetricProperty | None = METRIC_KEY_TO_PROPERTIES.get(
                col_metric_key_suffix_best
            )
            if (
                not metric_prop_for_best_val
                or metric_prop_for_best_val["type"] == "info"
                or metric_prop_for_best_val.get("higher_is_better") is None
            ):
                continue

            values_for_this_metric_in_ds_raw: list[float] = []
            original_row_indices_in_ds: list[int] = []

            for row_data_item_best in algo_rows_in_ds_best:
                # Find original index of this row in table_cell_text_data
                # This is tricky if rows are not unique; assume they are for now by (dataset, algo_name) pair logic
                # A more robust way would be to pass original_row_idx along with row_data_item_best
                original_row_idx_lookup: int = -1
                for i_lookup, r_lookup in enumerate(table_cell_text_data):
                    if r_lookup == row_data_item_best:  # Simplistic match
                        original_row_idx_lookup = i_lookup
                        break
                if original_row_idx_lookup == -1:
                    continue  # Should not happen

                cell_text_val_best: str = row_data_item_best[col_idx_best]
                if cell_text_val_best not in ["N/A", "SKIPPED"]:
                    algo_name_of_row_best: str = row_data_item_best[
                        TABLE_COLUMN_KEYS.index("algorithm_name")
                    ]
                    algo_meta_of_row_best: dict[str, Any] | None = next(
                        (am for am in ALGORITHMS_METADATA if am["name"] == algo_name_of_row_best), None
                    )
                    original_ds_res_best: BenchmarkResultItem | None = next(
                        (
                            dsr
                            for dsr in benchmark_results_valid
                            if dsr["dataset"] == dataset_name_group_best
                        ),
                        None,
                    )

                    if algo_meta_of_row_best and original_ds_res_best:
                        raw_val_for_comp_best: Any = original_ds_res_best.get(
                            f"{algo_meta_of_row_best['prefix']}{col_metric_key_suffix_best}"
                        )
                        if raw_val_for_comp_best is not None and not (
                            isinstance(raw_val_for_comp_best, float) and np.isnan(raw_val_for_comp_best)
                        ):
                            try:
                                values_for_this_metric_in_ds_raw.append(float(raw_val_for_comp_best))
                                original_row_indices_in_ds.append(original_row_idx_lookup)
                            except (ValueError, TypeError):
                                pass

            if not values_for_this_metric_in_ds_raw:
                continue

            best_raw_val_found: float = (
                max(values_for_this_metric_in_ds_raw)
                if metric_prop_for_best_val["higher_is_better"]
                else min(values_for_this_metric_in_ds_raw)
            )

            for i_raw, raw_v_comp in enumerate(values_for_this_metric_in_ds_raw):
                if abs(raw_v_comp - best_raw_val_found) < 1e-9:  # Tolerance for float comparison
                    original_row_idx_of_best_val: int = original_row_indices_in_ds[i_raw]
                    best_value_flags[original_row_idx_of_best_val][col_idx_best] = True

    num_actual_rows_plot: int = len(table_cell_text_data)
    num_actual_cols_plot: int = len(TABLE_COLUMN_NAMES)

    base_col_width_plot: float = 0.04
    col_widths_plot: list[float] = [base_col_width_plot] * num_actual_cols_plot
    for c_idx_w_plot, key_w_plot in enumerate(TABLE_COLUMN_KEYS):
        key_type = METRIC_KEY_TO_PROPERTIES[key_w_plot]["type"]
        if key_w_plot == "dataset":
            col_widths_plot[c_idx_w_plot] = 0.10
        elif key_w_plot == "algorithm_name":
            col_widths_plot[c_idx_w_plot] = 0.12
        elif key_type == "info":
            col_widths_plot[c_idx_w_plot] = 0.05

    fig_width_final: float = sum(col_widths_plot) * 25
    fig_height_final: float = max(5.0, min((num_actual_rows_plot + 1) * 0.35, 50.0))
    fig_width_final = max(10.0, min(fig_width_final, 60.0))

    fig_table: plt.Figure
    ax_table_plot: plt.Axes
    fig_table, ax_table_plot = plt.subplots(figsize=(fig_width_final, fig_height_final), dpi=220)
    ax_table_plot.axis("tight")
    ax_table_plot.axis("off")

    mpl_table_obj: plt.Table = ax_table_plot.table(
        cellText=table_cell_text_data,
        colLabels=TABLE_COLUMN_NAMES,
        colWidths=col_widths_plot,
        loc="center",
        cellLoc="center",
    )
    mpl_table_obj.auto_set_font_size(False)
    mpl_table_obj.set_fontsize(6)

    for r_style in range(num_actual_rows_plot):
        dataset_name_for_row_color_style: str = table_cell_text_data[r_style][
            TABLE_COLUMN_KEYS.index("dataset")
        ]
        algo_name_for_row_color_style: str = table_cell_text_data[r_style][
            TABLE_COLUMN_KEYS.index("algorithm_name")
        ]
        algo_meta_for_row_color_style: dict[str, Any] | None = next(
            (am for am in ALGORITHMS_METADATA if am["name"] == algo_name_for_row_color_style), None
        )
        is_run_overall_skipped_style: bool = False
        original_ds_res_color_style: BenchmarkResultItem | None = None

        if algo_meta_for_row_color_style:
            original_ds_res_color_style = next(
                (
                    dsr
                    for dsr in benchmark_results_valid
                    if dsr["dataset"] == dataset_name_for_row_color_style
                ),
                None,
            )
            if original_ds_res_color_style:
                elapsed_val_color_check_style: Any = original_ds_res_color_style.get(
                    f"{algo_meta_for_row_color_style['prefix']}_elapsed"
                )
                is_run_overall_skipped_style = elapsed_val_color_check_style == -1

        for c_style in range(num_actual_cols_plot):
            cell_obj_to_style: plt.Cell = mpl_table_obj[r_style + 1, c_style]
            metric_prop_for_style: MetricProperty = METRIC_KEY_TO_PROPERTIES[TABLE_COLUMN_KEYS[c_style]]
            cell_text_for_style: str = table_cell_text_data[r_style][c_style]
            raw_value_for_cell_style: Any = None

            if (
                metric_prop_for_style["type"] != "info"
                and algo_meta_for_row_color_style
                and original_ds_res_color_style
            ):
                raw_value_for_cell_style = original_ds_res_color_style.get(
                    f"{algo_meta_for_row_color_style['prefix']}{metric_prop_for_style['key']}"
                )

            is_na_cell_style: bool = cell_text_for_style == "N/A"
            is_skipped_cell_style: bool = cell_text_for_style == "SKIPPED" or (
                is_run_overall_skipped_style
                and metric_prop_for_style["type"] != "info"
                and metric_prop_for_style["key"] not in ["_elapsed", "_memory"]
            )

            if metric_prop_for_style["type"] != "info":
                norm_key_style: str = (
                    metric_prop_for_style["key"] + "@" + dataset_name_for_row_color_style
                )
                min_val_norm_style: float | None = metric_dataset_norms.get(norm_key_style, {}).get(
                    "min"
                )
                max_val_norm_style: float | None = metric_dataset_norms.get(norm_key_style, {}).get(
                    "max"
                )
                face_color_cell: str = get_color_for_cell_mpl(
                    raw_value_for_cell_style,
                    metric_prop_for_style,
                    min_val_norm_style,
                    max_val_norm_style,
                    is_skipped_cell_style,
                    is_na_cell_style,
                )
                cell_obj_to_style.set_facecolor(face_color_cell)

            if best_value_flags[r_style][c_style]:
                cell_obj_to_style.get_text().set_weight("bold")
                cell_obj_to_style.get_text().set_color("black")

    for c_header_style in range(num_actual_cols_plot):
        header_cell_style: plt.Cell = mpl_table_obj[0, c_header_style]
        header_cell_style.get_text().set_weight("bold")
        header_cell_style.set_facecolor("#E0E0E0")
        header_cell_style.get_text().set_fontsize(7)

    fig_table.suptitle("Community Detection Benchmark Results", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout(pad=0.8)  # Увеличим pad с 0.5 до 0.8
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory for Matplotlib table: {output_dir}")

        plt.savefig(output_file, bbox_inches="tight", pad_inches=0.3)  # Увеличим pad_inches с 0.2 до 0.3
        logger.info(f"Matplotlib table saved successfully to {output_file}")
    except Exception as e_save_mpl:
        logger.error(f"Error saving Matplotlib table to {output_file}: {e_save_mpl}", exc_info=True)
    finally:  # Always close the figure
        plt.close(fig_table)


def plot_gcr_pcp_vs_jaccard_threshold(
    benchmark_results: BenchmarkResultsList, output_folder: str
) -> None:
    """Plots GCR and PCP vs. Jaccard Threshold for each dataset and algorithm.

    Generates a separate plot for each dataset, showing how GCR and PCP
    metrics change as the Jaccard similarity threshold for matching
    clusters varies.

    Args:
        benchmark_results: A list of benchmark result items.
        output_folder: The directory to save the generated plot images.
    """
    benchmark_results_valid: BenchmarkResultsList = get_valid_benchmark_results(benchmark_results)
    if not benchmark_results_valid:
        logger.warning("No valid benchmark results for GCR/PCP plot after filtering.")
        return

    active_algorithms_for_plot: list[dict[str, Any]] = []
    if benchmark_results_valid:
        # Check against the first valid dataset to see which algos have GCR/PCP keys
        # This assumes that if an algo has GCR/PCP keys, it has them for all its runs.
        first_ds_res_check: BenchmarkResultItem = benchmark_results_valid[0]
        for algo_meta_item_gcr_check in ALGORITHMS_METADATA:
            algo_prefix_gcr_check_str: str = algo_meta_item_gcr_check["prefix"]
            # Check if at least one GCR key for any Jaccard threshold exists for this algo prefix
            has_gcr_data_flag: bool = any(
                f"{algo_prefix_gcr_check_str}_gcr_jt{str(jt_val).replace('.', 'p')}"
                in first_ds_res_check
                for jt_val in JACCARD_THRESHOLDS_TO_TEST
            )
            if has_gcr_data_flag:
                active_algorithms_for_plot.append(algo_meta_item_gcr_check)

    if not active_algorithms_for_plot:
        logger.warning("No algorithms found with GCR/PCP data structures for plotting.")
        return

    num_active_algos_plot: int = len(active_algorithms_for_plot)
    # No need to check num_active_algos_plot == 0, caught by above.

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            logger.info(f"Created output directory for GCR/PCP plots: {output_folder}")
        except OSError as e_mkdir:
            logger.error(
                f"Could not create output directory {output_folder}: {e_mkdir}. Plots will not be saved."
            )
            return

    for ds_res_item_gcr_pcp_plot in benchmark_results_valid:
        dataset_name_for_plot: str = ds_res_item_gcr_pcp_plot["dataset"]

        fig_gcr_pcp: plt.Figure
        ax_gcr_plot: plt.Axes
        ax_pcp_plot: plt.Axes
        fig_gcr_pcp, (ax_gcr_plot, ax_pcp_plot) = plt.subplots(2, 1, figsize=(14, 12), dpi=150)
        plt.subplots_adjust(hspace=0.45)
        fig_gcr_pcp.suptitle(
            f"GCR & PCP vs. Jaccard Threshold for {dataset_name_for_plot}",
            fontsize=16,
            fontweight="bold",
        )

        x_jaccard_thresholds_plot: list[float] = JACCARD_THRESHOLDS_TO_TEST

        for algo_meta_plot_current in active_algorithms_for_plot:
            algo_prefix_for_data_plot: str = algo_meta_plot_current["prefix"]
            algo_display_name_for_plot: str = algo_meta_plot_current["name"]
            algo_plot_color_val: str = algo_meta_plot_current["color"]

            gcr_values_to_plot_list: list[float | np.nan] = []  # Use np.nan for missing data
            pcp_values_to_plot_list: list[float | np.nan] = []

            elapsed_key_gcr_skip_check_plot: str = f"{algo_prefix_for_data_plot}_elapsed"
            if ds_res_item_gcr_pcp_plot.get(elapsed_key_gcr_skip_check_plot) == -1:
                continue

            for jt_threshold_val_plot in x_jaccard_thresholds_plot:
                jt_key_str_plot: str = str(jt_threshold_val_plot).replace(".", "p")
                gcr_data_key_plot: str = f"{algo_prefix_for_data_plot}_gcr_jt{jt_key_str_plot}"
                pcp_data_key_plot: str = f"{algo_prefix_for_data_plot}_pcp_jt{jt_key_str_plot}"

                gcr_val: Any = ds_res_item_gcr_pcp_plot.get(gcr_data_key_plot)
                pcp_val: Any = ds_res_item_gcr_pcp_plot.get(pcp_data_key_plot)

                gcr_values_to_plot_list.append(float(gcr_val) if gcr_val is not None else np.nan)
                pcp_values_to_plot_list.append(float(pcp_val) if pcp_val is not None else np.nan)

            if not all(np.isnan(gcr_values_to_plot_list)):
                ax_gcr_plot.plot(
                    x_jaccard_thresholds_plot,
                    gcr_values_to_plot_list,
                    marker="o",
                    linestyle="-",
                    color=algo_plot_color_val,
                    label=algo_display_name_for_plot,
                )
            if not all(np.isnan(pcp_values_to_plot_list)):
                ax_pcp_plot.plot(
                    x_jaccard_thresholds_plot,
                    pcp_values_to_plot_list,
                    marker="x",
                    linestyle="--",
                    color=algo_plot_color_val,
                    label=algo_display_name_for_plot,
                )

        axes_to_style: list[tuple[plt.Axes, str]] = [
            (ax_gcr_plot, "Ground Truth Recall (GCR)"),
            (ax_pcp_plot, "Predicted Cluster Precision (PCP)"),
        ]
        for ax_curr_style, title_part_style in axes_to_style:
            ax_curr_style.set_xlabel("Jaccard Threshold", fontsize=10)
            ax_curr_style.set_ylabel(title_part_style.split(" (")[0], fontsize=10)
            ax_curr_style.set_title(f"{title_part_style} vs. Jaccard Threshold", fontsize=12)
            ax_curr_style.set_xticks(x_jaccard_thresholds_plot)
            ax_curr_style.set_ylim(-0.05, 1.05)
            ax_curr_style.grid(True, linestyle="--", alpha=0.7)
            num_legend_cols_gcr_pcp_plot: int = max(
                1, num_active_algos_plot // 4 if num_active_algos_plot > 3 else 1
            )
            ax_curr_style.legend(loc="best", fontsize=7, ncol=num_legend_cols_gcr_pcp_plot)

        safe_dataset_name_for_file_plot: str = (
            dataset_name_for_plot.replace(" ", "_").replace("/", "_").replace(":", "_").replace(".", "_")
        )  # Added dot replacement
        final_plot_filename_path: str = os.path.join(
            output_folder, f"{safe_dataset_name_for_file_plot}_gcr_pcp_plot.png"
        )
        try:
            plt.savefig(final_plot_filename_path, bbox_inches="tight")
            logger.info(f"GCR/PCP plot for {dataset_name_for_plot} saved to {final_plot_filename_path}")
        except Exception as e_save_gcr_pcp_plot:
            logger.error(
                f"Error saving GCR/PCP plot for {dataset_name_for_plot} to {final_plot_filename_path}: {e_save_gcr_pcp_plot}",
                exc_info=True,
            )
        finally:  # Always close the figure
            plt.close(fig_gcr_pcp)
