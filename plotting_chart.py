# ruff: noqa: E501
import logging
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from benchmark_config_data import GRID_COLOR
from benchmark_utils import format_memory, format_time
from plotting_shared import (
    ALGORITHMS_METADATA,
    FORMATTER_FUNCTIONS,
    METRIC_KEY_TO_PROPERTIES,
    ORDERED_METRIC_PROPERTIES,
    BenchmarkResultsList,
    MetricProperty,
    get_valid_benchmark_results,
)

logger = logging.getLogger(__name__)


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
