# ruff: noqa: E501
import logging
import os
from collections import defaultdict
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from benchmark_utils import format_bool
from plotting_shared import (
    ALGORITHMS_METADATA,
    METRIC_KEY_TO_PROPERTIES,
    ORDERED_METRIC_PROPERTIES,
    TABLE_COLUMN_KEYS,
    TABLE_COLUMN_NAMES,
    TABLE_COLUMN_NAMES_MPL,
    BenchmarkResultItem,
    BenchmarkResultsList,
    MetricProperty,
    _coerce_float_metric_value,
    _format_metric_cell_value,
    _get_algo_run_elapsed_and_presence,
    get_valid_benchmark_results,
    style_exact_match_markdown_cell,
)

logger = logging.getLogger(__name__)


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

            # If the primary elapsed key is not found, this algo run was likely not even initialized for this dataset
            has_algo_run, raw_elapsed_val_md = _get_algo_run_elapsed_and_presence(
                dataset_res_item, algo_key_prefix_md
            )
            if not has_algo_run:
                continue

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
                    num_gt_clusters_for_algo_run: int = 0
                    try:
                        num_gt_clusters_raw = dataset_res_item.get("num_gt_clusters", 0)
                        num_gt_clusters_for_algo_run = int(num_gt_clusters_raw)
                    except (TypeError, ValueError):
                        num_gt_clusters_for_algo_run = 0

                    formatted_cell_value: str = _format_metric_cell_value(
                        raw_metric_val_md,
                        metric_prop_md_col,
                        raw_elapsed_val=raw_elapsed_val_md,
                        is_algo_run_skipped=is_skipped_this_algo_run,
                        num_gt_clusters=num_gt_clusters_for_algo_run,
                    )

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
                    current_md_row_cells.append(
                        style_exact_match_markdown_cell(col_key_md, formatted_cell_value)
                    )

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

    if metric_prop["key"].startswith("_exact_match_"):
        exact_status: str = format_bool(value)
        if exact_status == "Yes":
            return "#D9FBE0"
        if exact_status == "No":
            return "#FEE4E2"
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
            # Check if any key for this algo exists (not just _elapsed which might be missing if only info keys are present)
            has_algo_entry, elapsed_for_algo = _get_algo_run_elapsed_and_presence(
                dataset_res_item_mpl, algo_prefix_key_mpl
            )
            if has_algo_entry and elapsed_for_algo is not None:
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
            if raw_elapsed_val_for_run is None:
                continue

            is_skipped_entire_algo_run: bool = raw_elapsed_val_for_run == -1

            for metric_prop_col_mpl in ORDERED_METRIC_PROPERTIES:
                col_key_suffix_mpl: str = metric_prop_col_mpl["key"]
                is_cell_skipped: bool = is_skipped_entire_algo_run
                text_for_cell: str = "N/A"

                if metric_prop_col_mpl["type"] == "info":
                    text_for_cell = (
                        algo_name_display_mpl
                        if col_key_suffix_mpl == "algorithm_name"
                        else str(dataset_res_item_mpl.get(col_key_suffix_mpl, "N/A"))
                    )
                else:
                    full_metric_data_key: str = f"{algo_prefix_key_mpl}{col_key_suffix_mpl}"
                    raw_metric_val_for_cell: Any = dataset_res_item_mpl.get(full_metric_data_key)
                    text_for_cell = _format_metric_cell_value(
                        raw_metric_val_for_cell,
                        metric_prop_col_mpl,
                        raw_elapsed_val=raw_elapsed_val_for_run,
                        is_algo_run_skipped=is_skipped_entire_algo_run,
                        num_gt_clusters=int(dataset_res_item_mpl.get("num_gt_clusters", 0)),
                    )
                    raw_value_for_color_storage = _coerce_float_metric_value(raw_metric_val_for_cell)
                    if (
                        not is_cell_skipped
                        and raw_value_for_color_storage is not None
                        and metric_prop_col_mpl.get("higher_is_better") is not None
                        and text_for_cell != "SKIPPED"
                    ):
                        raw_cell_values_for_color[col_key_suffix_mpl][dataset_name_mpl].append(
                            raw_value_for_color_storage
                        )
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
        elif key_w_plot.startswith("_exact_match_"):
            col_widths_plot[c_idx_w_plot] = 0.07
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
        colLabels=TABLE_COLUMN_NAMES_MPL,
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

            if metric_prop_for_style["key"].startswith("_exact_match_"):
                if cell_text_for_style == "Yes":
                    cell_obj_to_style.get_text().set_color("black")
                    cell_obj_to_style.get_text().set_weight("bold")
                elif cell_text_for_style == "No":
                    cell_obj_to_style.get_text().set_color("black")
                    cell_obj_to_style.get_text().set_weight("bold")

            if best_value_flags[r_style][c_style]:
                cell_obj_to_style.get_text().set_weight("bold")
                cell_obj_to_style.get_text().set_color("black")

    for c_header_style in range(num_actual_cols_plot):
        header_cell_style: plt.Cell = mpl_table_obj[0, c_header_style]
        header_cell_style.get_text().set_weight("bold")
        header_cell_style.set_facecolor("#E0E0E0")
        if TABLE_COLUMN_KEYS[c_header_style].startswith("_exact_match_"):
            header_cell_style.get_text().set_fontsize(6)
        else:
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
