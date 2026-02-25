# ruff: noqa: E501
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plotting_shared import (
    ALGORITHMS_METADATA,
    JACCARD_THRESHOLDS_TO_TEST,
    BenchmarkResultItem,
    BenchmarkResultsList,
    get_valid_benchmark_results,
)

logger = logging.getLogger(__name__)


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
