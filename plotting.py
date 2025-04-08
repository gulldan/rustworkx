import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Import formatters from the utils module
from benchmark_utils import format_time, format_memory

# Define colors for 2 algorithms
NX_COLOR = "#4285F4"  # Blue
RX_LOUVAIN_COLOR = "#34A853"  # Green
# RX_LEIDEN_COLOR = "#EA4335"  # Red (Removed)
GRID_COLOR = "#E5E5E5"

def create_comparison_chart(benchmark_results, output_file="community_detection_comparison.png"):
    """
    Create a comparative bar chart showing performance metrics with improved aesthetics and save to file.
    Handles NaN values for metrics where ground truth was unavailable.
    Includes NetworkX Louvain and RustWorkX Louvain.
    """
    if not benchmark_results:
        print("No benchmark results available to create comparison chart")
        return None

    datasets = [result["dataset"] for result in benchmark_results]
    n_datasets = len(datasets)
    n_algorithms = 2 # Now 2 algorithms
    
    plt.switch_backend("Agg")
    
    # Create figure for 12 subplots (7 external + 3 internal + 2 performance)
    num_metrics = 12 
    fig_height_per_metric = 6.0 
    fig, axes = plt.subplots(num_metrics, 1, figsize=(16, num_metrics * fig_height_per_metric), dpi=300) 
    plt.subplots_adjust(hspace=0.7) 
    
    # Flatten axes for easier indexing
    ax_list = axes.flatten()
    if len(ax_list) != num_metrics:
        raise ValueError(f"Expected {num_metrics} axes, but got {len(ax_list)}")
    
    (ax_ari, ax_nmi, ax_homogeneity, ax_completeness, ax_vmeasure, ax_fmi, 
     ax_modularity, ax_conductance, ax_internal_density, ax_avg_internal_path, 
     ax_time, ax_memory) = ax_list
    
    # Adjust bar width and positions for two algorithms
    bar_width = 0.35 # Adjusted bar width for 2 bars
    r = np.arange(n_datasets)
    r1 = r - bar_width / 2
    r2 = r + bar_width / 2
    # r3 removed

    # Grid settings
    grid_linewidth = 0.5

    # Helper function to add labels (now for two bars)
    def add_labels(ax, rects_list, data_list, format_str=".3f"):
        if len(rects_list) != len(data_list):
            raise ValueError("Number of rects lists must match number of data lists")
            
        for rects, data in zip(rects_list, data_list):
             for i, rect in enumerate(rects):
                y_val = rect.get_height()
                x_val = rect.get_x() + rect.get_width() / 2.0
                label_y_offset = 0.015 * ax.get_ylim()[1]
                if not np.isnan(data[i]):
                    ax.text(x_val, y_val + label_y_offset, f"{data[i]:{format_str}}", 
                            ha="center", va="bottom", fontweight="bold", fontsize=8)

    # Helper function to style axes (adjust xticks for two bars)
    def style_axis(ax, ylabel, title):
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
        ax.set_title(title, fontweight="bold", fontsize=16)
        ax.set_xticks(r) # Center ticks on the group
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
        ax.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=GRID_COLOR)
        ax.set_axisbelow(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_algorithms, 
                  frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)

    # --- External Quality Metrics --- 
    # ARI
    ari_nx = [result["nx_ari"] for result in benchmark_results]
    ari_rx_louvain = [result["rx_louvain_ari"] for result in benchmark_results]
    # ari_rx_leiden removed
    rects1 = ax_ari.bar(r1, ari_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_ari.bar(r2, ari_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_ari.set_ylim(0, 1.05)
    add_labels(ax_ari, [rects1, rects2], [ari_nx, ari_rx_louvain]) # Updated lists
    style_axis(ax_ari, "Adjusted Rand Index (ARI)", "External Quality - ARI (higher is better, NaN if no ground truth)")

    # NMI
    nmi_nx = [result["nx_nmi"] for result in benchmark_results]
    nmi_rx_louvain = [result["rx_louvain_nmi"] for result in benchmark_results]
    # nmi_rx_leiden removed
    rects1 = ax_nmi.bar(r1, nmi_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_nmi.bar(r2, nmi_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_nmi.set_ylim(0, 1.05)
    add_labels(ax_nmi, [rects1, rects2], [nmi_nx, nmi_rx_louvain])
    style_axis(ax_nmi, "Normalized Mutual Info (NMI)", "External Quality - NMI (higher is better, NaN if no ground truth)")
    
    # Homogeneity
    homogeneity_nx = [result["nx_homogeneity"] for result in benchmark_results]
    homogeneity_rx_louvain = [result["rx_louvain_homogeneity"] for result in benchmark_results]
    # homogeneity_rx_leiden removed
    rects1 = ax_homogeneity.bar(r1, homogeneity_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_homogeneity.bar(r2, homogeneity_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_homogeneity.set_ylim(0, 1.05)
    add_labels(ax_homogeneity, [rects1, rects2], [homogeneity_nx, homogeneity_rx_louvain])
    style_axis(ax_homogeneity, "Homogeneity Score", "External Quality - Homogeneity (higher is better, NaN if no ground truth)")

    # Completeness
    completeness_nx = [result["nx_completeness"] for result in benchmark_results]
    completeness_rx_louvain = [result["rx_louvain_completeness"] for result in benchmark_results]
    # completeness_rx_leiden removed
    rects1 = ax_completeness.bar(r1, completeness_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_completeness.bar(r2, completeness_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_completeness.set_ylim(0, 1.05)
    add_labels(ax_completeness, [rects1, rects2], [completeness_nx, completeness_rx_louvain])
    style_axis(ax_completeness, "Completeness Score", "External Quality - Completeness (higher is better, NaN if no ground truth)")

    # V-Measure
    vmeasure_nx = [result["nx_v_measure"] for result in benchmark_results]
    vmeasure_rx_louvain = [result["rx_louvain_v_measure"] for result in benchmark_results]
    # vmeasure_rx_leiden removed
    rects1 = ax_vmeasure.bar(r1, vmeasure_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_vmeasure.bar(r2, vmeasure_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_vmeasure.set_ylim(0, 1.05)
    add_labels(ax_vmeasure, [rects1, rects2], [vmeasure_nx, vmeasure_rx_louvain])
    style_axis(ax_vmeasure, "V-Measure Score", "External Quality - V-Measure (higher is better, NaN if no ground truth)")

    # FMI
    fmi_nx = [result["nx_fmi"] for result in benchmark_results]
    fmi_rx_louvain = [result["rx_louvain_fmi"] for result in benchmark_results]
    # fmi_rx_leiden removed
    rects1 = ax_fmi.bar(r1, fmi_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_fmi.bar(r2, fmi_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_fmi.set_ylim(0, 1.05)
    add_labels(ax_fmi, [rects1, rects2], [fmi_nx, fmi_rx_louvain])
    style_axis(ax_fmi, "Fowlkes–Mallows Index (FMI)", "External Quality - FMI (higher is better, NaN if no ground truth)")

    # --- Internal Quality Metrics ---
    # Modularity
    modularity_nx = [result["nx_modularity"] for result in benchmark_results]
    modularity_rx_louvain = [result["rx_louvain_modularity"] for result in benchmark_results]
    # modularity_rx_leiden removed
    all_mods = np.array([modularity_nx, modularity_rx_louvain]).flatten()
    min_mod = np.nanmin([np.nanmin(all_mods), 0])
    max_mod = np.nanmax(all_mods)
    max_mod = max_mod * 1.1 if not np.isnan(max_mod) else 1.0 
    min_mod = min_mod if not np.isnan(min_mod) else 0.0
    rects1 = ax_modularity.bar(r1, modularity_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_modularity.bar(r2, modularity_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_modularity.set_ylim(min_mod - abs(min_mod)*0.05, max_mod + abs(max_mod)*0.05) 
    add_labels(ax_modularity, [rects1, rects2], [modularity_nx, modularity_rx_louvain])
    style_axis(ax_modularity, "Modularity Score", "Internal Quality - Modularity (higher is better)")

    # Conductance
    conductance_nx = [result["nx_conductance"] for result in benchmark_results]
    conductance_rx_louvain = [result["rx_louvain_conductance"] for result in benchmark_results]
    # conductance_rx_leiden removed
    all_conds = np.array([conductance_nx, conductance_rx_louvain]).flatten()
    min_cond = 0 
    max_cond = np.nanmax(all_conds)
    max_cond = max_cond * 1.1 if not np.isnan(max_cond) else 1.0 
    max_cond = max(max_cond, 0.1) 
    rects1 = ax_conductance.bar(r1, conductance_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_conductance.bar(r2, conductance_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_conductance.set_ylim(min_cond, max_cond)
    add_labels(ax_conductance, [rects1, rects2], [conductance_nx, conductance_rx_louvain])
    style_axis(ax_conductance, "Avg. Conductance", "Internal Quality - Conductance (lower is better)")

    # Internal Edge Density
    int_density_nx = [result["nx_internal_density"] for result in benchmark_results]
    int_density_rx_louvain = [result["rx_louvain_internal_density"] for result in benchmark_results]
    # int_density_rx_leiden removed
    all_dens = np.array([int_density_nx, int_density_rx_louvain]).flatten()
    min_dens = 0 
    max_dens = np.nanmax(all_dens)
    max_dens = max_dens * 1.1 if not np.isnan(max_dens) else 1.0 
    max_dens = max(max_dens, 0.1) 
    rects1 = ax_internal_density.bar(r1, int_density_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_internal_density.bar(r2, int_density_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_internal_density.set_ylim(min_dens, max_dens)
    add_labels(ax_internal_density, [rects1, rects2], [int_density_nx, int_density_rx_louvain])
    style_axis(ax_internal_density, "Avg. Internal Density", "Internal Quality - Cluster Density (higher is better)")

    # Average Internal Shortest Path Length
    avg_path_nx = [result.get("nx_avg_internal_path", float('nan')) for result in benchmark_results]
    avg_path_rx_louvain = [result.get("rx_louvain_avg_internal_path", float('nan')) for result in benchmark_results]
    # avg_path_rx_leiden removed
    all_paths = np.array([avg_path_nx, avg_path_rx_louvain]).flatten()
    min_path = 0 
    max_path = np.nanmax(all_paths)
    max_path = max_path * 1.1 if not np.isnan(max_path) else 1.0 
    max_path = max(max_path, 1.0) 
    rects1 = ax_avg_internal_path.bar(r1, avg_path_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_avg_internal_path.bar(r2, avg_path_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_avg_internal_path.set_ylim(min_path, max_path)
    add_labels(ax_avg_internal_path, [rects1, rects2], [avg_path_nx, avg_path_rx_louvain])
    style_axis(ax_avg_internal_path, "Avg. Internal Path Length", "Internal Quality - Cluster Compactness (lower is better)")

    # --- Performance Metrics ---
    # Execution time
    time_nx = [result["nx_elapsed"] for result in benchmark_results]
    time_rx_louvain = [result["rx_louvain_elapsed"] for result in benchmark_results]
    # time_rx_leiden removed
    time_nx = [max(t, 1e-10) for t in time_nx]
    time_rx_louvain = [max(t, 1e-10) for t in time_rx_louvain]
    # time_rx_leiden max removed
    rects1 = ax_time.bar(r1, time_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_time.bar(r2, time_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_time.set_yscale("log")
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}" if x < 0.001 else f"{x:.3g}"))
    # Add time labels (now for two bars)
    all_times = [time_nx, time_rx_louvain]
    all_rects = [rects1, rects2]
    for rects, times in zip(all_rects, all_times):
        for i, v in enumerate(times):
            if v > 1e-10:
                ax_time.text(rects[i].get_x() + rects[i].get_width() / 2.0, v * 1.5, f"{format_time(v)}", 
                             ha="center", va="bottom", fontweight="bold", fontsize=8, color="#333333")
    style_axis(ax_time, "Execution Time (seconds)", "Performance - Execution Time (lower is better, log scale)")
    
    # Memory usage
    memory_nx = [result.get("nx_memory", 0) for result in benchmark_results]
    memory_rx_louvain = [result.get("rx_louvain_memory", 0) for result in benchmark_results]
    # memory_rx_leiden removed
    memory_nx = [max(m, 1e-6) for m in memory_nx]
    memory_rx_louvain = [max(m, 1e-6) for m in memory_rx_louvain]
    # memory_rx_leiden max removed
    rects1 = ax_memory.bar(r1, memory_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_memory.bar(r2, memory_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    # rects3 removed
    ax_memory.set_yscale("log")
    ax_memory.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_memory(x)))
    # Add memory labels (now for two bars)
    all_mems = [memory_nx, memory_rx_louvain]
    all_rects = [rects1, rects2]
    for rects, mems in zip(all_rects, all_mems):
        for i, v in enumerate(mems):
             if v > 1e-6:
                ax_memory.text(rects[i].get_x() + rects[i].get_width() / 2.0, v * 1.5, f"{format_memory(v)}",
                               ha="center", va="bottom", fontweight="bold", fontsize=8, color="#333333")
    style_axis(ax_memory, "Memory Usage (MB)", "Performance - Memory Usage (lower is better, log scale)")
    
    # Overall title and layout
    fig.suptitle("Community Detection Algorithm Comparison", fontsize=22, fontweight="bold", y=1.0)
    plt.tight_layout(rect=(0, 0.05, 1, 0.99)) 
    
    # Save the chart
    plt.savefig(output_file, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"Enhanced comparison chart saved to '{output_file}'")
    plt.close(fig)
    
    return None


def generate_results_table(results):
    """Generates a Markdown table from the benchmark results."""
    if not results:
        return "No results to display."

    # Define headers for 2 algorithms
    headers = [
        "Dataset", "Nodes", "Edges", "Has GT", 
        "NX L Time", "RX L Time", 
        "NX L Mem", "RX L Mem", 
        "NX L Comms", "RX L Comms", 
        "NX L Mod", "RX L Mod", 
        "NX L Cond", "RX L Cond", 
        "NX L IntDens", "RX L IntDens", 
        "NX L AvgPath", "RX L AvgPath",  
        "NX L ARI", "RX L ARI", 
        "NX L NMI", "RX L NMI" 
    ]
    
    # Create separator line
    sep = ["---"] * len(headers)

    table = [" | ".join(headers), " | ".join(sep)]

    # Format result rows
    for res in results:
        if res is None:
            print("Warning: Skipping None result in generate_results_table.")
            continue

        # Format performance with bold for faster time across 2 algos
        def get_best_idx(values):
            non_zero_values = [(v, i) for i, v in enumerate(values) if v > 0]
            if not non_zero_values: return -1
            return min(non_zero_values)[1]

        # Adjust times/mems for 2 algorithms
        times = [
            res.get("nx_elapsed", 0),
            res.get("rx_louvain_elapsed", 0),
        ]
        mems = [
            res.get("nx_memory", 0),
            res.get("rx_louvain_memory", 0),
        ]

        best_time_idx = get_best_idx(times)
        best_mem_idx = get_best_idx(mems)

        time_strs = [format_time(t) for t in times]
        mem_strs = [format_memory(m) for m in mems]

        if best_time_idx != -1: time_strs[best_time_idx] = f"**{time_strs[best_time_idx]}**"
        if best_mem_idx != -1: mem_strs[best_mem_idx] = f"**{mem_strs[best_mem_idx]}**"

        # Format metrics, handling NaN
        def fmt(key, decimals=4):
            val = res.get(key, float("nan"))
            return f"{val:.{decimals}f}" if not np.isnan(val) else "NaN"

        row = [
            res["dataset"],
            str(res["nodes"]),
            str(res["edges"]),
            "Yes" if res["has_ground_truth"] else "No",
            time_strs[0], time_strs[1], 
            mem_strs[0], mem_strs[1], 
            str(res.get("nx_num_comms", 0)), str(res.get("rx_louvain_num_comms", 0)), 
            fmt("nx_modularity"), fmt("rx_louvain_modularity"), 
            fmt("nx_conductance"), fmt("rx_louvain_conductance"), 
            fmt("nx_internal_density"), fmt("rx_louvain_internal_density"), 
            fmt("nx_avg_internal_path"), fmt("rx_louvain_avg_internal_path"),
            fmt("nx_ari"), fmt("rx_louvain_ari"), 
            fmt("nx_nmi"), fmt("rx_louvain_nmi") 
        ]
        table.append(" | ".join(row))

    return "\n".join(table)


def generate_results_table_matplotlib(results, output_file="benchmark_results_table.png"):
    """Generates a table image from benchmark results using Matplotlib."""
    if not results:
        print("No results to generate table image.")
        return

    # Define headers for 2 algorithms
    headers = [
        "Dataset", "Nodes", "Edges", "Has GT", 
        "NX L Time", "RX L Time", 
        "NX L Mem", "RX L Mem", 
        "NX L Comms", "RX L Comms", 
        "NX L Mod", "RX L Mod", 
        "NX L Cond", "RX L Cond", 
        "NX L IntDens", "RX L IntDens", 
        "NX L AvgPath", "RX L AvgPath", 
        "NX L ARI", "RX L ARI", 
        "NX L NMI", "RX L NMI" 
    ]

    # Prepare data for the table
    table_data = []
    for res in results:
        def fmt(key, decimals=3):
            val = res.get(key, float("nan"))
            return f"{val:.{decimals}f}" if not np.isnan(val) else "NaN"

        row_data = [
            res["dataset"], str(res["nodes"]), str(res["edges"]),
            "Yes" if res["has_ground_truth"] else "No",
            format_time(res.get("nx_elapsed", 0)), format_time(res.get("rx_louvain_elapsed", 0)), 
            format_memory(res.get("nx_memory", 0)), format_memory(res.get("rx_louvain_memory", 0)), 
            str(res.get("nx_num_comms", 0)), str(res.get("rx_louvain_num_comms", 0)), 
            fmt("nx_modularity"), fmt("rx_louvain_modularity"), 
            fmt("nx_conductance"), fmt("rx_louvain_conductance"), 
            fmt("nx_internal_density"), fmt("rx_louvain_internal_density"), 
            fmt("nx_avg_internal_path"), fmt("rx_louvain_avg_internal_path"),
            fmt("nx_ari"), fmt("rx_louvain_ari"), 
            fmt("nx_nmi"), fmt("rx_louvain_nmi") 
        ]
        table_data.append(row_data)

    # Create figure and table
    num_rows = len(table_data) + 1
    num_cols = len(headers)
    fig_width = num_cols * 1.1 
    fig_height = num_rows * 0.4
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")
    
    the_table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8) 
    the_table.scale(1, 1.5)

    # Style the table header
    for (i, j), cell in the_table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#40466e")
        elif i % 2 == 0:
            cell.set_facecolor("#f2f2f2")
        cell.set_text_props(ha="center")

    # Apply bold formatting for best performance (Time and Memory) - adjust for 2 algos
    def get_best_idx(values):
        non_zero_values = [(v, i) for i, v in enumerate(values) if v > 0]
        if not non_zero_values: return -1
        return min(non_zero_values)[1]

    time_indices = [headers.index(h) for h in ["NX L Time", "RX L Time"]]
    mem_indices = [headers.index(h) for h in ["NX L Mem", "RX L Mem"]]

    for i, res in enumerate(results):
        row_idx = i + 1
        # Adjust times/mems for 2 algos
        times = [
            res.get("nx_elapsed", 0),
            res.get("rx_louvain_elapsed", 0),
        ]
        mems = [
            res.get("nx_memory", 0),
            res.get("rx_louvain_memory", 0),
        ]
        best_time_idx = get_best_idx(times)
        best_mem_idx = get_best_idx(mems)
        if best_time_idx != -1:
            the_table[(row_idx, time_indices[best_time_idx])].set_text_props(weight="bold")
        if best_mem_idx != -1:
            the_table[(row_idx, mem_indices[best_mem_idx])].set_text_props(weight="bold")

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"Results table image saved to: {output_file}")
    except Exception as e:
        print(f"Error saving table image: {e}")
    finally:
        plt.close(fig) 