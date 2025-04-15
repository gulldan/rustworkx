import matplotlib.pyplot as plt
import numpy as np

# Import formatters from the utils module
from benchmark_utils import format_memory, format_time
from matplotlib.ticker import FuncFormatter

# Define colors for 3 algorithms (NX, RX Louvain, RX Leiden)
NX_COLOR = "#4285F4"  # Blue
RX_LOUVAIN_COLOR = "#34A853"  # Green
RX_LEIDEN_COLOR = "#EA4335"  # Red (Was previously removed)
GRID_COLOR = "#E5E5E5"

def create_comparison_chart(benchmark_results, output_file="community_detection_comparison.png"):
    """
    Create a comparative bar chart showing performance metrics with improved aesthetics and save to file.
    Handles NaN values for metrics where ground truth was unavailable.
    Includes NetworkX Louvain, RustWorkX Louvain, and RustWorkX Leiden.
    """
    if not benchmark_results:
        print("No benchmark results available to create comparison chart")
        return None

    datasets = [result["dataset"] for result in benchmark_results]
    n_datasets = len(datasets)
    n_algorithms = 3 # Now 3 algorithms
    
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
    
    # Adjust bar width and positions for three algorithms
    bar_width = 0.25 # Adjusted bar width for 3 bars
    r = np.arange(n_datasets)
    r1 = r - bar_width
    r2 = r
    r3 = r + bar_width

    # Grid settings
    grid_linewidth = 0.5

    # Helper function to add labels (now for three bars)
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

    # Helper function to style axes (adjust xticks for three bars)
    def style_axis(ax, ylabel, title):
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
        ax.set_title(title, fontweight="bold", fontsize=16)
        ax.set_xticks(r) # Center ticks on the group
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
        ax.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=GRID_COLOR)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=n_algorithms, 
                  frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)

    # --- External Quality Metrics --- 
    # ARI
    ari_nx = [result["nx_ari"] for result in benchmark_results]
    ari_rx_louvain = [result["rx_louvain_ari"] for result in benchmark_results]
    ari_rx_leiden = [result["rx_leiden_ari"] for result in benchmark_results] # Add Leiden data
    rects1 = ax_ari.bar(r1, ari_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_ari.bar(r2, ari_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_ari.bar(r3, ari_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden bar
    ax_ari.set_ylim(0, 1.05)
    add_labels(ax_ari, [rects1, rects2, rects3], [ari_nx, ari_rx_louvain, ari_rx_leiden]) # Updated lists
    style_axis(ax_ari, "Adjusted Rand Index (ARI)", "External Quality - ARI (higher is better, NaN if no ground truth)")

    # NMI
    nmi_nx = [result["nx_nmi"] for result in benchmark_results]
    nmi_rx_louvain = [result["rx_louvain_nmi"] for result in benchmark_results]
    nmi_rx_leiden = [result["rx_leiden_nmi"] for result in benchmark_results] # Add Leiden
    rects1 = ax_nmi.bar(r1, nmi_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_nmi.bar(r2, nmi_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_nmi.bar(r3, nmi_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_nmi.set_ylim(0, 1.05)
    add_labels(ax_nmi, [rects1, rects2, rects3], [nmi_nx, nmi_rx_louvain, nmi_rx_leiden])
    style_axis(ax_nmi, "Normalized Mutual Info (NMI)", "External Quality - NMI (higher is better, NaN if no ground truth)")
    
    # Homogeneity
    homogeneity_nx = [result["nx_homogeneity"] for result in benchmark_results]
    homogeneity_rx_louvain = [result["rx_louvain_homogeneity"] for result in benchmark_results]
    homogeneity_rx_leiden = [result["rx_leiden_homogeneity"] for result in benchmark_results] # Add Leiden
    rects1 = ax_homogeneity.bar(r1, homogeneity_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_homogeneity.bar(r2, homogeneity_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_homogeneity.bar(r3, homogeneity_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_homogeneity.set_ylim(0, 1.05)
    add_labels(ax_homogeneity, [rects1, rects2, rects3], [homogeneity_nx, homogeneity_rx_louvain, homogeneity_rx_leiden])
    style_axis(ax_homogeneity, "Homogeneity Score", "External Quality - Homogeneity (higher is better, NaN if no ground truth)")

    # Completeness
    completeness_nx = [result["nx_completeness"] for result in benchmark_results]
    completeness_rx_louvain = [result["rx_louvain_completeness"] for result in benchmark_results]
    completeness_rx_leiden = [result["rx_leiden_completeness"] for result in benchmark_results] # Add Leiden
    rects1 = ax_completeness.bar(r1, completeness_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_completeness.bar(r2, completeness_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_completeness.bar(r3, completeness_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_completeness.set_ylim(0, 1.05)
    add_labels(ax_completeness, [rects1, rects2, rects3], [completeness_nx, completeness_rx_louvain, completeness_rx_leiden])
    style_axis(ax_completeness, "Completeness Score", "External Quality - Completeness (higher is better, NaN if no ground truth)")

    # V-Measure
    vmeasure_nx = [result["nx_v_measure"] for result in benchmark_results]
    vmeasure_rx_louvain = [result["rx_louvain_v_measure"] for result in benchmark_results]
    vmeasure_rx_leiden = [result["rx_leiden_v_measure"] for result in benchmark_results] # Add Leiden
    rects1 = ax_vmeasure.bar(r1, vmeasure_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_vmeasure.bar(r2, vmeasure_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_vmeasure.bar(r3, vmeasure_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_vmeasure.set_ylim(0, 1.05)
    add_labels(ax_vmeasure, [rects1, rects2, rects3], [vmeasure_nx, vmeasure_rx_louvain, vmeasure_rx_leiden])
    style_axis(ax_vmeasure, "V-Measure Score", "External Quality - V-Measure (higher is better, NaN if no ground truth)")

    # FMI
    fmi_nx = [result["nx_fmi"] for result in benchmark_results]
    fmi_rx_louvain = [result["rx_louvain_fmi"] for result in benchmark_results]
    fmi_rx_leiden = [result["rx_leiden_fmi"] for result in benchmark_results] # Add Leiden
    rects1 = ax_fmi.bar(r1, fmi_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_fmi.bar(r2, fmi_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_fmi.bar(r3, fmi_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_fmi.set_ylim(0, 1.05)
    add_labels(ax_fmi, [rects1, rects2, rects3], [fmi_nx, fmi_rx_louvain, fmi_rx_leiden])
    style_axis(ax_fmi, "Fowlkes–Mallows Index (FMI)", "External Quality - FMI (higher is better, NaN if no ground truth)")

    # --- Internal Quality Metrics ---
    # Modularity
    modularity_nx = [result["nx_modularity"] for result in benchmark_results]
    modularity_rx_louvain = [result["rx_louvain_modularity"] for result in benchmark_results]
    modularity_rx_leiden = [result["rx_leiden_modularity"] for result in benchmark_results] # Add Leiden
    all_mods = np.array([modularity_nx, modularity_rx_louvain, modularity_rx_leiden]).flatten()
    min_mod = np.nanmin([np.nanmin(all_mods), 0])
    max_mod = np.nanmax(all_mods)
    max_mod = max_mod * 1.1 if not np.isnan(max_mod) else 1.0 
    min_mod = min_mod if not np.isnan(min_mod) else 0.0
    rects1 = ax_modularity.bar(r1, modularity_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_modularity.bar(r2, modularity_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_modularity.bar(r3, modularity_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_modularity.set_ylim(min_mod - abs(min_mod)*0.05, max_mod + abs(max_mod)*0.05) 
    add_labels(ax_modularity, [rects1, rects2, rects3], [modularity_nx, modularity_rx_louvain, modularity_rx_leiden])
    style_axis(ax_modularity, "Modularity Score", "Internal Quality - Modularity (higher is better)")

    # Conductance
    conductance_nx = [result["nx_conductance"] for result in benchmark_results]
    conductance_rx_louvain = [result["rx_louvain_conductance"] for result in benchmark_results]
    conductance_rx_leiden = [result["rx_leiden_conductance"] for result in benchmark_results] # Add Leiden
    all_conds = np.array([conductance_nx, conductance_rx_louvain, conductance_rx_leiden]).flatten()
    min_cond = 0 
    max_cond = np.nanmax(all_conds)
    max_cond = max_cond * 1.1 if not np.isnan(max_cond) else 1.0 
    max_cond = max(max_cond, 0.1) 
    rects1 = ax_conductance.bar(r1, conductance_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_conductance.bar(r2, conductance_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_conductance.bar(r3, conductance_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_conductance.set_ylim(min_cond, max_cond)
    add_labels(ax_conductance, [rects1, rects2, rects3], [conductance_nx, conductance_rx_louvain, conductance_rx_leiden])
    style_axis(ax_conductance, "Avg. Conductance", "Internal Quality - Conductance (lower is better)")

    # Internal Edge Density
    int_density_nx = [result["nx_internal_density"] for result in benchmark_results]
    int_density_rx_louvain = [result["rx_louvain_internal_density"] for result in benchmark_results]
    int_density_rx_leiden = [result["rx_leiden_internal_density"] for result in benchmark_results] # Add Leiden
    all_dens = np.array([int_density_nx, int_density_rx_louvain, int_density_rx_leiden]).flatten()
    min_dens = 0 
    max_dens = np.nanmax(all_dens)
    max_dens = max_dens * 1.1 if not np.isnan(max_dens) else 1.0 
    max_dens = max(max_dens, 0.1) 
    rects1 = ax_internal_density.bar(r1, int_density_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_internal_density.bar(r2, int_density_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_internal_density.bar(r3, int_density_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_internal_density.set_ylim(min_dens, max_dens)
    add_labels(ax_internal_density, [rects1, rects2, rects3], [int_density_nx, int_density_rx_louvain, int_density_rx_leiden])
    style_axis(ax_internal_density, "Avg. Internal Density", "Internal Quality - Cluster Density (higher is better)")

    # Average Internal Path Length (Assuming this metric is not calculated for Leiden yet)
    avg_path_nx = [result.get("nx_avg_internal_path", float("nan")) for result in benchmark_results]
    avg_path_rx_louvain = [result.get("rx_louvain_avg_internal_path", float("nan")) for result in benchmark_results]
    avg_path_rx_leiden = [result.get("rx_leiden_avg_internal_path", float("nan")) for result in benchmark_results] # Add Leiden placeholder
    all_paths = np.array([avg_path_nx, avg_path_rx_louvain, avg_path_rx_leiden]).flatten()
    min_path = 0 
    max_path = np.nanmax(all_paths)
    max_path = max_path * 1.1 if not np.isnan(max_path) else 1.0 
    max_path = max(max_path, 1.0) 
    rects1 = ax_avg_internal_path.bar(r1, avg_path_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_avg_internal_path.bar(r2, avg_path_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_avg_internal_path.bar(r3, avg_path_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_avg_internal_path.set_ylim(min_path, max_path)
    add_labels(ax_avg_internal_path, [rects1, rects2, rects3], [avg_path_nx, avg_path_rx_louvain, avg_path_rx_leiden])
    style_axis(ax_avg_internal_path, "Avg. Internal Path Length", "Internal Quality - Cluster Compactness (lower is better)")

    # --- Performance Metrics ---
    # Execution time
    time_nx = [result["nx_elapsed"] for result in benchmark_results]
    time_rx_louvain = [result["rx_louvain_elapsed"] for result in benchmark_results]
    time_rx_leiden = [result["rx_leiden_elapsed"] for result in benchmark_results] # Add Leiden
    time_nx = [max(t, 1e-10) for t in time_nx]
    time_rx_louvain = [max(t, 1e-10) for t in time_rx_louvain]
    time_rx_leiden = [max(t, 1e-10) for t in time_rx_leiden] # Add Leiden
    rects1 = ax_time.bar(r1, time_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_time.bar(r2, time_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_time.bar(r3, time_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_time.set_yscale("log")
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}" if x < 0.001 else f"{x:.3g}"))
    # Add time labels (now for three bars)
    all_times = [time_nx, time_rx_louvain, time_rx_leiden]
    all_rects = [rects1, rects2, rects3]
    for rects, times in zip(all_rects, all_times):
        for i, v in enumerate(times):
            if v > 1e-10:
                ax_time.text(rects[i].get_x() + rects[i].get_width() / 2.0, v * 1.5, f"{format_time(v)}", 
                             ha="center", va="bottom", fontweight="bold", fontsize=8, color="#333333")
    style_axis(ax_time, "Execution Time (seconds)", "Performance - Execution Time (lower is better, log scale)")
    
    # Memory usage
    memory_nx = [result.get("nx_memory", 0) for result in benchmark_results]
    memory_rx_louvain = [result.get("rx_louvain_memory", 0) for result in benchmark_results]
    memory_rx_leiden = [result.get("rx_leiden_memory", 0) for result in benchmark_results] # Add Leiden
    memory_nx = [max(m, 1e-6) for m in memory_nx]
    memory_rx_louvain = [max(m, 1e-6) for m in memory_rx_louvain]
    memory_rx_leiden = [max(m, 1e-6) for m in memory_rx_leiden] # Add Leiden
    rects1 = ax_memory.bar(r1, memory_nx, width=bar_width, label="NX Louvain", color=NX_COLOR)
    rects2 = ax_memory.bar(r2, memory_rx_louvain, width=bar_width, label="RX Louvain", color=RX_LOUVAIN_COLOR)
    rects3 = ax_memory.bar(r3, memory_rx_leiden, width=bar_width, label="RX Leiden", color=RX_LEIDEN_COLOR) # Add Leiden
    ax_memory.set_yscale("log")
    ax_memory.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_memory(x)))
    # Add memory labels (now for three bars)
    all_mems = [memory_nx, memory_rx_louvain, memory_rx_leiden]
    all_rects = [rects1, rects2, rects3]
    for rects, mems in zip(all_rects, all_mems):
        for i, v in enumerate(mems):
             if v > 1e-6:
                ax_memory.text(rects[i].get_x() + rects[i].get_width() / 2.0, v * 1.5, f"{format_memory(v)}",
                               ha="center", va="bottom", fontweight="bold", fontsize=8, color="#333333")
    style_axis(ax_memory, "Peak Memory Usage (MB)", "Performance - Peak Memory Usage (lower is better, log scale)")
    
    # Overall figure title
    fig.suptitle("Community Detection Benchmark: NetworkX vs RustWorkX (Louvain & Leiden)", fontsize=20, fontweight="bold", y=1.01)
    
    # Save the figure
    try:
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0.3)
        print(f"Comparison chart saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving comparison chart: {e}")
    plt.close(fig)
    return output_file


def generate_results_table(results):
    """
    Generates a markdown table string summarizing benchmark results.
    Includes NetworkX Louvain, RustWorkX Louvain, and RustWorkX Leiden.
    Highlights the best result for each metric.
    """
    if not results:
        return "No benchmark results to generate table."

    # --- Header --- 
    header = "| Dataset | Nodes | Edges | Algorithm | Time | Memory | # Comms | Modularity | Conductance | Int. Density | ARI | NMI | V-Measure |\n"
    separator = "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    table_string = header + separator

    # --- Helper function to get best index (lower is better for time/memory/conductance, higher for others) ---
    def get_best_idx(values, lower_is_better=False):
        valid_values = [(i, v) for i, v in enumerate(values) if not (isinstance(v, float) and np.isnan(v))]
        if not valid_values:
            return -1
        if lower_is_better:
            best_idx, _ = min(valid_values, key=lambda item: item[1])
        else:
            best_idx, _ = max(valid_values, key=lambda item: item[1])
        return best_idx

    # --- Format values, handling None/NaN and highlighting best ---
    def fmt(key, decimals=4, lower_is_better=False, is_time=False, is_memory=False):
        nx_val = result.get(f"nx_{key}", float("nan"))
        rx_l_val = result.get(f"rx_louvain_{key}", float("nan"))
        rx_p_val = result.get(f"rx_leiden_{key}", float("nan")) # Changed from cpm to leiden
        
        values = [nx_val, rx_l_val, rx_p_val]
        best_idx = get_best_idx(values, lower_is_better)
        
        formatted_values = []
        for i, v in enumerate(values):
            mark = " **" if i == best_idx else ""
            end_mark = "**" if i == best_idx else ""
            if isinstance(v, float) and np.isnan(v):
                formatted_values.append("-")
            elif is_time:
                formatted_values.append(f"{mark}{format_time(v)}{end_mark}")
            elif is_memory:
                 formatted_values.append(f"{mark}{format_memory(v)}{end_mark}")
            elif isinstance(v, int):
                 formatted_values.append(f"{mark}{v}{end_mark}")
            else:
                formatted_values.append(f"{mark}{v:.{decimals}f}{end_mark}")
        return formatted_values

    # --- Table Rows --- 
    for result in results:
        dataset_name = result["dataset"]
        nodes = result["nodes"]
        edges = result["edges"]

        # Format metrics for all three algorithms
        time_fmt = fmt("elapsed", is_time=True, lower_is_better=True)
        mem_fmt = fmt("memory", is_memory=True, lower_is_better=True)
        comms_fmt = fmt("num_comms", decimals=0)
        mod_fmt = fmt("modularity")
        cond_fmt = fmt("conductance", lower_is_better=True)
        dens_fmt = fmt("internal_density")
        ari_fmt = fmt("ari")
        nmi_fmt = fmt("nmi")
        vm_fmt = fmt("v_measure")

        # Create rows for each algorithm
        table_string += f"| {dataset_name} | {nodes} | {edges} | NX Louvain | {time_fmt[0]} | {mem_fmt[0]} | {comms_fmt[0]} | {mod_fmt[0]} | {cond_fmt[0]} | {dens_fmt[0]} | {ari_fmt[0]} | {nmi_fmt[0]} | {vm_fmt[0]} |\n"
        table_string += f"| {' ':<{len(dataset_name)}} | {' ':<{len(str(nodes))}} | {' ':<{len(str(edges))}} | RX Louvain | {time_fmt[1]} | {mem_fmt[1]} | {comms_fmt[1]} | {mod_fmt[1]} | {cond_fmt[1]} | {dens_fmt[1]} | {ari_fmt[1]} | {nmi_fmt[1]} | {vm_fmt[1]} |\n"
        table_string += f"| {' ':<{len(dataset_name)}} | {' ':<{len(str(nodes))}} | {' ':<{len(str(edges))}} | RX Leiden  | {time_fmt[2]} | {mem_fmt[2]} | {comms_fmt[2]} | {mod_fmt[2]} | {cond_fmt[2]} | {dens_fmt[2]} | {ari_fmt[2]} | {nmi_fmt[2]} | {vm_fmt[2]} |\n" # Changed CPM to Leiden

    return table_string

def generate_results_table_matplotlib(results, output_file="benchmark_results_table.png"):
    """
    Generates a table summarizing benchmark results as a PNG image using Matplotlib.
    Includes NetworkX Louvain, RustWorkX Louvain, and RustWorkX Leiden.
    Highlights the best result for each metric.
    """
    if not results:
        print("No benchmark results to generate table image.")
        return

    plt.switch_backend("Agg")

    # --- Data Preparation ---
    col_labels = ["Dataset", "Nodes", "Edges", "Algorithm", "Time", "Memory", "# Comms", 
                  "Modularity", "Conductance", "Int. Density", "ARI", "NMI", "V-Measure"]
    cell_text = []
    row_colors = []
    base_color = "#FFFFFF" # White
    alt_color = "#F0F0F0"  # Light grey
    highlight_color = "#FFFACD" # Lemon chiffon for best value

    # --- Helper function to format values and get highlight info ---
    def fmt(key, decimals=3, lower_is_better=False, is_time=False, is_memory=False):
        nx_val = result.get(f"nx_{key}", float("nan"))
        rx_l_val = result.get(f"rx_louvain_{key}", float("nan"))
        rx_p_val = result.get(f"rx_leiden_{key}", float("nan")) # Changed CPM to Leiden
        
        values = [nx_val, rx_l_val, rx_p_val]
        best_idx = get_best_idx(values, lower_is_better) # Use the same helper as markdown table
        
        formatted_values = []
        highlight_flags = []
        for i, v in enumerate(values):
            is_best = (i == best_idx)
            if isinstance(v, float) and np.isnan(v):
                formatted_values.append("-")
            elif is_time:
                formatted_values.append(format_time(v))
            elif is_memory:
                 formatted_values.append(format_memory(v))
            elif isinstance(v, int):
                 formatted_values.append(f"{v}")
            else:
                formatted_values.append(f"{v:.{decimals}f}")
            highlight_flags.append(is_best)
        return formatted_values, highlight_flags

    # Define get_best_idx locally for matplotlib table generation
    def get_best_idx(values, lower_is_better=False):
        valid_values = [(i, v) for i, v in enumerate(values) if not (isinstance(v, float) and np.isnan(v))]
        if not valid_values:
            return -1
        if lower_is_better:
            best_idx, _ = min(valid_values, key=lambda item: item[1])
        else:
            best_idx, _ = max(valid_values, key=lambda item: item[1])
        return best_idx

    # --- Populate cell text and colors ---
    row_idx = 0
    for result in results:
        dataset_name = result["dataset"]
        nodes = result["nodes"]
        edges = result["edges"]
        
        # Format metrics for the three algorithms
        time_fmt, time_hl = fmt("elapsed", is_time=True, lower_is_better=True)
        mem_fmt, mem_hl = fmt("memory", is_memory=True, lower_is_better=True)
        comms_fmt, comms_hl = fmt("num_comms", decimals=0)
        mod_fmt, mod_hl = fmt("modularity")
        cond_fmt, cond_hl = fmt("conductance", lower_is_better=True)
        dens_fmt, dens_hl = fmt("internal_density")
        ari_fmt, ari_hl = fmt("ari")
        nmi_fmt, nmi_hl = fmt("nmi")
        vm_fmt, vm_hl = fmt("v_measure")

        # Combine formatted strings and highlight flags for each row
        nx_row = [dataset_name, nodes, edges, "NX Louvain"] + [time_fmt[0], mem_fmt[0], comms_fmt[0], mod_fmt[0], cond_fmt[0], dens_fmt[0], ari_fmt[0], nmi_fmt[0], vm_fmt[0]]
        rx_l_row = ["", "", "", "RX Louvain"] + [time_fmt[1], mem_fmt[1], comms_fmt[1], mod_fmt[1], cond_fmt[1], dens_fmt[1], ari_fmt[1], nmi_fmt[1], vm_fmt[1]]
        rx_p_row = ["", "", "", "RX Leiden"] + [time_fmt[2], mem_fmt[2], comms_fmt[2], mod_fmt[2], cond_fmt[2], dens_fmt[2], ari_fmt[2], nmi_fmt[2], vm_fmt[2]] # Changed CPM to Leiden
        
        nx_hl_flags = [False]*4 + [time_hl[0], mem_hl[0], comms_hl[0], mod_hl[0], cond_hl[0], dens_hl[0], ari_hl[0], nmi_hl[0], vm_hl[0]]
        rx_l_hl_flags = [False]*4 + [time_hl[1], mem_hl[1], comms_hl[1], mod_hl[1], cond_hl[1], dens_hl[1], ari_hl[1], nmi_hl[1], vm_hl[1]]
        rx_p_hl_flags = [False]*4 + [time_hl[2], mem_hl[2], comms_hl[2], mod_hl[2], cond_hl[2], dens_hl[2], ari_hl[2], nmi_hl[2], vm_hl[2]] # Changed CPM to Leiden
        
        cell_text.extend([nx_row, rx_l_row, rx_p_row])
        row_idx += 3
        
    # Correct way to set cell colors based on highlight flags
    cell_colours = []
    for i, result in enumerate(results):
        time_fmt, time_hl = fmt("elapsed", is_time=True, lower_is_better=True)
        mem_fmt, mem_hl = fmt("memory", is_memory=True, lower_is_better=True)
        comms_fmt, comms_hl = fmt("num_comms", decimals=0)
        mod_fmt, mod_hl = fmt("modularity")
        cond_fmt, cond_hl = fmt("conductance", lower_is_better=True)
        dens_fmt, dens_hl = fmt("internal_density")
        ari_fmt, ari_hl = fmt("ari")
        nmi_fmt, nmi_hl = fmt("nmi")
        vm_fmt, vm_hl = fmt("v_measure")
        
        nx_hl_flags = [False]*4 + [time_hl[0], mem_hl[0], comms_hl[0], mod_hl[0], cond_hl[0], dens_hl[0], ari_hl[0], nmi_hl[0], vm_hl[0]]
        rx_l_hl_flags = [False]*4 + [time_hl[1], mem_hl[1], comms_hl[1], mod_hl[1], cond_hl[1], dens_hl[1], ari_hl[1], nmi_hl[1], vm_hl[1]]
        rx_p_hl_flags = [False]*4 + [time_hl[2], mem_hl[2], comms_hl[2], mod_hl[2], cond_hl[2], dens_hl[2], ari_hl[2], nmi_hl[2], vm_hl[2]]
        
        row_base = base_color if (i % 2 == 0) else alt_color
        row_alt = alt_color if (i % 2 == 0) else base_color
        
        cell_colours.append([row_base if not hl else highlight_color for hl in nx_hl_flags])
        cell_colours.append([row_alt if not hl else highlight_color for hl in rx_l_hl_flags])
        cell_colours.append([row_base if not hl else highlight_color for hl in rx_p_hl_flags])
        
    # --- Create Table --- 
    fig, ax = plt.subplots(figsize=(20, max(8, len(cell_text) * 0.4))) 
    ax.axis("tight")
    ax.axis("off")
    the_table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellColours=cell_colours)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1.2, 1.2)

    # Style header
    for (i, j), cell in the_table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#444444")
        # Span dataset name over 3 rows
        if j == 0 and i > 0 and i % 3 == 1:
            # cell._loc = 'center' # Center text vertically
            cell.set_text_props(va="center") # Use vertical alignment property
        elif j == 0 and i > 0 and (i % 3 == 2 or i % 3 == 0):
             # cell.visible = False
             cell.set_visible(False)
             
    # Merge cells for Dataset name
    # Note: This requires careful indexing based on the 3 rows per dataset
    for i in range(0, len(cell_text), 3):
        # Merge Dataset cell
        if i + 2 < len(cell_text):
             # the_table[(i+1, 0)]._visible = True
             # the_table[(i+2, 0)]._visible = False
             # the_table[(i+3, 0)]._visible = False
             the_table[(i+1, 0)].set_visible(True)
             the_table[(i+2, 0)].set_visible(False)
             the_table[(i+3, 0)].set_visible(False)
             # the_table[(i+1, 0)]._text = cell_text[i][0]
             the_table[(i+1, 0)].get_text().set_text(cell_text[i][0])
             the_table[(i+1, 0)].set_height(3 * the_table[(i+1, 0)].get_height())
             # Merge Nodes cell
             # the_table[(i+1, 1)]._visible = True
             # the_table[(i+2, 1)]._visible = False
             # the_table[(i+3, 1)]._visible = False
             the_table[(i+1, 1)].set_visible(True)
             the_table[(i+2, 1)].set_visible(False)
             the_table[(i+3, 1)].set_visible(False)
             # the_table[(i+1, 1)]._text = str(cell_text[i][1])
             the_table[(i+1, 1)].get_text().set_text(str(cell_text[i][1]))
             the_table[(i+1, 1)].set_height(3 * the_table[(i+1, 1)].get_height())
             # Merge Edges cell
             # the_table[(i+1, 2)]._visible = True
             # the_table[(i+2, 2)]._visible = False
             # the_table[(i+3, 2)]._visible = False
             the_table[(i+1, 2)].set_visible(True)
             the_table[(i+2, 2)].set_visible(False)
             the_table[(i+3, 2)].set_visible(False)
             # the_table[(i+1, 2)]._text = str(cell_text[i][2])
             the_table[(i+1, 2)].get_text().set_text(str(cell_text[i][2]))
             the_table[(i+1, 2)].set_height(3 * the_table[(i+1, 2)].get_height())

    # --- Save Figure ---
    plt.title("Community Detection Benchmark Results", fontsize=16, fontweight="bold")
    try:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        print(f"Results table image saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving results table image: {e}")
        plt.close(fig) 