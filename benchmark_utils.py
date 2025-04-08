import math
import tracemalloc

import cdlib
import numpy as np
import rustworkx as rx
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

# --- Helper Function for Parallel ASPL Calculation ---
# Must be defined at top level for multiprocessing pickling

def _calculate_subgraph_aspl(community_indices_and_main_rx_graph):
    """
    Calculates the average shortest path length for a single community subgraph using rustworkx.
    Attempts to use graph_unweighted_average_shortest_path_length.
    Accepts rustworkx node indices and the main rustworkx graph.
    Handles disconnected subgraphs by calculating ASPL on the largest connected component.
    Explicitly returns NaN for components with >=2 nodes but 0 edges.
    Designed to be used with multiprocessing.map.

    Args:
        community_indices_and_main_rx_graph (tuple): A tuple containing:
            - community_indices (list): List of rustworkx node indices in the community.
            - main_rx_graph (rx.PyGraph): The main rustworkx graph.

    Returns:
        float: The average shortest path length of the largest connected component,
               or np.nan if calculation fails, LCC < 2 nodes, or LCC has 0 edges.
    """
    community_indices, main_rx_graph = community_indices_and_main_rx_graph
    target_subgraph = None # This will hold the RX subgraph we actually calculate on
    try:
        if not community_indices or len(community_indices) < 2:
            return np.nan

        # Create RustWorkX subgraph
        rx_subgraph = main_rx_graph.subgraph(community_indices)
        num_nodes = rx_subgraph.num_nodes()

        if num_nodes < 2:
            return np.nan

        # --- Determine the subgraph to calculate ASPL on (Connected or LCC) using RustWorkX --- 
        if rx.is_connected(rx_subgraph):
            target_subgraph = rx_subgraph
        else:
            components = list(rx.connected_components(rx_subgraph))
            if not components:
                 return np.nan
            largest_cc_nodes_indices = max(components, key=len) # Indices relative to rx_subgraph
            if len(largest_cc_nodes_indices) < 2:
                return np.nan
            # Create LCC subgraph from the main community subgraph
            target_subgraph = rx_subgraph.subgraph(largest_cc_nodes_indices)

        # --- Check target subgraph validity --- 
        if target_subgraph is None:
             return np.nan 

        num_target_nodes = target_subgraph.num_nodes()
        num_target_edges = target_subgraph.num_edges()

        if num_target_nodes < 2:
             return np.nan

        # Handle case with >= 2 nodes but 0 edges (ASPL is infinite/undefined)
        if num_target_edges == 0:
            return np.nan

        # --- Calculate ASPL using rustworkx.graph_unweighted_average_shortest_path_length --- 
        # This function is specifically for PyGraph and unweighted cases
        aspl = rx.graph_unweighted_average_shortest_path_length(target_subgraph)
        return aspl
            
    except rx.NoPathFound as npf_err:
        # Catch specific error if path not found within the target component
        target_info = f"Target Nodes: {target_subgraph.num_nodes() if target_subgraph else 'N/A'}, Edges: {target_subgraph.num_edges() if target_subgraph else 'N/A'}"
        print(f"RX ASPL Warning (NoPathFound) ({target_info}): {npf_err!r}")
        return np.nan
    except Exception as e:
        # Catch any other unexpected error 
        initial_node_count = len(community_indices) if community_indices else 0
        target_info = f"Target Nodes: {target_subgraph.num_nodes() if target_subgraph else 'N/A'}, Edges: {target_subgraph.num_edges() if target_subgraph else 'N/A'}"
        print(f"RX ASPL Error (Initial Nodes: {initial_node_count}, {target_info}): {e!r}") 
        return np.nan
# --- End Helper Function ---


def convert_nx_to_rx(nx_graph):
    """Convert NetworkX graph to RustWorkX graph, ensuring positive similarity weights."""
    node_map = {}
    # Fix: Use rustworkx.PyGraph directly if available, handle potential AttributeError
    try:
        # Linter might flag PyGraph if using an older rustworkx version
        rx_graph = rx.PyGraph()
    except AttributeError:
        print("Warning: rx.PyGraph not found. Check rustworkx installation/version.")
        # Fallback or raise error depending on desired behavior
        raise AttributeError("rx.PyGraph not found in rustworkx module.")

    for node in nx_graph.nodes():
        rx_idx = rx_graph.add_node(node)
        node_map[node] = rx_idx

    for u, v, data in nx_graph.edges(data=True):
        weight = 1.0 # Default weight
        if "weight" in data:
             weight = data["weight"]
        elif "distance" in data:
             # Convert distance to similarity if 'weight' isn't present
             distance = float(data.get("distance", 2.0))
             similarity = max(1e-9, 1.0 - distance / 2.0)
             weight = similarity
             # print(f"Converted distance {distance} to similarity {similarity}") # Optional debug print
        
        # Ensure weight is positive for algorithms like Louvain/Leiden/Infomap
        final_weight = float(max(weight, 1e-9)) 
        rx_graph.add_edge(node_map[u], node_map[v], final_weight)

    return rx_graph, node_map


def compare_with_true_labels(communities, true_labels, node_map=None):
    """
    Compare detected communities with ground truth.
    
    Returns adjusted Rand index (ARI), normalized mutual information (NMI),
    homogeneity, completeness, V-measure, and Fowlkes-Mallows Index (FMI).
    """
    # === Handle empty communities list ===
    if not communities:
        print("Warning: compare_with_true_labels received empty communities list.")
        return tuple([float("nan")] * 6) # Return NaNs for ARI, NMI, Hom, Comp, V, FMI
    # === End handle empty ===

    pred_labels = np.zeros(len(true_labels), dtype=int)
    if node_map is not None:
        reverse_map = {v: k for k, v in node_map.items()}
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                if node in reverse_map:
                    orig_node = reverse_map[node]
                    if isinstance(orig_node, int) and 0 <= orig_node < len(true_labels):
                        pred_labels[orig_node] = comm_idx
    else:
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                if isinstance(node, int) and 0 <= node < len(true_labels):
                    pred_labels[node] = comm_idx
    
    # Calculate metrics, handle potential errors if only one cluster is predicted
    try:
        ari = adjusted_rand_score(true_labels, pred_labels)
    except ValueError:
        ari = 0.0 # Or handle as appropriate, e.g., if true_labels has only one class
    try:
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    except ValueError:
        nmi = 0.0
    try:
        homogeneity = homogeneity_score(true_labels, pred_labels)
    except ValueError:
        homogeneity = 0.0
    try:
        completeness = completeness_score(true_labels, pred_labels)
    except ValueError:
        completeness = 0.0
    try:
        v_measure = v_measure_score(true_labels, pred_labels)
    except ValueError:
        v_measure = 0.0
    try:
        fmi = fowlkes_mallows_score(true_labels, pred_labels)
    except ValueError:
        fmi = 0.0

    return ari, nmi, homogeneity, completeness, v_measure, fmi


def measure_memory(func):
    """Decorator to measure memory usage of a function."""
    def wrapper(*args, **kwargs):
        # Start tracking memory
        tracemalloc.start()
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Measure peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        
        # Stop tracking memory
        tracemalloc.stop()
        
        # Return both the result and the peak memory usage (in MB)
        return result, peak / (1024 * 1024)  # Convert to MB
    
    return wrapper


def format_memory(memory_mb):
    """
    Format memory usage with adaptive units (KB, MB, GB) with max 3 significant digits.
    
    Args:
        memory_mb: Memory in MB.
        
    Returns:
        Formatted memory string with appropriate unit.
    """
    if memory_mb < 0.1:
        # For values less than 0.1 MB, show in KB
        memory_kb = memory_mb * 1024
        value = round(memory_kb, 3 - int(math.floor(math.log10(abs(memory_kb)))) - 1) if memory_kb >= 1 else memory_kb
        return f"{value:.3g} KB"
    elif memory_mb < 1000:
        # For values between 0.1 MB and 1000 MB, show in MB
        value = round(memory_mb, 3 - int(math.floor(math.log10(abs(memory_mb)))) - 1) if memory_mb >= 1 else memory_mb
        return f"{value:.3g} MB"
    else:
        # For values >= 1000 MB, show in GB
        memory_gb = memory_mb / 1024
        value = round(memory_gb, 3 - int(math.floor(math.log10(abs(memory_gb)))) - 1) if memory_gb >= 1 else memory_gb
        return f"{value:.3g} GB"


def format_time(seconds):
    """
    Format time with adaptive units (μs, ms, s) with max 3 significant digits.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string with appropriate unit.
    """
    if seconds == 0:
        return "0 s"
    
    # Convert to microseconds
    microseconds = seconds * 1000000
    
    if microseconds < 1000:
        # For values less than 1000 μs, show in μs
        value = round(microseconds, 3 - int(math.floor(math.log10(abs(microseconds)))) - 1) if microseconds >= 1 else microseconds
        return f"{value:.3g} μs"
    elif microseconds < 1000000:
        # For values between 1000 μs and 1000000 μs, show in ms
        milliseconds = microseconds / 1000
        value = round(milliseconds, 3 - int(math.floor(math.log10(abs(milliseconds)))) - 1) if milliseconds >= 1 else milliseconds
        return f"{value:.3g} ms"
    else:
        # For values >= 1000000 μs (1 second), show in s
        value = round(seconds, 3 - int(math.floor(math.log10(abs(seconds)))) - 1) if seconds >= 1 else seconds
        return f"{value:.3g} s"


def rx_modularity_calculation(rx_graph, communities, weight_fn):
    """Calculates modularity using rustworkx.community.modularity."""
    try:
        # Use the correct path via the community submodule
        return rx.community.modularity(rx_graph, communities, weight_fn=weight_fn)
    except AttributeError:
        print("Warning: Modularity function not found in rustworkx.community.")
        return float("nan")
    except Exception as e:
        print(f"Warning: Error calculating rustworkx modularity: {e}")
        return float("nan")


def calculate_internal_metrics(nx_graph, communities, main_rx_graph, node_map):
    """Calculate various internal community quality metrics, removing ASPL calculation."""
    if not communities: # Handle empty community list
        print("Warning: Received empty community list for internal metrics.")
        # Return NaNs for the 7 metrics
        return (float("nan"),) * 7
        
    # Filter communities for size >= 2 (can be done later if needed)
    # Keep original nodes for cdlib compatibility
    formatted_communities = [list(c) for c in communities if c] # Ensure no empty communities
    
    if not formatted_communities:
        print("Warning: No valid communities found for internal metrics.")
        return (float("nan"),) * 7

    # --- Prepare for cdlib metrics (using original nx_graph) ---
    base_clustering = None
    try:
        node_list = list(nx_graph.nodes())
        node_to_int = {node: i for i, node in enumerate(node_list)}
        # Map original node IDs to integer indices for cdlib
        int_communities = [[node_to_int[node] for node in comm if node in node_to_int] for comm in formatted_communities]
        int_communities = [c for c in int_communities if c] # Remove any empty lists after mapping
        if int_communities:
            base_clustering = cdlib.NodeClustering(communities=int_communities, graph=nx_graph, method_name="generic")
        else:
             print("Warning: No valid communities left after mapping nodes to integers for cdlib.")
    except KeyError as e:
        print(f"Error mapping node ID during internal metric prep: {e}. Skipping cdlib metrics.")
    except Exception as e:
         print(f"Unexpected error during node mapping for internal metrics: {e}")

    # Initialize metrics to NaN
    conductance = float("nan")
    internal_density = float("nan")
    avg_internal_degree = float("nan")
    tpr = float("nan") 
    cut_ratio = float("nan")
    surprise = float("nan")
    significance = float("nan")
    # avg_internal_path = float('nan') # Removed ASPL

    # Calculate cdlib-based metrics safely ONLY if base_clustering is valid
    if base_clustering:
        try: conductance = base_clustering.conductance().score
        except Exception as e: print(f"Warning: Conductance calculation failed: {e}")
        try: internal_density = base_clustering.internal_edge_density().score
        except Exception as e: print(f"Warning: Internal Density calculation failed: {e}")
        try: avg_internal_degree = base_clustering.average_internal_degree().score
        except Exception as e: print(f"Warning: Avg Internal Degree calculation failed: {e}")
        try:
            tpr = base_clustering.triangle_participation_ratio().score
        except ZeroDivisionError:
            print("Warning: TPR calculation failed due to division by zero (no nodes with degree >= 2). Assigning 0.0")
            tpr = 0.0
        except Exception as tpr_e: print(f"Warning: TPR calculation failed: {tpr_e}")
        try: cut_ratio = base_clustering.cut_ratio().score
        except ZeroDivisionError:
            print("Warning: Cut Ratio calculation failed due to division by zero. Assigning NaN")
            cut_ratio = float("nan")
        except Exception as e: print(f"Warning: Cut Ratio calculation failed: {e}")
        try: surprise = base_clustering.surprise().score
        except Exception as e: print(f"Warning: Surprise calculation failed: {e}")
        try: significance = base_clustering.significance().score
        except Exception as e: print(f"Warning: Significance calculation failed: {e}")
    else:
        print("Warning: Skipping cdlib internal metric calculations due to setup issues.")
        
    # --- ASPL Calculation Removed --- 
    # The parallel ASPL calculation block using _calculate_subgraph_aspl is removed.
    # print("  ASPL calculation removed from internal metrics.")

    # Return only the 7 metrics
    return (
        conductance, internal_density, avg_internal_degree, tpr, cut_ratio,
        surprise, significance
    ) 