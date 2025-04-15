#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Benchmark script comparing rustworkx and networkx Louvain community detection
on various graph datasets with ground truth communities.
"""

import os
import random
import time
import traceback
from datetime import datetime

# cdlib imports
import cdlib
import cdlib.algorithms  # Added for Leiden/Infomap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx

# Import functions from the new modules
from benchmark_utils import (
    calculate_internal_metrics,
    compare_with_true_labels,
    convert_nx_to_rx,
    format_memory,
    format_time,
    measure_memory,
    rx_modularity_calculation,
)
from dataset_loaders import (
    load_citeseer,
    load_davis_women,
    load_dolphins,
    load_email_eu_core,
    load_facebook,
    load_florentine_families,
    load_football,
    load_graph_edges_9m_parquet,
    load_graph_edges_csv,
    load_graph_edges_parquet,
    load_karate_club,
    load_large_synthetic,
    load_les_miserables,
    load_lfr,
    load_polblogs,
    # load_livejournal, # Uncomment if needed
    # load_orkut, # Uncomment if needed
    load_political_books,
)
from plotting import (
    create_comparison_chart,
    generate_results_table,
    generate_results_table_matplotlib,
)

# Verify available rustworkx attributes and fix imports if needed
try:
    # Check for PyGraph
    if not hasattr(rx, "PyGraph"):
        print("Warning: rx.PyGraph not found, checking for alternative imports...")
        rx_attrs = dir(rx)
        print(f"Available rustworkx attributes: {', '.join(rx_attrs)}")
except Exception as e:
    print(f"Error checking rustworkx attributes: {e}")

# Импорт spatial биндингов
# from rustworkx import spatial # REMOVE THIS LINE


def run_benchmark():
    """Run benchmark comparison between rustworkx and networkx on all datasets."""
    print("=" * 80)
    print("RUSTWORKX vs NETWORKX COMMUNITY DETECTION BENCHMARK")
    print("=" * 80)
    print("\nThis benchmark will compare RustWorkX and NetworkX implementations of")
    print("the Louvain community detection algorithm on various datasets.")
    print("Results will be saved as high-quality image files.\n")
    
    # Setup result folder with timestamp to avoid overwriting
    base_result_folder = "results" # Base folder for all results
    timestamp = datetime.now().strftime("%Y%m%d") # Shorter timestamp (date only)
    result_folder = os.path.join(base_result_folder, timestamp) # Path like results/YYYYMMDD
    
    # Create base folder if it doesn't exist
    if not os.path.exists(base_result_folder):
        os.makedirs(base_result_folder)
    # Create timestamped subfolder
    os.makedirs(result_folder, exist_ok=True)
    
    print(f"Results will be saved to '{result_folder}'")
    
    # Dictionary mapping dataset names to loader functions (uses imported functions)
    DATASETS = {
        "Karate Club": load_karate_club,
        "Davis Southern Women": load_davis_women,
        "Florentine Families": load_florentine_families,
        "Les Misérables": load_les_miserables,
        "Football": load_football,
        "Political Books": load_political_books,
        "Dolphins": load_dolphins,
        "LFR Benchmark (Small)": lambda: load_lfr(n=500, mu=0.3, name="LFR Small"),
        "Political Blogs": load_polblogs,
        # "Cora": load_cora, # Still using the function from dataset_loaders
        "Facebook": load_facebook,
        "Citeseer": load_citeseer,
        "Email EU Core": load_email_eu_core,
        "Graph Edges CSV": load_graph_edges_csv,
        "Graph Edges Parquet": load_graph_edges_parquet,
        # "LFR Benchmark (Medium)": lambda: load_lfr(n=5000, mu=0.4, name="LFR Medium"), 
        # --- Large Datasets (uncomment selectively to run) ---
        # "Amazon Co-purchase": load_amazon_copurchase, # Requires loader function
        # "Orkut": load_orkut,
        # "LiveJournal": load_livejournal,
        "Large Synthetic (SBM)": load_large_synthetic,
        # "LFR Benchmark (Large)": lambda: load_lfr(n=100000, mu=0.5, name="LFR Large"),
        "Graph Edges 9M Parquet": load_graph_edges_9m_parquet, # Added new dataset
    }
    
    # Use non-interactive backend for all plots
    plt.switch_backend("Agg")
    
    benchmark_results = []
    for dataset_name, load_func in DATASETS.items():
        print(f"\n{'-'*40}")
        print(f"Processing {dataset_name} dataset...")
        print(f"{'-'*40}")
        try:
            result = run_benchmark_on_dataset(dataset_name, load_func, result_folder)
            # --- Add this print statement for debugging ---
            print(f"DEBUG: Result type for {dataset_name}: {type(result)}")
            # --------------------------------------------
            benchmark_results.append(result)
            print(f"Completed {dataset_name} dataset. Continuing to next dataset...")
        except Exception as e:
            print(f"❌ Error processing {dataset_name} dataset: {str(e)}")
            traceback.print_exc()
    
    # Generate final comparison chart (uses imported function)
    print("\n" + "=" * 80)
    print("GENERATING FINAL PERFORMANCE COMPARISON CHART")
    print("=" * 80)
    
    try:
        chart_path = os.path.join(result_folder, "community_detection_comparison.png")
        create_comparison_chart(benchmark_results, output_file=chart_path)
    except Exception as e:
        print(f"Error generating comparison chart: {str(e)}")
        traceback.print_exc()
    
    # Display list of generated files (Now includes the comparison chart)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - VISUALIZATION FILES GENERATED:")
    print("=" * 80)
    
    # List all PNG files in the result folder with their sizes
    png_files = sorted([f for f in os.listdir(result_folder) if f.endswith(".png")])
    
    if png_files:
        for i, png_file in enumerate(png_files):
            file_path = os.path.join(result_folder, png_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"{i+1}. {png_file} ({file_size:.2f} MB)")
            
        # Display instructions for viewing
        print("\nAll visualizations have been saved as PNG files.")
        print(f"You can find them in the '{result_folder}' directory.")
    else:
        print("No visualization files were generated. Please check for errors.")
        
    # Generate and print the results table (uses imported function)
    print("\n" + "=" * 80)
    print("GENERATING RESULTS TABLE")
    print("=" * 80)
    table_string = generate_results_table(benchmark_results)
    print(table_string)
    # Optionally save the table to a file
    table_filename = os.path.join(result_folder, "benchmark_results_table.md")
    try:
        with open(table_filename, "w") as f:
            f.write(table_string)
        print(f"\nResults table saved to: {table_filename}")
    except Exception as e:
        print(f"Error saving results table: {e}")

    # Generate the table as a PNG image using Matplotlib (uses imported function)
    print("\n" + "=" * 80)
    print("GENERATING RESULTS TABLE IMAGE")
    print("=" * 80)
    table_image_filename = os.path.join(result_folder, "benchmark_results_table.png")
    try:
        generate_results_table_matplotlib(benchmark_results, output_file=table_image_filename)
    except Exception as e:
        print(f"Error generating results table image: {e}")
        traceback.print_exc()

    print("\nBenchmark completed successfully!")

# ... existing code ...
# Threshold for considering a graph 'large' and skipping NetworkX
LARGE_GRAPH_THRESHOLD = 94000 

@measure_memory
def run_nx_algorithm(graph):
    """Run NetworkX Louvain algorithm, ensuring it returns a list of sets."""
    communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    # Ensure output is list of sets
    if not isinstance(communities, list) or not all(isinstance(c, set) for c in communities):
        return [set(c) for c in communities]
    return communities

@measure_memory
def run_rx_algorithm(graph):
    """Run RustWorkX Louvain algorithm, ensuring it returns a list of lists (node indices)."""
    weight_fn = lambda edge: float(edge)
    communities = rx.louvain_communities(graph, weight_fn=weight_fn, seed=42)
    return communities

@measure_memory
def run_cdlib_algorithm(graph, algorithm_name):
    """Run a cdlib algorithm (e.g., Leiden) and return communities."""
    if algorithm_name == "leiden":
        coms = cdlib.algorithms.leiden(graph, weights="weight")
    elif algorithm_name == "infomap":
        # Check if Infomap is installed and available
        try:
            coms = cdlib.algorithms.infomap(graph, flags="--seed 42")
        except Exception as e:
            print(f"Warning: Could not run Infomap (check installation?): {e}")
            return [] # Return empty list if Infomap fails
    else:
        raise ValueError(f"Unknown cdlib algorithm: {algorithm_name}")
        
    # Ensure output is list of lists (cdlib often uses NodeClustering)
    if hasattr(coms, "communities") and isinstance(coms.communities, list):
        return coms.communities
    elif isinstance(coms, list): # Sometimes it might return a raw list
        return coms
    else:
        print(f"Warning: Unexpected output type from cdlib {algorithm_name}: {type(coms)}")
        return [] # Return empty list if format is unexpected

@measure_memory
def run_rx_leiden_algorithm(graph):
    """Run RustWorkX Leiden algorithm, ensuring it returns a list of lists (node indices)."""
    try:
        communities = rx.leiden_communities(graph, _weight_fn=None, resolution=1.0, seed=42)
        return communities
    except Exception as e:
        print(f"Warning: Leiden algorithm failed: {e}")
        return []

def run_benchmark_on_dataset(dataset_name, load_func, result_folder):
    """Run benchmark comparison on a specific dataset."""
    # ... (setup code: print header, set seed) ...
    print(f"\n{'='*50}")
    print(f"Running Louvain benchmark on {dataset_name} dataset")
    print(f"{'='*50}")
    fixed_seed = 42 
    random.seed(fixed_seed)
    np.random.seed(fixed_seed)

    # Initialize results dictionary *before* the main try block
    # Use dummy values initially, they will be updated in the try block
    num_nodes = 0
    num_edges = 0
    has_ground_truth = False
    results = {
        "dataset": dataset_name,
        "nodes": num_nodes, # Will be updated after loading
        "edges": num_edges,   # Will be updated after loading
        "has_ground_truth": has_ground_truth, # Will be updated after loading
        # External Metrics (default NaN) - NX, RX(Louvain), RX(Leiden)
        "nx_ari": float("nan"), "nx_nmi": float("nan"), "nx_homogeneity": float("nan"),
        "nx_completeness": float("nan"), "nx_v_measure": float("nan"), "nx_fmi": float("nan"),
        "rx_louvain_ari": float("nan"), "rx_louvain_nmi": float("nan"), "rx_louvain_homogeneity": float("nan"),
        "rx_louvain_completeness": float("nan"), "rx_louvain_v_measure": float("nan"), "rx_louvain_fmi": float("nan"),
        # Добавляем метрики для Leiden
        "rx_leiden_ari": float("nan"), "rx_leiden_nmi": float("nan"), "rx_leiden_homogeneity": float("nan"),
        "rx_leiden_completeness": float("nan"), "rx_leiden_v_measure": float("nan"), "rx_leiden_fmi": float("nan"),
        # Internal Metrics (default NaN/0.0) - NX, RX(Louvain), RX(Leiden)
        "nx_modularity": 0.0, "nx_conductance": float("nan"), "nx_internal_density": float("nan"),
        "nx_avg_internal_degree": float("nan"), "nx_tpr": float("nan"), "nx_cut_ratio": float("nan"),
        "nx_surprise": float("nan"), "nx_significance": float("nan"),
        "rx_louvain_modularity": 0.0, "rx_louvain_conductance": float("nan"), "rx_louvain_internal_density": float("nan"),
        "rx_louvain_avg_internal_degree": float("nan"), "rx_louvain_tpr": float("nan"), "rx_louvain_cut_ratio": float("nan"),
        "rx_louvain_surprise": float("nan"), "rx_louvain_significance": float("nan"),
        # Добавляем метрики для Leiden
        "rx_leiden_modularity": 0.0, "rx_leiden_conductance": float("nan"), "rx_leiden_internal_density": float("nan"),
        "rx_leiden_avg_internal_degree": float("nan"), "rx_leiden_tpr": float("nan"), "rx_leiden_cut_ratio": float("nan"),
        "rx_leiden_surprise": float("nan"), "rx_leiden_significance": float("nan"),
        # Performance (default 0) - NX, RX(Louvain), RX(Leiden)
        "nx_elapsed": 0, "nx_memory": 0, "nx_num_comms": 0,
        "rx_louvain_elapsed": 0, "rx_louvain_memory": 0, "rx_louvain_num_comms": 0,
        "rx_leiden_elapsed": 0, "rx_leiden_memory": 0, "rx_leiden_num_comms": 0,
    }

    try:
        # Load data (uses imported loader)
        nx_graph, true_labels, has_ground_truth = load_func()
        num_nodes = len(nx_graph.nodes())
        num_edges = len(nx_graph.edges())
        # Update results with actual graph info
        results["nodes"] = num_nodes
        results["edges"] = num_edges
        results["has_ground_truth"] = has_ground_truth
        print(f"Loaded {dataset_name}: {num_nodes} nodes, {num_edges} edges. Ground truth available: {has_ground_truth}")

        # Convert graph (uses imported utility)
        rx_graph, node_map = convert_nx_to_rx(nx_graph)
        
        # Prepare the graph that will be used by cdlib (needs to be NetworkX)
        # Create an undirected copy for algorithms that require it (like Louvain)
        graph_for_nx_cdlib = nx.Graph(nx_graph)

        # Ensure all edges have a 'weight' attribute for compatibility
        default_weight = 1.0
        edges_without_weight = 0
        for u, v, data in graph_for_nx_cdlib.edges(data=True):
            if "weight" not in data or data["weight"] is None:
                data["weight"] = default_weight
                edges_without_weight += 1
        if edges_without_weight > 0:
            print(f"Info: Added default weight ({default_weight}) to {edges_without_weight} edges for cdlib compatibility.")

        # --- Run NetworkX Louvain (Baseline) ---
        nx_communities = []
        if num_nodes <= LARGE_GRAPH_THRESHOLD:
            try:
                print("\nRunning NetworkX Louvain...")
                nx_start = time.time()
                # Use the run_nx_algorithm wrapper
                nx_communities_raw, results["nx_memory"] = run_nx_algorithm(graph_for_nx_cdlib)
                # Ensure communities are list of sets for consistency
                nx_communities = [set(c) for c in nx_communities_raw]
                results["nx_elapsed"] = time.time() - nx_start
                results["nx_num_comms"] = len(nx_communities)
                print(f"NetworkX Louvain: {results['nx_num_comms']} communities in {format_time(results['nx_elapsed'])} using {format_memory(results['nx_memory'])}")

                if nx_communities: # Check if communities were found
                    try:
                        results["nx_modularity"] = nx.community.modularity(graph_for_nx_cdlib, nx_communities, weight="weight")
                    except Exception as mod_e:
                        print(f"Warning: NX modularity calculation failed: {mod_e}")

                    try:
                        (
                            results["nx_conductance"], results["nx_internal_density"],
                            results["nx_avg_internal_degree"], results["nx_tpr"], results["nx_cut_ratio"],
                            results["nx_surprise"], results["nx_significance"]
                        ) = calculate_internal_metrics(graph_for_nx_cdlib, nx_communities, rx_graph, node_map)
                    except Exception as int_met_e:
                        print(f"Warning: NX Louvain internal metric calculation failed: {int_met_e}")

                    if has_ground_truth:
                        try:
                            (results["nx_ari"], results["nx_nmi"], results["nx_homogeneity"],
                             results["nx_completeness"], results["nx_v_measure"], results["nx_fmi"]) = compare_with_true_labels(nx_communities, true_labels)
                        except Exception as ext_met_e:
                            print(f"Warning: NX external metric calculation failed: {ext_met_e}")

            except Exception as e:
                print(f"Error running NetworkX Louvain: {str(e)}")
                # Ensure performance metrics are 0 if it fails mid-run
                results["nx_elapsed"], results["nx_memory"], results["nx_num_comms"] = 0, 0, 0
        else:
            print(f"\nSkipping NetworkX Louvain for large graph ({num_nodes} nodes > {LARGE_GRAPH_THRESHOLD})")

        # --- Run RustWorkX Louvain (Baseline) ---
        rx_communities = []
        rx_communities_rx_indices = [] # Store communities with RX indices
        try:
            print("\nRunning RustWorkX Louvain...")
            rx_start = time.time()
            # Use the run_rx_algorithm wrapper
            rx_communities_raw, results["rx_louvain_memory"] = run_rx_algorithm(rx_graph)
            rx_communities_rx_indices = rx_communities_raw # Keep RX indices for modularity
            # Ensure communities are list of sets for consistency (using original node IDs)
            reverse_map = {v: k for k, v in node_map.items()}
            rx_communities = [set(reverse_map[node_idx] for node_idx in c) for c in rx_communities_raw]

            results["rx_louvain_elapsed"] = time.time() - rx_start
            results["rx_louvain_num_comms"] = len(rx_communities)
            print(f"RustWorkX Louvain: {results['rx_louvain_num_comms']} communities in {format_time(results['rx_louvain_elapsed'])} using {format_memory(results['rx_louvain_memory'])}")

            if rx_communities: # Check if communities were found
                # Calculate RX Modularity using RX indices
                try:
                    weight_fn_for_rx = lambda edge: float(edge)
                    # Pass communities with RX indices (rx_communities_rx_indices)
                    results["rx_louvain_modularity"] = rx_modularity_calculation(
                        rx_graph, rx_communities_rx_indices, weight_fn=weight_fn_for_rx
                    )
                except Exception as rx_mod_e:
                    print(f"Warning: RX Modularity calculation failed: {rx_mod_e}")

                # Correct indentation for internal metrics calculation (uses original node IDs)
                try:
                    (
                        results["rx_louvain_conductance"], results["rx_louvain_internal_density"],
                        results["rx_louvain_avg_internal_degree"], results["rx_louvain_tpr"], results["rx_louvain_cut_ratio"],
                        results["rx_louvain_surprise"], results["rx_louvain_significance"]
                    ) = calculate_internal_metrics(graph_for_nx_cdlib, rx_communities, rx_graph, node_map)
                except Exception as int_met_e:
                    print(f"Warning: RX Louvain internal metric calculation failed: {int_met_e}")

                # Correct indentation for external metrics calculation (uses original node IDs)
                if has_ground_truth:
                    try:
                        (results["rx_louvain_ari"], results["rx_louvain_nmi"], results["rx_louvain_homogeneity"],
                         results["rx_louvain_completeness"], results["rx_louvain_v_measure"], results["rx_louvain_fmi"]) = compare_with_true_labels(rx_communities, true_labels)
                    except Exception as ext_met_e:
                        print(f"Warning: RX external metric calculation failed: {ext_met_e}")

        except Exception as e:
            print(f"Error running RustWorkX Louvain: {str(e)}")
            # Ensure performance metrics are 0 if it fails mid-run
            results["rx_louvain_elapsed"], results["rx_louvain_memory"], results["rx_louvain_num_comms"] = 0, 0, 0

        # --- Run RustWorkX Leiden ---
        rx_leiden_communities = []
        rx_leiden_communities_rx_indices = [] # Leiden возвращает индексы RX
        try:
            print("\nRunning RustWorkX Leiden...")
            rx_leiden_start = time.time()
            rx_leiden_communities_raw, results["rx_leiden_memory"] = run_rx_leiden_algorithm(rx_graph)
            rx_leiden_communities_rx_indices = rx_leiden_communities_raw # Сохраняем для модулярности
            
            # Конвертируем индексы RX в оригинальные ID узлов
            reverse_map = {v: k for k, v in node_map.items()} # Используем ранее созданный node_map
            rx_leiden_communities = [set(reverse_map[node_idx] for node_idx in c) for c in rx_leiden_communities_raw]

            results["rx_leiden_elapsed"] = time.time() - rx_leiden_start
            results["rx_leiden_num_comms"] = len(rx_leiden_communities)
            print(f"RustWorkX Leiden: {results['rx_leiden_num_comms']} communities in {format_time(results['rx_leiden_elapsed'])} using {format_memory(results['rx_leiden_memory'])}")

            # --- Метрики для Leiden --- 
            if rx_leiden_communities:
                 # Модулярность (используем индексы RX)
                try:
                    weight_fn_for_rx = lambda edge: float(edge)
                    results["rx_leiden_modularity"] = rx_modularity_calculation(
                        rx_graph, rx_leiden_communities_rx_indices, weight_fn=weight_fn_for_rx
                    )
                except Exception as rx_mod_e:
                    print(f"Warning: RX Leiden Modularity calculation failed: {rx_mod_e}")
                    
                # Внутренние метрики (используем оригинальные ID)
                try:
                    (
                        results["rx_leiden_conductance"], results["rx_leiden_internal_density"],
                        results["rx_leiden_avg_internal_degree"], results["rx_leiden_tpr"], results["rx_leiden_cut_ratio"],
                        results["rx_leiden_surprise"], results["rx_leiden_significance"]
                    ) = calculate_internal_metrics(graph_for_nx_cdlib, rx_leiden_communities, rx_graph, node_map)
                except Exception as int_met_e:
                    print(f"Warning: RX Leiden internal metric calculation failed: {int_met_e}")

                # Внешние метрики (используем оригинальные ID)
                if has_ground_truth:
                    try:
                        (results["rx_leiden_ari"], results["rx_leiden_nmi"], results["rx_leiden_homogeneity"],
                         results["rx_leiden_completeness"], results["rx_leiden_v_measure"], results["rx_leiden_fmi"]) = compare_with_true_labels(rx_leiden_communities, true_labels)
                    except Exception as ext_met_e:
                        print(f"Warning: RX Leiden external metric calculation failed: {ext_met_e}")
        
        except Exception as e:
            print(f"Error running RustWorkX Leiden: {str(e)}")
            results["rx_leiden_elapsed"] = 0
            results["rx_leiden_memory"] = 0
            results["rx_leiden_num_comms"] = 0

        # --- Spatial metrics based on 2D layout --- 
        if rx_leiden_communities_rx_indices: # Only if Leiden communities were found
            print("\nCalculating Spatial Metrics based on 2D Random Layout...")
            try:
                # 1. Generate 2D layout (embeddings)
                print("  Generating random layout...")
                layout_start_time = time.time()
                # Ensure layout uses the RX graph and a seed for reproducibility
                layout = rx.random_layout(rx_graph, seed=fixed_seed)
                print(f"  Layout generated in {format_time(time.time() - layout_start_time)}")

                # 2. Prepare embeddings array in the correct order
                # Assuming node indices in rx_graph are contiguous from 0 to num_nodes - 1
                embeddings_np = np.zeros((num_nodes, 2), dtype=np.float32)
                valid_indices = 0
                for node_idx in rx_graph.node_indices():
                    if node_idx in layout:
                        embeddings_np[node_idx] = layout[node_idx]
                        valid_indices += 1
                    else:
                        # Handle cases where layout might miss a node (shouldn't happen often)
                        print(f"Warning: Node index {node_idx} not found in layout result.")
                
                if valid_indices != num_nodes:
                     print(f"Warning: Layout generated for {valid_indices}/{num_nodes} nodes.")

                # 3. Clusters are the RX Leiden results (rx_leiden_communities_rx_indices)
                # Ensure it's the list of lists of RX indices
                clusters_for_spatial = rx_leiden_communities_rx_indices

                # 4. Call the spatial metrics function - USE rx.spatial
                print("  Calculating cluster spatial metrics...")
                spatial_metrics_start_time = time.time()
                # Use rx.spatial directly
                spatial_metrics_list = rx.spatial.cluster_spatial_metrics(embeddings_np, clusters_for_spatial)
                print(f"  Spatial metrics calculated in {format_time(time.time() - spatial_metrics_start_time)}")

                # 5. Calculate and print aggregated spatial metrics
                if spatial_metrics_list:
                    total_nodes_in_clusters = sum(m.size for m in spatial_metrics_list)
                    if total_nodes_in_clusters > 0:
                        weighted_avg_radius = sum(m.avg_radius * m.size for m in spatial_metrics_list) / total_nodes_in_clusters
                        weighted_avg_std_radius = sum(m.std_radius * m.size for m in spatial_metrics_list) / total_nodes_in_clusters
                        # Max radius aggregation might be less intuitive, perhaps report the overall max?
                        overall_max_radius = max(m.max_radius for m in spatial_metrics_list if m.size > 0) if any(m.size > 0 for m in spatial_metrics_list) else 0.0
                        weighted_mean_pairwise = sum(m.mean_pairwise * m.size for m in spatial_metrics_list) / total_nodes_in_clusters

                        print(f"\n--- Aggregated Spatial Metrics for {dataset_name} (Leiden clusters, 2D Layout) ---")
                        print(f"  Weighted Avg Radius: {weighted_avg_radius:.4f}")
                        print(f"  Weighted Avg Std Radius: {weighted_avg_std_radius:.4f}")
                        print(f"  Overall Max Radius: {overall_max_radius:.4f}")
                        print(f"  Weighted Mean Pairwise Dist: {weighted_mean_pairwise:.4f}")
                        print("------------------------------------------------------------------------------")
                    else:
                        print("\nSkipping aggregated spatial metrics: No nodes found in clusters.")
                else:
                    print("\nSkipping aggregated spatial metrics: Spatial metrics list is empty.")

            except AttributeError as ae:
                 # Catch potential issues if rx.spatial functions aren't available yet
                 if "rustworkx" in str(ae) or "spatial" in str(ae):
                     print(f"  Skipping spatial metrics: rustworkx.spatial function not found (likely needs compilation or version update): {ae}") # Updated message
                 else:
                     print(f"  Error calculating spatial metrics: {ae}")
                     traceback.print_exc()
            except Exception as e:
                print(f"  Error calculating spatial metrics: {e}")
                traceback.print_exc()
        else:
            print("\nSkipping spatial metrics calculation as Leiden communities were not found.")

        # --- t-SNE/UMAP visualization removed ---

    except Exception as e:
        print(f"❌ Critical Error processing {dataset_name}: {str(e)}")
        traceback.print_exc() 
        # Ensure results dict is returned even on critical error, possibly with defaults
        # Make sure 'nodes' and 'edges' might be 0 if loading failed early
        results["nodes"] = results.get("nodes", 0) 
        results["edges"] = results.get("edges", 0)
        return results 

    print(f"\nFinished benchmark for {dataset_name}.")
    return results

def main():
    """
    Main function to run benchmark for different community detection algorithms.
    """
    # Just call the run_benchmark function which handles everything
    run_benchmark()


if __name__ == "__main__":
    main()
