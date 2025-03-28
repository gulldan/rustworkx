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

import math
import random
import time
import tracemalloc

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Verify available rustworkx attributes and fix imports if needed
try:
    # Check for PyGraph
    if not hasattr(rx, "PyGraph"):
        print("Warning: rx.PyGraph not found, checking for alternative imports...")
        rx_attrs = dir(rx)
        print(f"Available rustworkx attributes: {', '.join(rx_attrs)}")
except Exception as e:
    print(f"Error checking rustworkx attributes: {e}")


def load_karate_club():
    """Load Zachary's Karate Club dataset with ground truth communities."""
    G = nx.karate_club_graph()
    true_labels = [1 if G.nodes[node]["club"] == "Mr. Hi" else 0 for node in G.nodes()]
    return G, true_labels


def load_les_miserables():
    """Load Les Misérables dataset with ground truth communities."""
    G = nx.les_miserables_graph()
    
    # Communities are based on character groups
    old_groups = {
        "Myriel": 0, "Napoleon": 0, "MlleBaptistine": 0, "MmeMagloire": 0, "CountessDeLo": 0,
        "Geborand": 0, "Champtercier": 0, "Cravatte": 0, "Count": 0, "OldMan": 0,
        "Valjean": 1, "Labarre": 1, "Marguerite": 1, "MmeDeR": 1, "Isabeau": 1,
        "Gervais": 1, "Tholomyes": 2, "Listolier": 2, "Fameuil": 2, "Blacheville": 2,
        "Favourite": 2, "Dahlia": 2, "Zephine": 2, "Fantine": 2, "MmeThenardier": 3,
        "Thenardier": 3, "Cosette": 3, "Javert": 4, "Fauchelevent": 4, "Bamatabois": 4,
        "Perpetue": 4, "Simplice": 4, "Scaufflaire": 4, "Woman1": 4, "Judge": 4,
        "Champmathieu": 4, "Brevet": 4, "Chenildieu": 4, "Cochepaille": 4, "Pontmercy": 5,
        "Boulatruelle": 5, "Eponine": 5, "Anzelma": 5, "Woman2": 5, "MotherInnocent": 6,
        "Gribier": 6, "Jondrette": 7, "MmeBurgon": 7, "Gavroche": 7, "Gillenormand": 8,
        "Magnon": 8, "MlleGillenormand": 8, "MmePontmercy": 8, "MlleVaubois": 8,
        "LtGillenormand": 8, "Marius": 8, "BaronessT": 8, "Mabeuf": 9, "Enjolras": 9,
        "Combeferre": 9, "Prouvaire": 9, "Feuilly": 9, "Courfeyrac": 9, "Bahorel": 9,
        "Bossuet": 9, "Joly": 9, "Grantaire": 9, "MotherPlutarch": 9, "Gueulemer": 10,
        "Babet": 10, "Claquesous": 10, "Montparnasse": 10, "Toussaint": 11, "Child1": 11,
        "Child2": 11, "Brujon": 11, "MmeHucheloup": 11
    }
    
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50, seed=42)
    H = nx.Graph()
    mapping = {old: i for i, old in enumerate(G.nodes())}
    for old_node, new_node in mapping.items():
        H.add_node(new_node, pos=pos[old_node], value=old_groups[old_node])
    for u, v in G.edges():
        H.add_edge(mapping[u], mapping[v])
    true_labels = [H.nodes[node]["value"] for node in H.nodes()]
    return H, true_labels


def load_football():
    """Load American College Football dataset with ground truth communities."""
    # Assumes the file 'datasets/football.gml' exists.
    G = nx.read_gml("datasets/football.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels


def load_political_books():
    """Load Political Books dataset with ground truth communities."""
    # Assumes the file 'datasets/polbooks.gml' exists.
    G = nx.read_gml("datasets/polbooks.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels


def load_dolphins():
    """Load Dolphins Social Network dataset with ground truth communities.
    
    Assumes file 'datasets/dolphins.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/dolphins.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels


def load_polblogs():
    """Load Political Blogs dataset with ground truth communities.
    
    Assumes file 'datasets/polblogs.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/polblogs.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels


def load_cora():
    """Load Cora citation network with ground truth communities.
    
    Assumes file 'datasets/cora.gml' exists and that node attribute 'value'
    contains the class label.
    """
    G = nx.read_gml("datasets/cora.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels


def load_facebook():
    """Load Facebook Ego Networks / Social Circles dataset with ground truth communities.
    
    Assumes file 'datasets/facebook.gml' exists and that node attribute 'value'
    contains the community label.
    """
    G = nx.read_gml("datasets/facebook.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels


def load_lfr():
    """Generate a synthetic LFR benchmark graph with ground truth communities."""
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    # Generate the LFR benchmark graph (node attribute "community" is a frozenset)
    G = nx.LFR_benchmark_graph(n, tau1, tau2, mu,
                              average_degree=5, min_community=20, seed=42)
    # For ground truth, assign each node the minimum element of its community set as label
    true_labels = [min(G.nodes[node]["community"]) for node in G.nodes()]
    return G, true_labels


def load_large_synthetic():
    """Generate a large synthetic graph with many communities."""
    print("Generating large synthetic graph...")
    # Вместо LFR_benchmark_graph используем stochastic_block_model для создания большого графа
    n = 1500  # общее количество узлов
    # Создаем 15 сообществ с разными размерами
    sizes = [50, 80, 100, 120, 70, 90, 110, 60, 150, 130, 100, 80, 120, 140, 100]
    # Вероятности соединения внутри и между сообществами
    p_in = 0.1  # высокая вероятность соединения внутри сообщества
    p_out = 0.001  # низкая вероятность соединения между сообществами
    
    # Создаем матрицу вероятностей
    probs = np.ones((len(sizes), len(sizes))) * p_out
    np.fill_diagonal(probs, p_in)
    
    # Генерируем граф
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    
    # Создаем метки сообществ
    true_labels = []
    for i, size in enumerate(sizes):
        true_labels.extend([i] * size)
    
    print(f"Generated graph with {len(G.nodes())} nodes, {len(G.edges())} edges, and {len(sizes)} communities")
    return G, true_labels


def convert_nx_to_rx(nx_graph):
    """Convert NetworkX graph to RustWorkX graph, preserving weights."""
    node_map = {}
    rx_graph = rx.PyGraph()

    for node in nx_graph.nodes():
        rx_idx = rx_graph.add_node(node)
        node_map[node] = rx_idx

    # Add edges with weights from the NetworkX graph
    for u, v, data in nx_graph.edges(data=True):
        # Use data.get('weight', 1.0) to handle unweighted or differently named weight attributes
        weight = data.get("weight", 1.0)
        rx_graph.add_edge(node_map[u], node_map[v], weight) # Store weight as edge payload

    return rx_graph, node_map


def compare_with_true_labels(communities, true_labels, node_map=None):
    """
    Compare detected communities with ground truth.
    
    Returns adjusted Rand index (ARI) and normalized mutual information (NMI).
    """
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
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi


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


def create_comparison_chart(benchmark_results, output_file="algorithm_comparison.png"):
    """
    Create a comparative bar chart showing performance metrics.
    """
    if not benchmark_results:
        print("No benchmark results available to create comparison chart")
        return

    datasets = [result["dataset"] for result in benchmark_results]
    n_datasets = len(datasets)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 18))
    bar_width = 0.35
    r1 = np.arange(n_datasets)
    r2 = [x + bar_width for x in r1]

    # ARI comparison
    ari_nx = [result["nx_ari"] for result in benchmark_results]
    ari_rx_quality = [result["rx_quality_ari"] for result in benchmark_results]
    ax1.bar(r1, ari_nx, width=bar_width, label="NetworkX", color="blue")
    ax1.bar(r2, ari_rx_quality, width=bar_width, label="RustWorkX", color="green")
    ax1.set_ylabel("Adjusted Rand Index (ARI)", fontweight="bold")
    ax1.set_title("Louvain Community Detection Quality - ARI (higher is better)", fontweight="bold")
    ax1.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax1.set_xticklabels(datasets)
    ax1.set_ylim(0, 1.05)
    for i, v in enumerate(ari_nx):
        ax1.text(r1[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center")
    for i, v in enumerate(ari_rx_quality):
        ax1.text(r2[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center")
    ax1.legend()

    # NMI comparison
    nmi_nx = [result["nx_nmi"] for result in benchmark_results]
    nmi_rx_quality = [result["rx_quality_nmi"] for result in benchmark_results]
    ax2.bar(r1, nmi_nx, width=bar_width, label="NetworkX", color="blue")
    ax2.bar(r2, nmi_rx_quality, width=bar_width, label="RustWorkX", color="green")
    ax2.set_ylabel("Normalized Mutual Info (NMI)", fontweight="bold")
    ax2.set_title("Louvain Community Detection Quality - NMI (higher is better)", fontweight="bold")
    ax2.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax2.set_xticklabels(datasets)
    ax2.set_ylim(0, 1.05)
    for i, v in enumerate(nmi_nx):
        ax2.text(r1[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center")
    for i, v in enumerate(nmi_rx_quality):
        ax2.text(r2[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center")
    ax2.legend()

    # Execution time comparison
    time_nx = [result["nx_elapsed"] for result in benchmark_results]
    time_rx_quality = [result["rx_elapsed_quality"] for result in benchmark_results]
    ax3.bar(r1, time_nx, width=bar_width, label="NetworkX", color="blue")
    ax3.bar(r2, time_rx_quality, width=bar_width, label="RustWorkX", color="green")
    ax3.set_ylabel("Execution Time (seconds)", fontweight="bold")
    ax3.set_title("Louvain Performance Comparison - Execution Time (lower is better)", fontweight="bold")
    ax3.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax3.set_xticklabels(datasets)
    max_time = max(max(time_nx), max(time_rx_quality)) * 1.1 if time_nx and time_rx_quality else 0.1 # Avoid error if lists are empty
    ax3.set_ylim(0, max_time)
    for i, v in enumerate(time_nx):
        ax3.text(r1[i] + bar_width/2, v + 0.02 * max_time, f"{format_time(v)}", ha="center")
    for i, v in enumerate(time_rx_quality):
        ax3.text(r2[i] + bar_width/2, v + 0.02 * max_time, f"{format_time(v)}", ha="center")
    ax3.legend()

    # Memory usage comparison
    memory_nx = [result.get("nx_memory", 0) for result in benchmark_results]
    memory_rx_quality = [result.get("rx_memory_quality", 0) for result in benchmark_results]
    ax4.bar(r1, memory_nx, width=bar_width, label="NetworkX", color="blue")
    ax4.bar(r2, memory_rx_quality, width=bar_width, label="RustWorkX", color="green")
    ax4.set_ylabel("Memory Usage (MB)", fontweight="bold")
    ax4.set_title("Louvain Performance Comparison - Memory Usage (lower is better)", fontweight="bold")
    ax4.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax4.set_xticklabels(datasets)
    max_memory = max(max(memory_nx), max(memory_rx_quality)) * 1.1 if memory_nx and memory_rx_quality else 0.1 # Avoid error if lists are empty
    ax4.set_ylim(0, max_memory)
    for i, v in enumerate(memory_nx):
        ax4.text(r1[i] + bar_width/2, v + 0.02 * max_memory, f"{format_memory(v)}", ha="center")
    for i, v in enumerate(memory_rx_quality):
        ax4.text(r2[i] + bar_width/2, v + 0.02 * max_memory, f"{format_memory(v)}", ha="center")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Comparison chart saved to '{output_file}'")


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


def run_benchmark():
    """Run benchmark comparison between rustworkx and networkx on all datasets."""
    datasets = [
        ("Karate Club", load_karate_club),
        ("Les Misérables", load_les_miserables),
        ("Football", load_football),
        ("Political Books", load_political_books),
        ("Dolphins", load_dolphins),
        ("Political Blogs", load_polblogs),
        ("Cora", load_cora),
        ("Facebook", load_facebook),
        ("LFR Benchmark", load_lfr),
        ("Large Synthetic", load_large_synthetic)
    ]
    
    benchmark_results = []
    for dataset_name, load_func in datasets:
        try:
            result = run_benchmark_on_dataset(dataset_name, load_func)
            benchmark_results.append(result)
        except Exception as e:
            print(f"Error processing {dataset_name} dataset: {str(e)}")
    
    create_comparison_chart(benchmark_results, "community_detection_comparison.png")


@measure_memory
def run_nx_algorithm(nx_graph):
    """Run NetworkX Louvain algorithm with memory measurement."""
    # NetworkX Louvain implicitly uses the 'weight' attribute if present
    return list(nx.community.louvain_communities(nx_graph, seed=42, weight="weight"))


@measure_memory
def run_rx_quality_algorithm(rx_graph):
    """Run RustWorkX Louvain algorithm with memory measurement."""
    # Define the weight function: simply return the edge payload (which is the weight)
    def weight_fn(edge_payload):
        return edge_payload

    # Call louvain_communities with the weight function
    return rx.louvain_communities(rx_graph, weight_fn=weight_fn, seed=42)


def run_benchmark_on_dataset(dataset_name, load_func):
    """Run benchmark comparison on a specific dataset."""
    print(f"\n{'='*50}")
    print(f"Running Louvain benchmark on {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Use fixed seed for reproducibility in this function scope
    fixed_seed = 42 
    random.seed(fixed_seed)
    np.random.seed(fixed_seed)
    
    try:
        nx_graph, true_labels = load_func()
        print(f"Loaded {dataset_name} dataset: {len(nx_graph.nodes())} nodes, {len(nx_graph.edges())} edges")
        print(f"Ground truth: {len(set(true_labels))} communities")
        
        rx_graph, node_map = convert_nx_to_rx(nx_graph)
        
        # Initialize results with default values
        nx_communities = []
        nx_memory = 0
        nx_elapsed = 0
        rx_communities_quality = []
        rx_memory_quality = 0
        rx_elapsed_quality = 0
        
        # Run NetworkX with timeout protection
        try:
            print("\nRunning NetworkX Louvain...")
            nx_start = time.time()
            
            # Check if the graph is directed for NetworkX
            if nx.is_directed(nx_graph):
                print("Converting directed graph to undirected for NetworkX Louvain...")
                nx_graph_undirected = nx.Graph(nx_graph)
                # Pass seed to nx.community.louvain_communities
                nx_communities, nx_memory = run_nx_algorithm(nx_graph_undirected) 
            else:
                # Pass seed to nx.community.louvain_communities
                nx_communities, nx_memory = run_nx_algorithm(nx_graph)
                
            nx_elapsed = time.time() - nx_start
            print(f"NetworkX completed in {format_time(nx_elapsed)}")
        except Exception as e:
            print(f"Error running NetworkX Louvain: {str(e)}")
            nx_communities = []
            nx_memory = 0
            nx_elapsed = 0
        
        # Run RustWorkX with timeout protection
        try:
            print("Running RustWorkX Louvain...")
            rx_quality_start = time.time()
            # Pass seed to rx.louvain_communities
            rx_communities_quality, rx_memory_quality = run_rx_quality_algorithm(rx_graph) 
            rx_elapsed_quality = time.time() - rx_quality_start
            print(f"RustWorkX completed in {format_time(rx_elapsed_quality)}")
        except Exception as e:
            print(f"Error running RustWorkX Louvain: {str(e)}")
            rx_communities_quality = []
            rx_memory_quality = 0
            rx_elapsed_quality = 0
        
        print("\nResults:")
        print(f"NetworkX found {len(nx_communities)} communities in {format_time(nx_elapsed)} using {format_memory(nx_memory)}")
        # Limit printing community details for large datasets
        if len(nx_communities) < 50: 
            for i, comm in enumerate(nx_communities):
                print(f"  Community {i+1}: {len(comm)} members")
        print(f"RustWorkX found {len(rx_communities_quality)} communities in {format_time(rx_elapsed_quality)} using {format_memory(rx_memory_quality)}")
        if len(rx_communities_quality) < 50:
            for i, comm in enumerate(rx_communities_quality):
                print(f"  Community {i+1}: {len(comm)} members")
        
        nx_ari, nx_nmi = compare_with_true_labels(nx_communities, true_labels)
        rx_quality_ari, rx_quality_nmi = compare_with_true_labels(rx_communities_quality, true_labels, node_map)
        
        print("\nComparison with ground truth:")
        print(f"NetworkX - Adjusted Rand Index: {nx_ari:.4f}, Normalized Mutual Information: {nx_nmi:.4f}")
        print(f"RustWorkX - Adjusted Rand Index: {rx_quality_ari:.4f}, Normalized Mutual Information: {rx_quality_nmi:.4f}")
        
        print("\nPerformance comparison:")
        if rx_elapsed_quality > 0 and nx_elapsed > 0:
            print(f"RustWorkX is {nx_elapsed / rx_elapsed_quality:.2f}x faster than NetworkX")
        else:
            # Avoid division by zero
            print("RustWorkX execution time too small to compare meaningfully." if nx_elapsed > 0 else "Both execution times are zero.")
        
        print("\nMemory usage comparison:")
        print(f"NetworkX: {format_memory(nx_memory)}")
        if rx_memory_quality > 0 and nx_memory > 0:
            print(f"RustWorkX: {format_memory(rx_memory_quality)} ({nx_memory / rx_memory_quality:.2f}x of NetworkX)")
        else:
            print(f"RustWorkX: {format_memory(rx_memory_quality)}")
        
        try:
            visualize_results_extended(
                nx_graph, nx_communities, rx_communities_quality,
                node_map, true_labels, dataset_name,
                nx_ari, nx_nmi, nx_elapsed,
                rx_quality_ari, rx_quality_nmi, rx_elapsed_quality,
                nx_memory, rx_memory_quality
            )
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
        
        return {
            "dataset": dataset_name,
            "nx_ari": nx_ari,
            "nx_nmi": nx_nmi,
            "nx_elapsed": nx_elapsed,
            "nx_memory": nx_memory,
            "rx_quality_ari": rx_quality_ari,
            "rx_quality_nmi": rx_quality_nmi,
            "rx_elapsed_quality": rx_elapsed_quality,
            "rx_memory_quality": rx_memory_quality
        }
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "nx_ari": 0,
            "nx_nmi": 0,
            "nx_elapsed": 0,
            "nx_memory": 0,
            "rx_quality_ari": 0,
            "rx_quality_nmi": 0,
            "rx_elapsed_quality": 0,
            "rx_memory_quality": 0
        }


def visualize_results_extended(nx_graph, nx_communities, rx_communities_quality,
                               node_map, true_labels, dataset_name,
                               nx_ari, nx_nmi, nx_elapsed,
                               rx_quality_ari, rx_quality_nmi, rx_elapsed_quality,
                               nx_memory, rx_memory_quality):
    """Visualize the network with ground truth and detected communities."""
    try:
        # Для больших графов пропускаем визуализацию
        if len(nx_graph.nodes()) > 1000:
            print(f"Skipping visualization for large graph: {dataset_name}")
            return
            
        plt.figure(figsize=(20, 6))
        plt.suptitle(f"Louvain Community Detection Results for {dataset_name} Dataset", fontsize=16)
        
        rx_quality_node_comm = {}
        for i, comm in enumerate(rx_communities_quality):
            for node in comm:
                original_node = next((k for k, v in node_map.items() if v == node), None)
                if original_node is not None:
                    rx_quality_node_comm[original_node] = i
        
        true_node_comm = {i: label for i, label in enumerate(true_labels)}
    
        pos = nx.get_node_attributes(nx_graph, "pos")
        if not pos:
            pos = nx.spring_layout(nx_graph, k=1/np.sqrt(len(nx_graph.nodes())), iterations=50, seed=42)
    
        # Prepare colors (use a colormap for potentially many communities)
        cmap = plt.get_cmap("viridis")
        max_comm_id = max(len(set(true_labels)), 
                            len(nx_communities),
                            len(rx_communities_quality))
        # Generate RGBA tuples first
        rgba_colors = [cmap(i / max(1, max_comm_id - 1)) for i in range(max_comm_id)]
        # Convert to hex for networkx drawing
        hex_colors = [mcolors.to_hex(c) for c in rgba_colors]
        
        # Create a mapping from potentially non-numeric true labels to integer indices for coloring
        unique_true_labels = sorted(list(set(true_labels)))
        label_to_int_map = {label: i for i, label in enumerate(unique_true_labels)}
        
        # Ground Truth
        plt.subplot(1, 3, 1)
        num_true_communities = len(unique_true_labels)
        plt.title(f"Ground Truth: {num_true_communities} communities")
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.3)
        # Use the mapping for colors and handle label display
        for comm_id in unique_true_labels:
            node_list = [node for node, comm in true_node_comm.items() if comm == comm_id]
            if not node_list: continue # Skip empty communities
            color_index = label_to_int_map[comm_id] # Get integer index for color
            nx.draw_networkx_nodes(
                nx_graph, pos,
                nodelist=node_list,
                node_color=hex_colors[color_index % len(hex_colors)], # Use integer index
                node_size=80,
                alpha=0.8,
                label=f"True Comm {comm_id}" # Display original comm_id, avoid +1
            )
        nx.draw_networkx_labels(nx_graph, pos, font_size=8)
        plt.axis("off")
    
        # NetworkX Results
        plt.subplot(1, 3, 2)
        num_nx_communities = len(nx_communities)
        plt.title(f"NetworkX (Louvain): {num_nx_communities} communities\nARI: {nx_ari:.3f}, NMI: {nx_nmi:.3f}\nTime: {format_time(nx_elapsed)}, Mem: {format_memory(nx_memory)}")
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.3)
        nx_node_comm = {}
        for i, comm in enumerate(nx_communities):
            if not comm: continue # Skip empty communities
            for node in comm:
                nx_node_comm[node] = i
            # Use i directly as comm_id for color mapping
            nx.draw_networkx_nodes(
                nx_graph, pos,
                nodelist=list(comm), # Ensure nodelist is a list
                node_color=hex_colors[i % len(hex_colors)],
                node_size=80,
                alpha=0.8,
                label=f"NX Comm {i+1}"
            )
        nx.draw_networkx_labels(nx_graph, pos, font_size=8)
        plt.axis("off")
    
        # RustWorkX (Louvain)
        plt.subplot(1, 3, 3)
        num_rx_communities = len(rx_communities_quality)
        plt.title(f"RustWorkX (Louvain): {num_rx_communities} communities\nARI: {rx_quality_ari:.3f}, NMI: {rx_quality_nmi:.3f}\nTime: {format_time(rx_elapsed_quality)}, Mem: {format_memory(rx_memory_quality)}")
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.3)
        for i, comm in enumerate(rx_communities_quality): # Use index i as comm_id
            if not comm: continue # Skip empty communities
            # Get original nodes corresponding to rx node indices
            original_node_list = [k for k, v in node_map.items() if v in comm]
            if not original_node_list: continue # Skip if mapping fails or comm is effectively empty
            # Use i directly as comm_id for color mapping
            nx.draw_networkx_nodes(
                nx_graph, pos,
                nodelist=original_node_list, # Use original nodes for drawing
                node_color=hex_colors[i % len(hex_colors)],
                node_size=80,
                alpha=0.8,
                label=f"RX Comm {i+1}"
            )
        nx.draw_networkx_labels(nx_graph, pos, font_size=8)
        plt.axis("off")
    
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        filename = f"{dataset_name.lower().replace(' ', '_')}_louvain_comparison.png" # Updated filename
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        print(f"Visualization saved to '{filename}'")
    except Exception as e:
        # Print more details on visualization error
        import traceback
        print(f"Error during visualization for {dataset_name}: {str(e)}")
        traceback.print_exc() 
        plt.close()


if __name__ == "__main__":
    run_benchmark()
