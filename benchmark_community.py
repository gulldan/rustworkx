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
import os
import random
import shutil
import time
import tracemalloc
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
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


def load_karate_club():
    """Load Zachary's Karate Club dataset with ground truth communities."""
    G = nx.karate_club_graph()
    true_labels = [1 if G.nodes[node]["club"] == "Mr. Hi" else 0 for node in G.nodes()]
    return G, true_labels


def load_davis_women():
    """Load Davis Southern Women dataset with ground truth communities."""
    G = nx.davis_southern_women_graph()
    # No explicit ground truth in networkx, creating dummy labels (all in one group)
    # In practice, you might derive labels from attributes if available, or use graphs where ground truth is known.
    true_labels = [0] * len(G.nodes())
    print("Warning: Davis Southern Women graph loaded with dummy ground truth (all nodes in one group).")
    return G, true_labels


def load_florentine_families():
    """Load Florentine Families dataset with ground truth communities."""
    G = nx.florentine_families_graph()
    # Again, no standard ground truth. Using dummy labels.
    true_labels = [0] * len(G.nodes())
    print("Warning: Florentine Families graph loaded with dummy ground truth (all nodes in one group).")
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


def load_citeseer():
    """Load Citeseer dataset with ground truth communities.
    
    Assumes file 'datasets/citeseer.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/citeseer.gml", label="id")
    true_labels = [G.nodes[node].get("value", 0) for node in G.nodes()] # Use .get for safety
    return G, true_labels


def load_email_eu_core():
    """Load Email EU Core dataset with ground truth communities.
    
    Assumes file 'datasets/email_eu_core.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/email_eu_core.gml", label="id")
    true_labels = [G.nodes[node].get("value", 0) for node in G.nodes()] # Use .get for safety
    return G, true_labels


def load_lfr(n=250, mu=0.1, name="LFR Benchmark"):
    """Generate a synthetic LFR benchmark graph with ground truth communities."""
    tau1 = 3
    tau2 = 1.5
    # Generate the LFR benchmark graph (node attribute "community" is a frozenset)
    print(f"Generating {name} graph (n={n}, mu={mu})...")
    start_time = time.time()
    G = nx.LFR_benchmark_graph(n, tau1, tau2, mu,
                              average_degree=10, # Increased density slightly
                              min_community=50, 
                              max_community=max(100, n // 5), # Adjust max community size based on n
                              seed=42)
    gen_time = time.time() - start_time
    print(f"Generated {name} in {format_time(gen_time)}")
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


def load_snap_text_dataset(name, edge_file_base, community_file_base):
    """Generic function to load SNAP text datasets (ungraph.txt, all.cmty.txt).
    
    Handles unzipping .gz files if necessary.
    """
    edge_file_gz = f"datasets/{edge_file_base}.ungraph.txt.gz"
    community_file_gz = f"datasets/{community_file_base}.all.cmty.txt.gz"
    edge_file = f"datasets/{edge_file_base}.ungraph.txt"
    community_file = f"datasets/{community_file_base}.all.cmty.txt"

    # Check if unzipped files exist, if not, try to unzip from .gz
    if not os.path.exists(edge_file):
        if os.path.exists(edge_file_gz):
            print(f"Unzipping {edge_file_gz}...")
            import gzip
            with gzip.open(edge_file_gz, "rb") as f_in:
                with open(edge_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Unzipped to {edge_file}")
        else:
            raise FileNotFoundError(
                f"{name} dataset edge file not found. Please download '{edge_file_gz}' "
                f"from SNAP, place it in 'datasets', and optionally unzip it."
            )

    if not os.path.exists(community_file):
        if os.path.exists(community_file_gz):
            print(f"Unzipping {community_file_gz}...")
            import gzip
            with gzip.open(community_file_gz, "rb") as f_in:
                with open(community_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Unzipped to {community_file}")
        else:
            raise FileNotFoundError(
                f"{name} dataset community file not found. Please download '{community_file_gz}' "
                f"from SNAP, place it in 'datasets', and optionally unzip it."
            )

    print(f"Loading {name} graph ({edge_file})...")
    start_time = time.time()
    # Load graph, skipping comments and assuming node IDs are integers
    G = nx.read_edgelist(
        edge_file, 
        comments="#", 
        create_using=nx.Graph(), 
        nodetype=int
    )
    load_time = time.time() - start_time
    print(f"Loaded {name} graph ({len(G.nodes())} nodes, {len(G.edges())} edges) in {format_time(load_time)}")

    print(f"Loading {name} communities ({community_file})...")
    start_time = time.time()
    communities = {}
    nodes_in_communities = set()
    # Read communities file (nodes are space-separated)
    with open(community_file) as f:
        for comm_id, line in enumerate(f):
            # Handle potential empty lines or non-integer nodes gracefully
            try:
                nodes = [int(n) for n in line.strip().split()] 
                if nodes: # Only add non-empty communities
                    communities[comm_id] = set(nodes) # Store as set for faster lookups
                    nodes_in_communities.update(nodes)
            except ValueError:
                print(f"Warning: Skipping invalid line in community file: {line.strip()}")
                continue
    
    load_comm_time = time.time() - start_time
    num_communities = len(communities)
    print(f"Loaded communities (approx {num_communities} groups) in {format_time(load_comm_time)}")

    # Check if all nodes from the graph are listed in any community
    nodes_not_in_comm = set(G.nodes()) - nodes_in_communities
    if nodes_not_in_comm:
        print(f"Warning: {len(nodes_not_in_comm)} nodes from the graph are not listed in any community.")

    # Create true_labels mapping based on the first community a node appears in.
    # Map original node IDs to consecutive integers starting from 0 for the graph nodes
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)
    true_labels = [-1] * num_nodes # Initialize with -1 (or another indicator for 'no community')

    # Assign labels based on communities
    # Note: For overlapping communities, this assigns the ID of the *last* community read containing the node.
    # This might not be the standard way to handle overlaps, but provides *some* ground truth.
    # Consider alternative strategies if overlap handling is critical.
    labeled_nodes_count = 0
    for comm_id, nodes in communities.items():
        for node in nodes:
            if node in node_to_idx:
                idx = node_to_idx[node]
                if true_labels[idx] == -1: # Assign label only if not already assigned
                    labeled_nodes_count += 1
                true_labels[idx] = comm_id # Overwrite with last seen community ID

    print(f"Assigned community labels to {labeled_nodes_count}/{num_nodes} nodes based on provided communities.")
    
    # Optional: Handle nodes without assigned communities (true_labels == -1)
    # One strategy: Assign them to a unique 'unassigned' community ID
    max_assigned_label = max(true_labels) if labeled_nodes_count > 0 else -1
    unassigned_label = max_assigned_label + 1
    for i in range(num_nodes):
        if true_labels[i] == -1:
            true_labels[i] = unassigned_label
    print(f"Nodes without an explicit community assigned label {unassigned_label}.")

    # Ensure the graph nodes match the true_labels length
    if len(G.nodes()) != len(true_labels):
        # This should ideally not happen if node_list and true_labels are derived correctly
        raise ValueError("Mismatch between number of graph nodes and true labels generated.")

    return G, true_labels


def load_orkut():
    """Load the Orkut dataset from SNAP text files."""
    return load_snap_text_dataset("Orkut", "com-orkut", "com-orkut")


def load_livejournal():
    """Load the LiveJournal dataset from SNAP text files."""
    return load_snap_text_dataset("LiveJournal", "com-lj", "com-lj")


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
    
    Returns adjusted Rand index (ARI), normalized mutual information (NMI),
    homogeneity, completeness, V-measure, and Fowlkes-Mallows Index (FMI).
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


def create_comparison_chart(benchmark_results, output_file="community_detection_comparison.png"):
    """
    Create a comparative bar chart showing performance metrics with improved aesthetics and save to file.
    """
    if not benchmark_results:
        print("No benchmark results available to create comparison chart")
        return None

    datasets = [result["dataset"] for result in benchmark_results]
    n_datasets = len(datasets)
    
    # Use non-interactive backend for better image quality
    plt.switch_backend("Agg")
    
    # Create figure with more space between subplots and higher DPI
    # Adjust figsize and number of subplots for new metrics
    # Correct number of subplots to 9 (7 metrics + time + memory)
    fig, axes = plt.subplots(9, 1, figsize=(14, 48), dpi=300) # Changed 7 to 9, adjusted height
    plt.subplots_adjust(hspace=0.55)  # Slightly increased space between plots
    
    # Assign axes for clarity
    # Now unpacking should work correctly
    ax_ari, ax_nmi, ax_homogeneity, ax_completeness, ax_vmeasure, ax_fmi, ax_modularity, ax_time, ax_memory = axes.flatten()
    
    bar_width = 0.35
    r1 = np.arange(n_datasets)
    r2 = [x + bar_width for x in r1]

    # Use visually distinct colors
    nx_color = "#4285F4"  # blue
    rx_color = "#34A853"  # green
    
    # Add a light grid for better readability
    grid_color = "#E5E5E5"
    grid_linewidth = 0.5

    # ARI comparison
    ari_nx = [result["nx_ari"] for result in benchmark_results]
    ari_rx_quality = [result["rx_quality_ari"] for result in benchmark_results]
    
    # Draw bars with slightly rounded corners and edge color
    ax_ari.bar(r1, ari_nx, width=bar_width, label="NetworkX", color=nx_color, 
           edgecolor="#3B77DB", linewidth=0.8)
    ax_ari.bar(r2, ari_rx_quality, width=bar_width, label="RustWorkX", color=rx_color,
           edgecolor="#2D9249", linewidth=0.8)
    
    # Styling
    ax_ari.set_ylabel("Adjusted Rand Index (ARI)", fontweight="bold", fontsize=14)
    ax_ari.set_title("Louvain Community Detection Quality - ARI (higher is better)", 
                 fontweight="bold", fontsize=16)
    ax_ari.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_ari.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_ari.set_ylim(0, 1.05)
    
    # Add grid for better readability
    ax_ari.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_ari.set_axisbelow(True)  # Put grid behind bars
    
    # Add value labels with better formatting
    for i, v in enumerate(ari_nx):
        ax_ari.text(r1[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center", 
                va="bottom", fontweight="bold", fontsize=10)
    for i, v in enumerate(ari_rx_quality):
        ax_ari.text(r2[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center", 
                va="bottom", fontweight="bold", fontsize=10)
    
    # Add a more stylish legend
    ax_ari.legend(loc="upper right", frameon=True, framealpha=0.9, 
              edgecolor="#CCCCCC", fontsize=12)
    
    # NMI comparison - similar styling
    nmi_nx = [result["nx_nmi"] for result in benchmark_results]
    nmi_rx_quality = [result["rx_quality_nmi"] for result in benchmark_results]
    
    ax_nmi.bar(r1, nmi_nx, width=bar_width, label="NetworkX", color=nx_color,
           edgecolor="#3B77DB", linewidth=0.8)
    ax_nmi.bar(r2, nmi_rx_quality, width=bar_width, label="RustWorkX", color=rx_color,
           edgecolor="#2D9249", linewidth=0.8)
    
    ax_nmi.set_ylabel("Normalized Mutual Info (NMI)", fontweight="bold", fontsize=14)
    ax_nmi.set_title("Louvain Community Detection Quality - NMI (higher is better)", 
                 fontweight="bold", fontsize=16)
    ax_nmi.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_nmi.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_nmi.set_ylim(0, 1.05)
    
    # Add grid
    ax_nmi.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_nmi.set_axisbelow(True)
    
    # Add value labels
    for i, v in enumerate(nmi_nx):
        ax_nmi.text(r1[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center", 
                va="bottom", fontweight="bold", fontsize=10)
    for i, v in enumerate(nmi_rx_quality):
        ax_nmi.text(r2[i] + bar_width/2, v + 0.02, f"{v:.3f}", ha="center", 
                va="bottom", fontweight="bold", fontsize=10)
    
    ax_nmi.legend(loc="upper right", frameon=True, framealpha=0.9, 
              edgecolor="#CCCCCC", fontsize=12)
    
    # Homogeneity comparison
    homogeneity_nx = [result["nx_homogeneity"] for result in benchmark_results]
    homogeneity_rx = [result["rx_quality_homogeneity"] for result in benchmark_results]
    ax_homogeneity.bar(r1, homogeneity_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_homogeneity.bar(r2, homogeneity_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_homogeneity.set_ylabel("Homogeneity Score", fontweight="bold", fontsize=14)
    ax_homogeneity.set_title("Community Detection Quality - Homogeneity (higher is better)", fontweight="bold", fontsize=16)
    ax_homogeneity.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_homogeneity.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_homogeneity.set_ylim(0, 1.05)
    ax_homogeneity.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_homogeneity.set_axisbelow(True)
    for i, v in enumerate(homogeneity_nx): ax_homogeneity.text(r1[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for i, v in enumerate(homogeneity_rx): ax_homogeneity.text(r2[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_homogeneity.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # Completeness comparison
    completeness_nx = [result["nx_completeness"] for result in benchmark_results]
    completeness_rx = [result["rx_quality_completeness"] for result in benchmark_results]
    ax_completeness.bar(r1, completeness_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_completeness.bar(r2, completeness_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_completeness.set_ylabel("Completeness Score", fontweight="bold", fontsize=14)
    ax_completeness.set_title("Community Detection Quality - Completeness (higher is better)", fontweight="bold", fontsize=16)
    ax_completeness.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_completeness.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_completeness.set_ylim(0, 1.05)
    ax_completeness.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_completeness.set_axisbelow(True)
    for i, v in enumerate(completeness_nx): ax_completeness.text(r1[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for i, v in enumerate(completeness_rx): ax_completeness.text(r2[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_completeness.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # V-Measure comparison
    vmeasure_nx = [result["nx_v_measure"] for result in benchmark_results]
    vmeasure_rx = [result["rx_quality_v_measure"] for result in benchmark_results]
    ax_vmeasure.bar(r1, vmeasure_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_vmeasure.bar(r2, vmeasure_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_vmeasure.set_ylabel("V-Measure Score", fontweight="bold", fontsize=14)
    ax_vmeasure.set_title("Community Detection Quality - V-Measure (higher is better)", fontweight="bold", fontsize=16)
    ax_vmeasure.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_vmeasure.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_vmeasure.set_ylim(0, 1.05)
    ax_vmeasure.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_vmeasure.set_axisbelow(True)
    for i, v in enumerate(vmeasure_nx): ax_vmeasure.text(r1[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for i, v in enumerate(vmeasure_rx): ax_vmeasure.text(r2[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_vmeasure.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # FMI comparison
    fmi_nx = [result["nx_fmi"] for result in benchmark_results]
    fmi_rx = [result["rx_quality_fmi"] for result in benchmark_results]
    ax_fmi.bar(r1, fmi_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_fmi.bar(r2, fmi_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_fmi.set_ylabel("Fowlkes–Mallows Index (FMI)", fontweight="bold", fontsize=14)
    ax_fmi.set_title("Community Detection Quality - FMI (higher is better)", fontweight="bold", fontsize=16)
    ax_fmi.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_fmi.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_fmi.set_ylim(0, 1.05)
    ax_fmi.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_fmi.set_axisbelow(True)
    for i, v in enumerate(fmi_nx): ax_fmi.text(r1[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for i, v in enumerate(fmi_rx): ax_fmi.text(r2[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_fmi.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # Modularity comparison
    modularity_nx = [result["nx_modularity"] for result in benchmark_results]
    modularity_rx = [result["rx_modularity"] for result in benchmark_results]
    # Determine appropriate y-limits for modularity (can be negative)
    min_mod = min(min(modularity_nx), min(modularity_rx), 0) # Include 0 for reference
    max_mod = max(max(modularity_nx), max(modularity_rx)) * 1.1 # Add some padding
    
    ax_modularity.bar(r1, modularity_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_modularity.bar(r2, modularity_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_modularity.set_ylabel("Modularity Score", fontweight="bold", fontsize=14)
    ax_modularity.set_title("Community Detection Quality - Modularity (higher is better)", fontweight="bold", fontsize=16)
    ax_modularity.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_modularity.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_modularity.set_ylim(min_mod, max_mod)
    ax_modularity.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_modularity.set_axisbelow(True)
    # Adjust text label placement for potentially negative values
    for i, v in enumerate(modularity_nx): ax_modularity.text(r1[i], v + (max_mod - min_mod)*0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for i, v in enumerate(modularity_rx): ax_modularity.text(r2[i], v + (max_mod - min_mod)*0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax_modularity.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # Execution time comparison with logarithmic scale
    time_nx = [result["nx_elapsed"] for result in benchmark_results]
    time_rx_quality = [result["rx_elapsed_quality"] for result in benchmark_results]
    
    # Replace zeros with a very small value to avoid log(0) issues
    time_nx = [max(t, 1e-10) for t in time_nx]
    time_rx_quality = [max(t, 1e-10) for t in time_rx_quality]
    
    ax_time.bar(r1, time_nx, width=bar_width, label="NetworkX", color=nx_color,
           edgecolor="#3B77DB", linewidth=0.8)
    ax_time.bar(r2, time_rx_quality, width=bar_width, label="RustWorkX", color=rx_color,
           edgecolor="#2D9249", linewidth=0.8)
    
    ax_time.set_ylabel("Execution Time (seconds)", fontweight="bold", fontsize=14)
    ax_time.set_title("Louvain Performance Comparison - Execution Time (lower is better)", 
                 fontweight="bold", fontsize=16)
    ax_time.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_time.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    
    # Set logarithmic scale with better formatting
    ax_time.set_yscale("log")
    ax_time.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_time.set_axisbelow(True)
    
    # Format y-axis tick labels to be more readable
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}" if x < 0.001 else f"{x:.3g}"))
    
    # Position text labels appropriately for log scale with better formatting
    for i, v in enumerate(time_nx):
        ax_time.text(r1[i], v * 1.2, f"{format_time(v)}", ha="center", 
                fontweight="bold", fontsize=10, color="#333333")
    for i, v in enumerate(time_rx_quality):
        ax_time.text(r2[i], v * 1.2, f"{format_time(v)}", ha="center", 
                fontweight="bold", fontsize=10, color="#333333")
    
    ax_time.legend(loc="upper right", frameon=True, framealpha=0.9, 
              edgecolor="#CCCCCC", fontsize=12)
    
    # Memory usage comparison with logarithmic scale
    memory_nx = [result.get("nx_memory", 0) for result in benchmark_results]
    memory_rx_quality = [result.get("rx_memory_quality", 0) for result in benchmark_results]
    
    # Replace zeros with a very small value to avoid log(0) issues
    memory_nx = [max(m, 1e-6) for m in memory_nx]
    memory_rx_quality = [max(m, 1e-6) for m in memory_rx_quality]
    
    # Use visually distinct colors
    ax_memory.bar(r1, memory_nx, width=bar_width, label="NetworkX", color=nx_color,
           edgecolor="#3B77DB", linewidth=0.8)
    ax_memory.bar(r2, memory_rx_quality, width=bar_width, label="RustWorkX", color=rx_color,
           edgecolor="#2D9249", linewidth=0.8)
    
    ax_memory.set_ylabel("Memory Usage (MB)", fontweight="bold", fontsize=14)
    ax_memory.set_title("Louvain Performance Comparison - Memory Usage (lower is better)", 
                 fontweight="bold", fontsize=16)
    ax_memory.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_memory.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    
    # Set logarithmic scale and add grid
    ax_memory.set_yscale("log")
    ax_memory.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_memory.set_axisbelow(True)
    
    # Format y-axis tick labels using the format_memory function
    ax_memory.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_memory(x)))
    
    # Position text labels appropriately for log scale
    for i, v in enumerate(memory_nx):
        ax_memory.text(r1[i], v * 1.2, f"{format_memory(v)}", ha="center", 
                fontweight="bold", fontsize=10, color="#333333")
    for i, v in enumerate(memory_rx_quality):
        ax_memory.text(r2[i], v * 1.2, f"{format_memory(v)}", ha="center", 
                fontweight="bold", fontsize=10, color="#333333")
    
    ax_memory.legend(loc="upper right", frameon=True, framealpha=0.9, 
              edgecolor="#CCCCCC", fontsize=12)
    
    # Add an overall title and improve overall layout
    fig.suptitle("Community Detection Algorithm Comparison", 
                fontsize=20, fontweight="bold", y=0.995)
    
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    
    # Save the chart with high quality
    plt.savefig(output_file, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"Enhanced comparison chart saved to '{output_file}'")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return None


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
    
    # Dictionary mapping dataset names to loader functions
    DATASETS = {
        "Karate Club": load_karate_club,
        "Davis Southern Women": load_davis_women,
        "Florentine Families": load_florentine_families,
        "Les Misérables": load_les_miserables,
        "Football": load_football,
        "Political Books": load_political_books,
        "Dolphins": load_dolphins,
        "LFR Benchmark (Small)": lambda: load_lfr(n=500, mu=0.3, name="LFR Small"), # Small LFR
        "Political Blogs": load_polblogs,
        "Cora": load_cora,
        "Facebook": load_facebook,
        "Citeseer": load_citeseer,
        "Email EU Core": load_email_eu_core,
        # "LFR Benchmark (Medium)": lambda: load_lfr(n=5000, mu=0.4, name="LFR Medium"), # Optional medium LFR
        # --- Large Datasets (uncomment selectively to run) ---
        # "Amazon Co-purchase": load_amazon_copurchase, # ~300k nodes
        "Orkut": load_orkut,                             # ~3M nodes
        "LiveJournal": load_livejournal,                 # ~4.8M nodes
        # "Large Synthetic (SBM)": load_large_synthetic, # ~1.5k nodes (example)
        # "LFR Benchmark (Large)": lambda: load_lfr(n=100000, mu=0.5, name="LFR Large"), # Large LFR (can take time to generate)
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
            benchmark_results.append(result)
            print(f"Completed {dataset_name} dataset. Continuing to next dataset...")
            
        except Exception as e:
            print(f"❌ Error processing {dataset_name} dataset: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Generate final comparison chart
    print("\n" + "=" * 80)
    print("GENERATING FINAL PERFORMANCE COMPARISON CHART")
    print("=" * 80)
    
    try:
        chart_path = os.path.join(result_folder, "community_detection_comparison.png")
        create_comparison_chart(benchmark_results, output_file=chart_path)
    except Exception as e:
        print(f"Error generating comparison chart: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Display list of generated files
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
        
    print("\nBenchmark completed successfully!")


# Threshold for considering a graph 'large' and skipping NetworkX
LARGE_GRAPH_THRESHOLD = 100000 


def run_benchmark_on_dataset(dataset_name, load_func, result_folder):
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
        
        # Initialize metrics and modularity scores
        nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = (0.0,) * 6
        rx_quality_ari, rx_quality_nmi, rx_quality_homogeneity, rx_quality_completeness, rx_quality_v_measure, rx_quality_fmi = (0.0,) * 6
        nx_modularity = 0.0
        rx_modularity = 0.0
        
        # --- Run NetworkX (only if graph is not too large) ---
        if len(nx_graph.nodes()) <= LARGE_GRAPH_THRESHOLD:
            try:
                print("\nRunning NetworkX Louvain...")
                nx_start = time.time()
                
                # Check if the graph is directed for NetworkX
                if nx.is_directed(nx_graph):
                    print("Converting directed graph to undirected for NetworkX Louvain...")
                    nx_graph_undirected = nx.Graph(nx_graph)
                    # Pass seed to nx.community.louvain_communities
                    nx_communities, nx_memory = run_nx_algorithm(nx_graph_undirected)
                    # Calculate modularity for NetworkX result using the appropriate graph
                    if nx_communities:
                        nx_modularity = nx.community.modularity(nx_graph_undirected, nx_communities)
                else:
                    # Pass seed to nx.community.louvain_communities
                    nx_communities, nx_memory = run_nx_algorithm(nx_graph)
                    # Calculate modularity for NetworkX result
                    if nx_communities:
                        nx_modularity = nx.community.modularity(nx_graph, nx_communities)
                    
                nx_elapsed = time.time() - nx_start
                print(f"NetworkX completed in {format_time(nx_elapsed)}")
                
                # Calculate NetworkX metrics only if it ran successfully
                nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = compare_with_true_labels(nx_communities, true_labels)
                
            except Exception as e:
                print(f"Error running NetworkX Louvain: {str(e)}")
                # Reset results if NetworkX failed
                nx_communities = []
                nx_memory = 0
                nx_elapsed = 0
                nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = (0.0,) * 6
                nx_modularity = 0.0
        else:
            print(f"\nSkipping NetworkX Louvain for large graph ({len(nx_graph.nodes())} nodes > {LARGE_GRAPH_THRESHOLD})")
            # Ensure default values are set if NetworkX is skipped
            nx_communities = []
            nx_memory = 0
            nx_elapsed = 0
            nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = (0.0,) * 6
            nx_modularity = 0.0

        # --- Run RustWorkX --- 
        try:
            print("Running RustWorkX Louvain...")
            rx_quality_start = time.time()
            # Pass seed to rx.louvain_communities
            rx_communities_quality, rx_memory_quality = run_rx_quality_algorithm(rx_graph) 
            rx_elapsed_quality = time.time() - rx_quality_start
            print(f"RustWorkX completed in {format_time(rx_elapsed_quality)}")
            
            # Calculate modularity for RustWorkX result using the new function
            if rx_communities_quality:
                try:
                    # Define weight function (consistent with run_rx_quality_algorithm)
                    def weight_fn(edge_payload):
                        return edge_payload if edge_payload is not None else 1.0
                    
                    # Call the newly exposed modularity function
                    rx_modularity = rx.modularity(rx_graph, rx_communities_quality, weight_fn=weight_fn) 
                except AttributeError:
                    # Fallback if the function isn't found (e.g., module not rebuilt/exported)
                    print("Warning: rx.modularity function not found. Ensure the Rust module is rebuilt and the function is exported. Skipping calculation.")
                    rx_modularity = 0.0
                except Exception as e:
                    print(f"Error calculating RustWorkX modularity: {str(e)}")
                    rx_modularity = 0.0
            else:
                rx_modularity = 0.0 # No communities, modularity is 0

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
        
        print("\nComparison with ground truth:")
        print(f"NetworkX - ARI: {nx_ari:.4f}, NMI: {nx_nmi:.4f}, Homogeneity: {nx_homogeneity:.4f}, Completeness: {nx_completeness:.4f}, V-Measure: {nx_v_measure:.4f}, FMI: {nx_fmi:.4f}")
        if rx_communities_quality:
            rx_quality_ari, rx_quality_nmi, rx_quality_homogeneity, rx_quality_completeness, rx_quality_v_measure, rx_quality_fmi = compare_with_true_labels(rx_communities_quality, true_labels, node_map)
        else:
            # Reset RustWorkX metrics if it failed
            rx_quality_ari, rx_quality_nmi, rx_quality_homogeneity, rx_quality_completeness, rx_quality_v_measure, rx_quality_fmi = (0.0,) * 6
        
        print("\nModularity comparison (higher is better):")
        print(f"NetworkX Modularity: {nx_modularity:.4f}")
        print(f"RustWorkX Modularity: {rx_modularity:.4f}")
        
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
            # Generate a safe filename from the dataset name
            safe_filename = dataset_name.lower().replace(" ", "_").replace("-", "_")
            
            # Create visualization with our new function
            visualize_communities(
                nx_graph, nx_communities, rx_communities_quality,
                node_map, true_labels, dataset_name,
                nx_ari, nx_nmi, nx_elapsed,
                rx_quality_ari, rx_quality_nmi, rx_elapsed_quality,
                nx_memory, rx_memory_quality,
                result_folder=result_folder,
                graph_name=safe_filename
            )
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return {
            "dataset": dataset_name,
            "nx_ari": nx_ari,
            "nx_nmi": nx_nmi,
            "nx_homogeneity": nx_homogeneity,
            "nx_completeness": nx_completeness,
            "nx_v_measure": nx_v_measure,
            "nx_fmi": nx_fmi,
            "nx_modularity": nx_modularity,
            "nx_elapsed": nx_elapsed,
            "nx_memory": nx_memory,
            "rx_quality_ari": rx_quality_ari,
            "rx_quality_nmi": rx_quality_nmi,
            "rx_quality_homogeneity": rx_quality_homogeneity,
            "rx_quality_completeness": rx_quality_completeness,
            "rx_quality_v_measure": rx_quality_v_measure,
            "rx_quality_fmi": rx_quality_fmi,
            "rx_modularity": rx_modularity,
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
            "nx_homogeneity": 0,
            "nx_completeness": 0,
            "nx_v_measure": 0,
            "nx_fmi": 0,
            "nx_modularity": 0,
            "nx_elapsed": 0,
            "nx_memory": 0,
            "rx_quality_ari": 0,
            "rx_quality_nmi": 0,
            "rx_quality_homogeneity": 0,
            "rx_quality_completeness": 0,
            "rx_quality_v_measure": 0,
            "rx_quality_fmi": 0,
            "rx_modularity": 0,
            "rx_elapsed_quality": 0,
            "rx_memory_quality": 0
        }


def visualize_communities(nx_graph, nx_communities, rx_communities_quality,
                            node_map, true_labels, dataset_name,
                            nx_ari, nx_nmi, nx_elapsed,
                            rx_quality_ari, rx_quality_nmi, rx_elapsed_quality,
                            nx_memory, rx_memory_quality,
                            result_folder="results", graph_name="graph"):
    """Visualize the network with ground truth and detected communities and save to file without displaying."""
    try:
        # Adjust visualization threshold based on graph size
        if len(nx_graph.nodes()) > 2000:
            print(f"Graph too large to visualize efficiently: {dataset_name} ({len(nx_graph.nodes())} nodes)")
            return
        
        # For medium-sized graphs (between 1000-2000 nodes), create a simplified visualization
        if len(nx_graph.nodes()) > 1000:
            print(f"Creating simplified visualization for large graph: {dataset_name}")
            # Sample a smaller portion of the graph for visualization
            nodes = list(nx_graph.nodes())
            sample_size = min(1000, len(nodes))
            sampled_nodes = random.sample(nodes, sample_size)
            nx_graph = nx_graph.subgraph(sampled_nodes)
            # Update true_labels and node_map for sampled graph
            true_labels_indices = {node: i for i, node in enumerate(nodes)}
            true_labels = [true_labels[true_labels_indices[node]] if node in true_labels_indices and true_labels_indices[node] < len(true_labels) else 0 for node in sampled_nodes]
            sampled_node_map = {k: v for k, v in node_map.items() if k in sampled_nodes}
            node_map = sampled_node_map
        
        # Make sure the result folder exists
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            
        # Use non-interactive backend for better image quality
        plt.switch_backend("Agg")
        
        # Create figure with adjusted size and higher DPI
        plt.figure(figsize=(24, 10), dpi=300) # Adjusted figsize
        plt.suptitle(f"Louvain Community Detection Results: {dataset_name}", 
                     fontsize=20, fontweight="bold") # Simplified title
        
        rx_quality_node_comm = {}
        for i, comm in enumerate(rx_communities_quality):
            for node_idx in comm: # Renamed 'node' to 'node_idx' to avoid confusion
                original_node = next((k for k, v in node_map.items() if v == node_idx), None)
                if original_node is not None:
                    rx_quality_node_comm[original_node] = i
        
        # Ensure true_node_comm mapping uses original node IDs if they are not sequential integers
        original_node_ids = list(nx_graph.nodes())
        true_node_comm = {original_node_ids[i]: label for i, label in enumerate(true_labels) if i < len(original_node_ids)}
    
        # Get node positions with improved layout algorithm
        pos = nx.get_node_attributes(nx_graph, "pos")
        if not pos:
            # Try different layouts for better visualization based on graph size and density
            num_nodes = len(nx_graph.nodes())
            
            print(f"Calculating layout for {dataset_name} ({num_nodes} nodes)...")
            layout_start = time.time()
            
            if num_nodes < 50:
                pos = nx.kamada_kawai_layout(nx_graph)
            elif num_nodes < 300: # Adjusted threshold
                pos = nx.spring_layout(nx_graph, k=0.3, iterations=100, seed=42)
            else: # Use faster spring layout for larger graphs
                pos = nx.spring_layout(nx_graph, k=1/np.sqrt(num_nodes), iterations=50, seed=42)
                
            layout_time = time.time() - layout_start
            print(f"Layout calculated in {format_time(layout_time)}")
            # Store calculated positions for potential reuse if needed elsewhere
            nx.set_node_attributes(nx_graph, pos, "pos")
    
        # Use a better colormap with modern API to avoid deprecation warnings
        import matplotlib.colors as mcolors
        
        # Create a custom colormap with distinct colors
        distinct_colors = [
            "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
            "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9", # Added gray
            "#46f0f0", "#f03280", "#bcf60c", "#fabebe", "#008080", # More distinct colors
            "#e6beff", "#9a6324", "#808080", "#000000" # Added black
        ]
        
        # Extend the color list if needed
        unique_true_labels = sorted(list(set(true_labels)))
        max_comm_id = max(len(unique_true_labels), 
                         len(nx_communities),
                         len(rx_communities_quality))
        
        if max_comm_id > len(distinct_colors):
            # Use a perceptually uniform colormap like 'viridis' or 'tab20'/'tab20b'/'tab20c'
            cmap = plt.colormaps["tab20"] # Use recommended API
            additional_colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, max_comm_id)]
            hex_colors = additional_colors # Replace if needs extension
        else:
            hex_colors = distinct_colors[:max_comm_id] # Use predefined if enough
        
        # Create a mapping from potentially non-numeric true labels to integer indices for coloring
        label_to_int_map = {label: i for i, label in enumerate(unique_true_labels)}
        
        # --- Ground Truth Subplot ---
        plt.subplot(1, 3, 1)
        num_true_communities = len(unique_true_labels)
        plt.title(f"Ground Truth ({num_true_communities} communities)", 
                 fontsize=14, fontweight="bold") # Concise title
        
        # Draw edges: lighter, thinner, less opaque
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, width=0.5, edge_color="#CCCCCC") # Lighter edges
        
        # Draw nodes by community
        for comm_label in unique_true_labels:
            node_list = [node for node, label in true_node_comm.items() if label == comm_label]
            if not node_list: continue
            color_index = label_to_int_map[comm_label]
            nx.draw_networkx_nodes(
                nx_graph, pos,
                nodelist=node_list,
                node_color=hex_colors[color_index % len(hex_colors)],
                node_size=200, # Increased node size
                edgecolors="white",
                linewidths=0.5, # Thinner border
                alpha=0.9, # Slightly more opaque nodes
                label=f"Community {comm_label}"
            )
            
        # Draw labels only for very small graphs
        if len(nx_graph.nodes()) < 50: 
            nx.draw_networkx_labels(
                nx_graph, pos, 
                font_size=8, # Smaller font size
                font_color="#333333", # Darker font
                font_family="sans-serif",
                # Simple labels without bbox for less clutter
            )
        plt.axis("off")
    
        # --- NetworkX Results Subplot ---
        plt.subplot(1, 3, 2)
        num_nx_communities = len(nx_communities)
        # More informative title
        title_nx = (
            f"NetworkX ({num_nx_communities} communities)\n"
            f"ARI: {nx_ari:.3f}, NMI: {nx_nmi:.3f}\n"
            f"Time: {format_time(nx_elapsed)}, Mem: {format_memory(nx_memory)}"
        )
        plt.title(title_nx, fontsize=14, fontweight="bold", loc="center") # Centered title
                
        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, width=0.5, edge_color="#CCCCCC")
        
        # Draw nodes by detected community
        for i, comm in enumerate(nx_communities):
            if not comm: continue
            # Ensure nodes in comm exist in the current (potentially sampled) graph
            comm_nodes_in_graph = [node for node in comm if node in nx_graph]
            if not comm_nodes_in_graph: continue
            
            nx.draw_networkx_nodes(
                nx_graph, pos,
                nodelist=comm_nodes_in_graph, # Use filtered list
                node_color=hex_colors[i % len(hex_colors)],
                node_size=200,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
                label=f"NX Comm {i+1}"
            )
        
        # Draw labels only for very small graphs
        if len(nx_graph.nodes()) < 50:
            nx.draw_networkx_labels(
                nx_graph, pos, font_size=8, font_color="#333333", font_family="sans-serif"
            )
        plt.axis("off")
    
        # --- RustWorkX Results Subplot ---
        plt.subplot(1, 3, 3)
        num_rx_communities = len(rx_communities_quality)
        # More informative title
        title_rx = (
            f"RustWorkX ({num_rx_communities} communities)\n"
            f"ARI: {rx_quality_ari:.3f}, NMI: {rx_quality_nmi:.3f}\n"
            f"Time: {format_time(rx_elapsed_quality)}, Mem: {format_memory(rx_memory_quality)}"
        )
        plt.title(title_rx, fontsize=14, fontweight="bold", loc="center") # Centered title
                
        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, width=0.5, edge_color="#CCCCCC")
        
        # Draw nodes by detected community
        for i, comm_indices in enumerate(rx_communities_quality): # Renamed 'comm' to 'comm_indices'
            if not comm_indices: continue
            # Map rustworkx indices back to original node IDs present in the current graph
            original_node_list = [k for k, v in node_map.items() if v in comm_indices and k in nx_graph]
            if not original_node_list: continue
            
            nx.draw_networkx_nodes(
                nx_graph, pos,
                nodelist=original_node_list, # Use mapped list
                node_color=hex_colors[i % len(hex_colors)],
                node_size=200,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
                label=f"RX Comm {i+1}"
            )
            
        # Draw labels only for very small graphs
        if len(nx_graph.nodes()) < 50:
            nx.draw_networkx_labels(
                nx_graph, pos, font_size=8, font_color="#333333", font_family="sans-serif"
            )
        plt.axis("off")
    
        # Improve layout with adjusted spacing
        plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust rect to prevent title overlap
        
        # Save with higher DPI and optimize for viewing
        filename = os.path.join(result_folder, f"{graph_name}_louvain_comparison.png")
        plt.savefig(filename, bbox_inches="tight", dpi=300, facecolor="white", format="png") # Use dpi=300
        print(f"Saved visualization image to: {filename} (PNG)") # Updated print message
        
        # Close the figure to free memory
        plt.close("all") # Close all figures just in case
                
    except Exception as e:
        import traceback
        print(f"Error during visualization for {dataset_name}: {str(e)}")
        traceback.print_exc() 
        plt.close()


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
        # Handle cases where edge payload might not exist or is None
        return edge_payload if edge_payload is not None else 1.0

    # Call louvain_communities directly from rustworkx module
    try:
        # Corrected call: removed '.community'
        return rx.louvain_communities(rx_graph, weight_fn=weight_fn, seed=42)
    except AttributeError:
        # Reraise with a more informative message if not found
        # Updated message to reflect the correct expected location
        raise AttributeError("Could not find louvain_communities function directly in rustworkx module.")


def main():
    """
    Main function to run benchmark for different community detection algorithms.
    """
    # Just call the run_benchmark function which handles everything
    run_benchmark()


if __name__ == "__main__":
    main()
