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
import polars as pl
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
    return G, true_labels, True # Added has_ground_truth flag


def load_davis_women():
    """Load Davis Southern Women dataset with ground truth communities."""
    G = nx.davis_southern_women_graph()
    # No explicit ground truth in networkx, creating dummy labels (all in one group)
    # In practice, you might derive labels from attributes if available, or use graphs where ground truth is known.
    true_labels = [0] * len(G.nodes())
    print("Warning: Davis Southern Women graph loaded with dummy ground truth (all nodes in one group).")
    return G, true_labels, True


def load_florentine_families():
    """Load Florentine Families dataset with ground truth communities."""
    G = nx.florentine_families_graph()
    # Again, no standard ground truth. Using dummy labels.
    true_labels = [0] * len(G.nodes())
    print("Warning: Florentine Families graph loaded with dummy ground truth (all nodes in one group).")
    return G, true_labels, True


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
    return H, true_labels, True


def load_football():
    """Load American College Football dataset with ground truth communities."""
    # Assumes the file 'datasets/football.gml' exists.
    G = nx.read_gml("datasets/football.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels, True


def load_political_books():
    """Load Political Books dataset with ground truth communities."""
    # Assumes the file 'datasets/polbooks.gml' exists.
    G = nx.read_gml("datasets/polbooks.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels, True


def load_dolphins():
    """Load Dolphins Social Network dataset with ground truth communities.
    
    Assumes file 'datasets/dolphins.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/dolphins.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels, True


def load_polblogs():
    """Load Political Blogs dataset with ground truth communities.
    
    Assumes file 'datasets/polblogs.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/polblogs.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels, True


def load_cora():
    """Load Cora citation network with ground truth communities.
    
    Assumes file 'datasets/cora.gml' exists and that node attribute 'value'
    contains the class label.
    """
    G = nx.read_gml("datasets/cora.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels, True


def load_facebook():
    """Load Facebook Ego Networks / Social Circles dataset with ground truth communities.
    
    Assumes file 'datasets/facebook.gml' exists and that node attribute 'value'
    contains the community label.
    """
    G = nx.read_gml("datasets/facebook.gml", label="id")
    true_labels = [G.nodes[node]["value"] for node in G.nodes()]
    return G, true_labels, True


def load_citeseer():
    """Load Citeseer dataset with ground truth communities.
    
    Assumes file 'datasets/citeseer.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/citeseer.gml", label="id")
    true_labels = [G.nodes[node].get("value", 0) for node in G.nodes()] # Use .get for safety
    return G, true_labels, True


def load_email_eu_core():
    """Load Email EU Core dataset with ground truth communities.
    
    Assumes file 'datasets/email_eu_core.gml' exists and that node attribute 'value'
    contains the ground truth community.
    """
    G = nx.read_gml("datasets/email_eu_core.gml", label="id")
    true_labels = [G.nodes[node].get("value", 0) for node in G.nodes()] # Use .get for safety
    return G, true_labels, True


def load_graph_edges_csv():
    """Load graph from CSV file without ground truth communities.
    
    Creates dummy ground truth communities (all nodes in one group)
    for benchmark compatibility.
    """
    print("Loading graph from datasets/graph_edges.csv...")
    try:
        # Read edges from CSV
        df = pl.read_csv("datasets/graph_edges.csv")
        print(f"CSV file columns: {df.columns}")
        
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add edges with weights (distance as weight)
        for row in df.iter_rows(named=True):
            src = row["src"]
            dst = row["dst"]
            # Use distance (if available) or default to 1.0
            weight = float(row.get("distance", 1.0))
            G.add_edge(src, dst, weight=weight)
        
        # Get node and edge counts (convert to list to avoid iterator issues)
        nodes = list(G.nodes())
        edges = list(G.edges())
        
        # Print graph info
        print(f"Created graph with {len(nodes)} nodes and {len(edges)} edges")
        
        # Create dummy ground truth (all in one community)
        true_labels = [0] * len(nodes)
        print("Warning: No ground truth available for this dataset.")
        
        return G, true_labels, False # Indicate no ground truth
    except Exception as e:
        print(f"Error loading graph from CSV: {e}")
        # Return tiny dummy graph in case of error
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False # Indicate no ground truth on error too


def load_graph_edges_parquet():
    """Load graph from Parquet file without ground truth communities.
    
    Creates dummy ground truth communities (all nodes in one group)
    for benchmark compatibility.
    """
    print("Loading graph from datasets/graph_edges_big.parquet...")
    try:
        # Read edges from Parquet
        df = pl.read_parquet("datasets/graph_edges_big.parquet")
        print(f"Parquet file columns: {df.columns}")
        
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add edges with weights (distance as weight)
        for row in df.iter_rows(named=True):
            src = row["src"]
            dst = row["dst"]
            # Use distance (if available) or default to 1.0
            weight = float(row.get("distance", 1.0))
            G.add_edge(src, dst, weight=weight)
        
        # Get node and edge counts (convert to list to avoid iterator issues)
        nodes = list(G.nodes())
        edges = list(G.edges())
        
        # Print graph info
        print(f"Created graph with {len(nodes)} nodes and {len(edges)} edges")
        
        # Create dummy ground truth (all in one community)
        true_labels = [0] * len(nodes)
        print("Warning: No ground truth available for this dataset.")
        
        return G, true_labels, False # Indicate no ground truth
    except Exception as e:
        print(f"Error loading graph from Parquet: {e}")
        # Return tiny dummy graph in case of error
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False # Indicate no ground truth on error too


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
    return G, true_labels, True


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
    return G, true_labels, True


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

    return G, true_labels, True


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
    Handles NaN values for metrics where ground truth was unavailable.
    """
    if not benchmark_results:
        print("No benchmark results available to create comparison chart")
        return None

    datasets = [result["dataset"] for result in benchmark_results]
    n_datasets = len(datasets)
    
    # Use non-interactive backend
    plt.switch_backend("Agg")
    
    # Create figure with adjusted size for 9 subplots
    fig, axes = plt.subplots(9, 1, figsize=(14, 48), dpi=300)
    plt.subplots_adjust(hspace=0.55)
    
    # Flatten axes for easier indexing
    ax_list = axes.flatten()
    ax_ari, ax_nmi, ax_homogeneity, ax_completeness, ax_vmeasure, ax_fmi, ax_modularity, ax_time, ax_memory = ax_list
    
    bar_width = 0.35
    r1 = np.arange(n_datasets)
    r2 = [x + bar_width for x in r1]

    # Colors
    nx_color = "#4285F4"
    rx_color = "#34A853"
    grid_color = "#E5E5E5"
    grid_linewidth = 0.5

    # Function to add value labels, skipping NaNs
    def add_labels(ax, rects1, rects2, data1, data2, format_str=".3f"):
        for i, rect in enumerate(rects1):
            height = rect.get_height()
            if not np.isnan(data1[i]): # Check for NaN before adding label
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.02, 
                        f"{data1[i]:{format_str}}", ha="center", va="bottom", 
                        fontweight="bold", fontsize=10)
        for i, rect in enumerate(rects2):
            height = rect.get_height()
            if not np.isnan(data2[i]): # Check for NaN before adding label
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.02, 
                        f"{data2[i]:{format_str}}", ha="center", va="bottom", 
                        fontweight="bold", fontsize=10)

    # --- Quality Metrics --- (Handle potential NaNs)
    # ARI comparison
    ari_nx = [result["nx_ari"] for result in benchmark_results]
    ari_rx_quality = [result["rx_quality_ari"] for result in benchmark_results]
    rects1 = ax_ari.bar(r1, ari_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_ari.bar(r2, ari_rx_quality, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_ari.set_ylabel("Adjusted Rand Index (ARI)", fontweight="bold", fontsize=14)
    ax_ari.set_title("Louvain Quality - ARI (higher is better, NaN if no ground truth)", fontweight="bold", fontsize=16)
    ax_ari.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_ari.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_ari.set_ylim(0, 1.05)
    ax_ari.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_ari.set_axisbelow(True)
    add_labels(ax_ari, rects1, rects2, ari_nx, ari_rx_quality)
    ax_ari.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # NMI comparison
    nmi_nx = [result["nx_nmi"] for result in benchmark_results]
    nmi_rx_quality = [result["rx_quality_nmi"] for result in benchmark_results]
    rects1 = ax_nmi.bar(r1, nmi_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_nmi.bar(r2, nmi_rx_quality, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_nmi.set_ylabel("Normalized Mutual Info (NMI)", fontweight="bold", fontsize=14)
    ax_nmi.set_title("Louvain Quality - NMI (higher is better, NaN if no ground truth)", fontweight="bold", fontsize=16)
    ax_nmi.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_nmi.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_nmi.set_ylim(0, 1.05)
    ax_nmi.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_nmi.set_axisbelow(True)
    add_labels(ax_nmi, rects1, rects2, nmi_nx, nmi_rx_quality)
    ax_nmi.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)
    
    # Homogeneity comparison
    homogeneity_nx = [result["nx_homogeneity"] for result in benchmark_results]
    homogeneity_rx = [result["rx_quality_homogeneity"] for result in benchmark_results]
    rects1 = ax_homogeneity.bar(r1, homogeneity_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_homogeneity.bar(r2, homogeneity_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_homogeneity.set_ylabel("Homogeneity Score", fontweight="bold", fontsize=14)
    ax_homogeneity.set_title("Louvain Quality - Homogeneity (higher is better, NaN if no ground truth)", fontweight="bold", fontsize=16)
    ax_homogeneity.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_homogeneity.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_homogeneity.set_ylim(0, 1.05)
    ax_homogeneity.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_homogeneity.set_axisbelow(True)
    add_labels(ax_homogeneity, rects1, rects2, homogeneity_nx, homogeneity_rx)
    ax_homogeneity.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # Completeness comparison
    completeness_nx = [result["nx_completeness"] for result in benchmark_results]
    completeness_rx = [result["rx_quality_completeness"] for result in benchmark_results]
    rects1 = ax_completeness.bar(r1, completeness_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_completeness.bar(r2, completeness_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_completeness.set_ylabel("Completeness Score", fontweight="bold", fontsize=14)
    ax_completeness.set_title("Louvain Quality - Completeness (higher is better, NaN if no ground truth)", fontweight="bold", fontsize=16)
    ax_completeness.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_completeness.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_completeness.set_ylim(0, 1.05)
    ax_completeness.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_completeness.set_axisbelow(True)
    add_labels(ax_completeness, rects1, rects2, completeness_nx, completeness_rx)
    ax_completeness.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # V-Measure comparison
    vmeasure_nx = [result["nx_v_measure"] for result in benchmark_results]
    vmeasure_rx = [result["rx_quality_v_measure"] for result in benchmark_results]
    rects1 = ax_vmeasure.bar(r1, vmeasure_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_vmeasure.bar(r2, vmeasure_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_vmeasure.set_ylabel("V-Measure Score", fontweight="bold", fontsize=14)
    ax_vmeasure.set_title("Louvain Quality - V-Measure (higher is better, NaN if no ground truth)", fontweight="bold", fontsize=16)
    ax_vmeasure.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_vmeasure.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_vmeasure.set_ylim(0, 1.05)
    ax_vmeasure.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_vmeasure.set_axisbelow(True)
    add_labels(ax_vmeasure, rects1, rects2, vmeasure_nx, vmeasure_rx)
    ax_vmeasure.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # FMI comparison
    fmi_nx = [result["nx_fmi"] for result in benchmark_results]
    fmi_rx = [result["rx_quality_fmi"] for result in benchmark_results]
    rects1 = ax_fmi.bar(r1, fmi_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_fmi.bar(r2, fmi_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_fmi.set_ylabel("Fowlkes–Mallows Index (FMI)", fontweight="bold", fontsize=14)
    ax_fmi.set_title("Louvain Quality - FMI (higher is better, NaN if no ground truth)", fontweight="bold", fontsize=16)
    ax_fmi.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_fmi.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_fmi.set_ylim(0, 1.05)
    ax_fmi.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_fmi.set_axisbelow(True)
    add_labels(ax_fmi, rects1, rects2, fmi_nx, fmi_rx)
    ax_fmi.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # --- Modularity --- (Always applicable)
    modularity_nx = [result["nx_modularity"] for result in benchmark_results]
    modularity_rx = [result["rx_modularity"] for result in benchmark_results]
    # Determine y-limits, handling potential NaNs in input (though modularity shouldn't be NaN)
    min_mod = np.nanmin([np.nanmin(modularity_nx), np.nanmin(modularity_rx), 0])
    max_mod = np.nanmax([np.nanmax(modularity_nx), np.nanmax(modularity_rx)]) * 1.1
    
    rects1 = ax_modularity.bar(r1, modularity_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_modularity.bar(r2, modularity_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_modularity.set_ylabel("Modularity Score", fontweight="bold", fontsize=14)
    ax_modularity.set_title("Louvain Quality - Modularity (higher is better)", fontweight="bold", fontsize=16)
    ax_modularity.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_modularity.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_modularity.set_ylim(min_mod, max_mod) 
    ax_modularity.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_modularity.set_axisbelow(True)
    # Use the add_labels function for modularity as well
    add_labels(ax_modularity, rects1, rects2, modularity_nx, modularity_rx)
    ax_modularity.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # --- Performance Metrics --- (Always applicable)
    # Execution time comparison
    time_nx = [result["nx_elapsed"] for result in benchmark_results]
    time_rx_quality = [result["rx_elapsed_quality"] for result in benchmark_results]
    time_nx = [max(t, 1e-10) for t in time_nx]
    time_rx_quality = [max(t, 1e-10) for t in time_rx_quality]
    ax_time.bar(r1, time_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_time.bar(r2, time_rx_quality, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_time.set_ylabel("Execution Time (seconds)", fontweight="bold", fontsize=14)
    ax_time.set_title("Louvain Performance - Execution Time (lower is better)", fontweight="bold", fontsize=16)
    ax_time.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_time.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_time.set_yscale("log")
    ax_time.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_time.set_axisbelow(True)
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}" if x < 0.001 else f"{x:.3g}"))
    for i, v in enumerate(time_nx): ax_time.text(r1[i], v * 1.2, f"{format_time(v)}", ha="center", fontweight="bold", fontsize=10, color="#333333")
    for i, v in enumerate(time_rx_quality): ax_time.text(r2[i], v * 1.2, f"{format_time(v)}", ha="center", fontweight="bold", fontsize=10, color="#333333")
    ax_time.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)
    
    # Memory usage comparison
    memory_nx = [result.get("nx_memory", 0) for result in benchmark_results]
    memory_rx_quality = [result.get("rx_memory_quality", 0) for result in benchmark_results]
    memory_nx = [max(m, 1e-6) for m in memory_nx]
    memory_rx_quality = [max(m, 1e-6) for m in memory_rx_quality]
    ax_memory.bar(r1, memory_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_memory.bar(r2, memory_rx_quality, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_memory.set_ylabel("Memory Usage (MB)", fontweight="bold", fontsize=14)
    ax_memory.set_title("Louvain Performance - Memory Usage (lower is better)", fontweight="bold", fontsize=16)
    ax_memory.set_xticks([r + bar_width/2 for r in range(n_datasets)])
    ax_memory.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
    ax_memory.set_yscale("log")
    ax_memory.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
    ax_memory.set_axisbelow(True)
    ax_memory.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_memory(x)))
    for i, v in enumerate(memory_nx): ax_memory.text(r1[i], v * 1.2, f"{format_memory(v)}", ha="center", fontweight="bold", fontsize=10, color="#333333")
    for i, v in enumerate(memory_rx_quality): ax_memory.text(r2[i], v * 1.2, f"{format_memory(v)}", ha="center", fontweight="bold", fontsize=10, color="#333333")
    ax_memory.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)
    
    # Overall title and layout
    fig.suptitle("Community Detection Algorithm Comparison", fontsize=20, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    
    # Save the chart
    plt.savefig(output_file, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"Enhanced comparison chart saved to '{output_file}'")
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
        # "Cora": load_cora,
        "Facebook": load_facebook,
        "Citeseer": load_citeseer,
        "Email EU Core": load_email_eu_core,
        # Add our new datasets
        "Graph Edges CSV": load_graph_edges_csv,
        "Graph Edges Parquet": load_graph_edges_parquet,
        # "LFR Benchmark (Medium)": lambda: load_lfr(n=5000, mu=0.4, name="LFR Medium"), # Optional medium LFR
        # --- Large Datasets (uncomment selectively to run) ---
        # "Amazon Co-purchase": load_amazon_copurchase, # ~300k nodes
        # "Orkut": load_orkut,                             # ~3M nodes
        # "LiveJournal": load_livejournal,                 # ~4.8M nodes
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
        # Load data and check for ground truth
        nx_graph, true_labels, has_ground_truth = load_func()
        print(f"Loaded {dataset_name} dataset: {len(nx_graph.nodes())} nodes, {len(nx_graph.edges())} edges")
        if has_ground_truth:
            print(f"Ground truth: {len(set(true_labels))} communities")
        else:
            print("Ground truth: Not available")

        # Convert graph
        rx_graph, node_map = convert_nx_to_rx(nx_graph)

        # Initialize results with default values
        nx_communities = []
        nx_memory = 0
        nx_elapsed = 0
        rx_communities_quality = []
        rx_memory_quality = 0
        rx_elapsed_quality = 0

        # Initialize metrics and modularity scores (use NaN for metrics when no ground truth)
        nan_val = float("nan")
        nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = (nan_val,) * 6
        rx_quality_ari, rx_quality_nmi, rx_quality_homogeneity, rx_quality_completeness, rx_quality_v_measure, rx_quality_fmi = (nan_val,) * 6
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
                    nx_communities, nx_memory = run_nx_algorithm(nx_graph_undirected)
                    if nx_communities:
                        # Explicitly use 'weight' attribute for modularity
                        nx_modularity = nx.community.modularity(nx_graph_undirected, nx_communities, weight="weight")
                else:
                    nx_communities, nx_memory = run_nx_algorithm(nx_graph)
                    if nx_communities:
                        # Explicitly use 'weight' attribute for modularity
                        nx_modularity = nx.community.modularity(nx_graph, nx_communities, weight="weight")

                nx_elapsed = time.time() - nx_start
                print(f"NetworkX completed in {format_time(nx_elapsed)}")

                # Calculate NetworkX metrics only if ground truth exists and communities were found
                if has_ground_truth and nx_communities:
                    nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = compare_with_true_labels(nx_communities, true_labels)

            except Exception as e:
                print(f"Error running NetworkX Louvain: {str(e)}")
                # Reset results if NetworkX failed
                nx_communities = []
                nx_memory = 0
                nx_elapsed = 0
                nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = (nan_val,) * 6
                nx_modularity = 0.0
        else:
            print(f"\nSkipping NetworkX Louvain for large graph ({len(nx_graph.nodes())} nodes > {LARGE_GRAPH_THRESHOLD})")
            # Ensure default values are set if NetworkX is skipped
            nx_communities = []
            nx_memory = 0
            nx_elapsed = 0
            nx_ari, nx_nmi, nx_homogeneity, nx_completeness, nx_v_measure, nx_fmi = (nan_val,) * 6
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
                    def weight_fn(edge_payload):
                        # Ensure weight is float, default to 1.0 if None or invalid
                        try:
                            return float(edge_payload) if edge_payload is not None else 1.0
                        except (ValueError, TypeError):
                            return 1.0

                    rx_modularity = rx.modularity(rx_graph, rx_communities_quality, weight_fn=weight_fn)
                except AttributeError:
                    print("Warning: rx.modularity function not found. Ensure the Rust module is rebuilt and the function is exported. Skipping calculation.")
                    rx_modularity = 0.0 # Keep as 0.0 if function not found
                except Exception as e:
                    print(f"Error calculating RustWorkX modularity: {str(e)}")
                    rx_modularity = 0.0 # Keep as 0.0 on other errors
            else:
                rx_modularity = 0.0 # No communities, modularity is 0

            # Calculate RustWorkX metrics only if ground truth exists and communities were found
            if has_ground_truth and rx_communities_quality:
                rx_quality_ari, rx_quality_nmi, rx_quality_homogeneity, rx_quality_completeness, rx_quality_v_measure, rx_quality_fmi = compare_with_true_labels(rx_communities_quality, true_labels, node_map)
            # No 'else' needed here, metrics are already NaN by default if no ground truth

        except Exception as e:
            print(f"Error running RustWorkX Louvain: {str(e)}")
            rx_communities_quality = []
            rx_memory_quality = 0
            rx_elapsed_quality = 0
            rx_quality_ari, rx_quality_nmi, rx_quality_homogeneity, rx_quality_completeness, rx_quality_v_measure, rx_quality_fmi = (nan_val,) * 6
            rx_modularity = 0.0

        print("\nResults:")
        print(f"NetworkX found {len(nx_communities)} communities in {format_time(nx_elapsed)} using {format_memory(nx_memory)}")
        # Limit printing community details for large datasets
        if len(nx_communities) < 50 and nx_communities:
            for i, comm in enumerate(nx_communities):
                print(f"  Community {i+1}: {len(comm)} members")
        print(f"RustWorkX found {len(rx_communities_quality)} communities in {format_time(rx_elapsed_quality)} using {format_memory(rx_memory_quality)}")
        if len(rx_communities_quality) < 50 and rx_communities_quality:
            for i, comm in enumerate(rx_communities_quality):
                print(f"  Community {i+1}: {len(comm)} members")

        # Conditional printing of quality comparison
        print("\nComparison with ground truth:")
        if has_ground_truth:
            # Print metrics only if they are not NaN (i.e., calculated)
            print(f"NetworkX - ARI: {nx_ari:.4f}, NMI: {nx_nmi:.4f}, Homogeneity: {nx_homogeneity:.4f}, Completeness: {nx_completeness:.4f}, V-Measure: {nx_v_measure:.4f}, FMI: {nx_fmi:.4f}")
            print(f"RustWorkX- ARI: {rx_quality_ari:.4f}, NMI: {rx_quality_nmi:.4f}, Homogeneity: {rx_quality_homogeneity:.4f}, Completeness: {rx_quality_completeness:.4f}, V-Measure: {rx_quality_v_measure:.4f}, FMI: {rx_quality_fmi:.4f}")
        else:
            print("Not applicable (no ground truth)")

        print("\nModularity comparison (higher is better):")
        print(f"NetworkX Modularity: {nx_modularity:.4f}")
        print(f"RustWorkX Modularity: {rx_modularity:.4f}")

        print("\nPerformance comparison:")
        # Improved speed comparison logic
        if rx_elapsed_quality > 0:
            if nx_elapsed > 0: # Both ran
                 speedup = nx_elapsed / rx_elapsed_quality
                 print(f"RustWorkX is {speedup:.2f}x faster than NetworkX")
            else: # NX was skipped or too fast
                 print("NetworkX skipped or took negligible time, speedup cannot be calculated.")
        elif nx_elapsed > 0: # Only NX ran (RX failed or was too fast)
             print("RustWorkX failed or took negligible time.")
        else: # Neither ran or both too fast
            print("Execution times too small to compare meaningfully.")


        print("\nMemory usage comparison:")
        print(f"NetworkX: {format_memory(nx_memory)}")
        # Improved memory comparison logic
        if rx_memory_quality > 0:
            if nx_memory > 0: # Both have memory usage
                 memory_ratio = nx_memory / rx_memory_quality
                 print(f"RustWorkX: {format_memory(rx_memory_quality)} ({memory_ratio:.2f}x of NetworkX)")
            else: # Only RX has memory usage
                 print(f"RustWorkX: {format_memory(rx_memory_quality)}")
        elif nx_memory > 0: # Only NX has memory usage
             print("RustWorkX used negligible memory.")
        else: # Neither used significant memory
             print("Memory usage too small to compare meaningfully.")

        try:
            safe_filename = dataset_name.lower().replace(" ", "_").replace("-", "_")
            # Pass has_ground_truth to visualization
            visualize_communities(
                nx_graph, nx_communities, rx_communities_quality,
                node_map, true_labels, dataset_name,
                nx_ari, nx_nmi, nx_elapsed,
                rx_quality_ari, rx_quality_nmi, rx_elapsed_quality,
                nx_memory, rx_memory_quality,
                has_ground_truth, # Pass the flag here
                result_folder=result_folder,
                graph_name=safe_filename
            )
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()

        return {
            "dataset": dataset_name,
            "has_ground_truth": has_ground_truth, # Include this flag in results
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
        # Return default dictionary indicating failure and lack of ground truth
        nan_val = float("nan")
        return {
            "dataset": dataset_name,
            "has_ground_truth": False,
            "nx_ari": nan_val, "nx_nmi": nan_val, "nx_homogeneity": nan_val,
            "nx_completeness": nan_val, "nx_v_measure": nan_val, "nx_fmi": nan_val,
            "nx_modularity": 0.0, "nx_elapsed": 0, "nx_memory": 0,
            "rx_quality_ari": nan_val, "rx_quality_nmi": nan_val, "rx_quality_homogeneity": nan_val,
            "rx_quality_completeness": nan_val, "rx_quality_v_measure": nan_val, "rx_quality_fmi": nan_val,
            "rx_modularity": 0.0, "rx_elapsed_quality": 0, "rx_memory_quality": 0
        }


def visualize_communities(nx_graph, nx_communities, rx_communities_quality,
                            node_map, true_labels, dataset_name,
                            nx_ari, nx_nmi, nx_elapsed,
                            rx_quality_ari, rx_quality_nmi, rx_elapsed_quality,
                            nx_memory, rx_memory_quality,
                            has_ground_truth, # Added flag
                            result_folder="results", graph_name="graph"):
    """Visualize the network with ground truth and detected communities and save to file without displaying."""
    try:
        # Adjust visualization threshold
        if len(nx_graph.nodes()) > 2000:
            print(f"Graph too large to visualize efficiently: {dataset_name} ({len(nx_graph.nodes())} nodes)")
            return
            
        # Simplify visualization for medium graphs
        if 1000 < len(nx_graph.nodes()) <= 2000:
            print(f"Creating simplified visualization for large graph: {dataset_name}")
            nodes = list(nx_graph.nodes())
            sample_size = min(1000, len(nodes))
            sampled_nodes = random.sample(nodes, sample_size)
            nx_graph = nx_graph.subgraph(sampled_nodes)
            # Update node_map and true_labels for sampled graph
            node_map = {k: v for k, v in node_map.items() if k in sampled_nodes}
            if has_ground_truth:
                original_node_indices = {node: i for i, node in enumerate(nodes)}
                true_labels = [true_labels[original_node_indices[node]] for node in sampled_nodes if node in original_node_indices and original_node_indices[node] < len(true_labels)]
            else:
                true_labels = [0] * len(sampled_nodes) # Keep dummy labels if no ground truth
        
        # Ensure result folder exists
        os.makedirs(result_folder, exist_ok=True)
            
        # Use non-interactive backend
        plt.switch_backend("Agg")
        
        # Determine number of subplots based on ground truth availability
        num_subplots = 3 if has_ground_truth else 2
        fig_width = 24 if has_ground_truth else 16 # Adjust width
        
        plt.figure(figsize=(fig_width, 10), dpi=300)
        plt.suptitle(f"Louvain Community Detection Results: {dataset_name}", 
                     fontsize=20, fontweight="bold")
        
        # Prepare community mappings
        rx_quality_node_comm = {}
        for i, comm_indices in enumerate(rx_communities_quality):
            for node_idx in comm_indices:
                original_node = next((k for k, v in node_map.items() if v == node_idx), None)
                if original_node is not None and original_node in nx_graph: # Check if node is in current graph
                    rx_quality_node_comm[original_node] = i
        
        # Prepare ground truth mapping if available
        true_node_comm = {}
        unique_true_labels = []
        if has_ground_truth:
            original_node_ids = list(nx_graph.nodes())
            true_node_comm = {original_node_ids[i]: label for i, label in enumerate(true_labels) if i < len(original_node_ids)}
            unique_true_labels = sorted(list(set(true_labels)))
    
        # Calculate layout
        pos = nx.get_node_attributes(nx_graph, "pos")
        if not pos:
            num_nodes = len(nx_graph.nodes())
            print(f"Calculating layout for {dataset_name} ({num_nodes} nodes)...")
            layout_start = time.time()
            if num_nodes < 50: pos = nx.kamada_kawai_layout(nx_graph)
            elif num_nodes < 300: pos = nx.spring_layout(nx_graph, k=0.3, iterations=100, seed=42)
            else: pos = nx.spring_layout(nx_graph, k=1/np.sqrt(num_nodes), iterations=50, seed=42)
            layout_time = time.time() - layout_start
            print(f"Layout calculated in {format_time(layout_time)}")
            nx.set_node_attributes(nx_graph, pos, "pos")
    
        # Colors
        import matplotlib.colors as mcolors
        distinct_colors = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#46f0f0", "#f03280", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#808080", "#000000"]
        max_comm_id = max(len(unique_true_labels) if has_ground_truth else 0,
                          len(nx_communities),
                          len(rx_communities_quality))
        if max_comm_id > len(distinct_colors):
            cmap = plt.colormaps["tab20"]
            hex_colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, max_comm_id)]
        else:
            hex_colors = distinct_colors[:max_comm_id]
            
        # Function to draw a single subplot
        def draw_subplot(ax_idx, title, communities_map, num_communities, metrics_text=""):
            plt.subplot(1, num_subplots, ax_idx)
            full_title = f"{title} ({num_communities} communities)"
            if metrics_text: full_title += f"\n{metrics_text}"
            plt.title(full_title, fontsize=14, fontweight="bold", loc="center")
            nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, width=0.5, edge_color="#CCCCCC")
            
            # Determine how to iterate through communities based on input type
            if isinstance(communities_map, dict): # Like true_node_comm or rx_quality_node_comm
                unique_comms = sorted(list(set(communities_map.values())))
                for i, comm_id in enumerate(unique_comms):
                    node_list = [node for node, c_id in communities_map.items() if c_id == comm_id and node in nx_graph]
                    if not node_list: continue
                    nx.draw_networkx_nodes(nx_graph, pos, nodelist=node_list,
                                        node_color=hex_colors[i % len(hex_colors)], node_size=200,
                                        edgecolors="white", linewidths=0.5, alpha=0.9)
            elif isinstance(communities_map, list): # Like nx_communities
                 for i, comm_nodes in enumerate(communities_map):
                    comm_nodes_in_graph = [node for node in comm_nodes if node in nx_graph]
                    if not comm_nodes_in_graph: continue
                    nx.draw_networkx_nodes(nx_graph, pos, nodelist=comm_nodes_in_graph,
                                        node_color=hex_colors[i % len(hex_colors)], node_size=200,
                                        edgecolors="white", linewidths=0.5, alpha=0.9)
                                        
            if len(nx_graph.nodes()) < 50:
                nx.draw_networkx_labels(nx_graph, pos, font_size=8, font_color="#333333", font_family="sans-serif")
            plt.axis("off")

        # --- Draw Subplots --- 
        subplot_idx = 1
        if has_ground_truth:
            # Create label mapping for consistent coloring if needed
            label_to_int_map = {label: i for i, label in enumerate(unique_true_labels)}
            # Use the mapped dict for consistent coloring
            true_node_comm_mapped = {node: label_to_int_map[label] for node, label in true_node_comm.items()}
            draw_subplot(subplot_idx, "Ground Truth", true_node_comm_mapped, len(unique_true_labels))
            subplot_idx += 1

        # NetworkX Subplot
        nx_metrics = f"ARI: {nx_ari:.3f}, NMI: {nx_nmi:.3f}\nTime: {format_time(nx_elapsed)}, Mem: {format_memory(nx_memory)}" if has_ground_truth else f"Time: {format_time(nx_elapsed)}, Mem: {format_memory(nx_memory)}"
        # Check if nx_communities is not empty before drawing
        if nx_communities or not has_ground_truth: # Draw even if empty if no ground truth
            draw_subplot(subplot_idx, "NetworkX", nx_communities, len(nx_communities), nx_metrics)
            subplot_idx += 1

        # RustWorkX Subplot
        rx_metrics = f"ARI: {rx_quality_ari:.3f}, NMI: {rx_quality_nmi:.3f}\nTime: {format_time(rx_elapsed_quality)}, Mem: {format_memory(rx_memory_quality)}" if has_ground_truth else f"Time: {format_time(rx_elapsed_quality)}, Mem: {format_memory(rx_memory_quality)}"
        # Check if rx_communities_quality is not empty before drawing
        if rx_communities_quality or not has_ground_truth: # Draw even if empty if no ground truth
            # Use the node->community mapping for RustWorkX drawing
            draw_subplot(subplot_idx, "RustWorkX", rx_quality_node_comm, len(rx_communities_quality), rx_metrics)
            subplot_idx += 1
    
        # Layout and Save
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        filename = os.path.join(result_folder, f"{graph_name}_louvain_comparison.png")
        plt.savefig(filename, bbox_inches="tight", dpi=300, facecolor="white", format="png")
        print(f"Saved visualization image to: {filename} (PNG)")
        plt.close("all")
                
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
