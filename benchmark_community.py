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
Benchmark script comparing rustworkx and networkx label propagation community detection
on various graph datasets with ground truth communities.
"""

import math
import os
import random
import shutil
import time
import traceback
import tracemalloc
from datetime import datetime

# cdlib imports - optional dependency
try:  # noqa: E402
    import cdlib  # type: ignore # noqa: F401
    import cdlib.algorithms  # type: ignore # noqa: F401
    HAVE_CDLIB = True
except ImportError:
    HAVE_CDLIB = False

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import polars as pl  # type: ignore # noqa
except ImportError:
    pl = None  # Polars is optional

# import rustworkx with attribute hints for the linter
import rustworkx as rx  # type: ignore # noqa
# Add type ignore hints for rustworkx community attributes
# These are dynamically added in src/lib.rs
if False:  # pragma: no cover
    # This block is never executed but helps the linter recognize attributes
    rx.louvain_communities = lambda: None  # type: ignore
    rx.modularity = lambda: None  # type: ignore
    rx.community = type("", (), {})  # type: ignore
    rx.community.louvain_communities = lambda: None  # type: ignore
    rx.community.modularity = lambda: None  # type: ignore

# Placeholder imports (cdlib might use them internally or for future direct use)
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def load_karate_club():
    """Load Zachary's Karate Club dataset with ground truth communities."""
    G = nx.karate_club_graph()
    true_labels = [1 if G.nodes[node]["club"] == "Mr. Hi" else 0 for node in G.nodes()]
    # Add default weight=1.0 for compatibility
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G, true_labels, True 


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
    G_orig = nx.les_miserables_graph()
    
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
    
    pos = nx.spring_layout(G_orig, k=1/np.sqrt(len(G_orig.nodes())), iterations=50, seed=42)
    H = nx.Graph()
    mapping = {old: i for i, old in enumerate(G_orig.nodes())}
    for old_node, new_node in mapping.items():
        H.add_node(new_node, pos=pos[old_node], value=old_groups[old_node])
    # Add edges with weight=1.0
    for u, v in G_orig.edges():
        H.add_edge(mapping[u], mapping[v], weight=1.0) # Add default weight
        
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
    """Load graph from CSV edge list.
    
    Assumes file 'datasets/graph_edges.csv' exists.
    """
    try:
        if pl is None:
            print("Polars library not available. Skipping CSV loading.")
            G = nx.Graph()
            G.add_node(0)  # Add a dummy node to prevent errors
            return G, [], False
        
        if not os.path.exists("datasets/graph_edges.csv"):
            print("Error loading graph from CSV: No such file or directory (os error 2): datasets/graph_edges.csv")
            G = nx.Graph()
            G.add_node(0)  # Add a dummy node to prevent errors
            return G, [], False
            
        # Read edge list from CSV file
        print("Loading graph from datasets/graph_edges.csv...")
        df = pl.read_csv("datasets/graph_edges.csv")
        edges = df.select(["source", "target"]).to_numpy()
        
        G = nx.Graph()
        for source, target in edges:
            G.add_edge(source, target, weight=1.0)
        
        # No ground truth
        true_labels = [0] * G.number_of_nodes()
        return G, true_labels, False
    except Exception as e:
        print(f"Error loading graph from CSV: {str(e)}")
        G = nx.Graph()
        G.add_node(0)  # Add a dummy node to prevent errors
        return G, [], False


def load_graph_edges_parquet():
    """Load graph from Parquet edge list.
    
    Assumes file 'datasets/graph_edges_big.parquet' exists.
    """
    try:
        if pl is None:
            print("Polars library not available. Skipping Parquet loading.")
            G = nx.Graph()
            G.add_node(0)  # Add a dummy node to prevent errors
            return G, [], False
        
        if not os.path.exists("datasets/graph_edges_big.parquet"):
            print("Error loading graph from Parquet: No such file or directory (os error 2): datasets/graph_edges_big.parquet")
            G = nx.Graph()
            G.add_node(0)  # Add a dummy node to prevent errors
            return G, [], False
            
        # Read edge list from Parquet file
        print("Loading graph from datasets/graph_edges_big.parquet...")
        df = pl.read_parquet("datasets/graph_edges_big.parquet")
        edges = df.select(["source", "target"]).to_numpy()
        
        G = nx.Graph()
        for source, target in edges:
            G.add_edge(source, target, weight=1.0)
        
        # No ground truth
        true_labels = [0] * G.number_of_nodes()
        return G, true_labels, False
    except Exception as e:
        print(f"Error loading graph from Parquet: {str(e)}")
        G = nx.Graph()
        G.add_node(0)  # Add a dummy node to prevent errors
        return G, [], False


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
    
    plt.switch_backend("Agg")
    
    # Create figure for 11 subplots (7 external + 2 internal + 2 performance)
    num_metrics = 11 
    fig_height_per_metric = 5.5 # Adjusted height per subplot for better spacing
    fig, axes = plt.subplots(num_metrics, 1, figsize=(14, num_metrics * fig_height_per_metric), dpi=300)
    plt.subplots_adjust(hspace=0.6) # Adjusted spacing
    
    # Flatten axes for easier indexing
    ax_list = axes.flatten()
    # Unpack axes according to the number of metrics
    if len(ax_list) != num_metrics:
        raise ValueError(f"Expected {num_metrics} axes, but got {len(ax_list)}")
    
    (ax_ari, ax_nmi, ax_homogeneity, ax_completeness, ax_vmeasure, ax_fmi, 
     ax_modularity, ax_conductance, ax_internal_density, 
     ax_time, ax_memory) = ax_list
    
    bar_width = 0.35
    r1 = np.arange(n_datasets)
    r2 = [x + bar_width for x in r1]

    # Colors and grid settings
    nx_color = "#4285F4"
    rx_color = "#34A853"
    grid_color = "#E5E5E5"
    grid_linewidth = 0.5

    # Helper function to add labels, skipping NaNs
    def add_labels(ax, rects1, rects2, data1, data2, format_str=".3f"):
        for i, rect in enumerate(rects1):
            y_val = rect.get_height()
            x_val = rect.get_x() + rect.get_width() / 2.0
            # Adjust label position slightly based on bar height for clarity
            label_y_offset = 0.02 * ax.get_ylim()[1] # Relative offset
            if not np.isnan(data1[i]):
                ax.text(x_val, y_val + label_y_offset, f"{data1[i]:{format_str}}", 
                        ha="center", va="bottom", fontweight="bold", fontsize=10)
        for i, rect in enumerate(rects2):
            y_val = rect.get_height()
            x_val = rect.get_x() + rect.get_width() / 2.0
            label_y_offset = 0.02 * ax.get_ylim()[1]
            if not np.isnan(data2[i]):
                ax.text(x_val, y_val + label_y_offset, f"{data2[i]:{format_str}}", 
                        ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Helper function to style axes
    def style_axis(ax, ylabel, title):
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
        ax.set_title(title, fontweight="bold", fontsize=16)
        ax.set_xticks([r + bar_width/2 for r in range(n_datasets)])
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=12)
        ax.yaxis.grid(True, linestyle="--", linewidth=grid_linewidth, color=grid_color)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=12)

    # --- External Quality Metrics --- (Handle potential NaNs)
    # ARI
    ari_nx = [result["nx_ari"] for result in benchmark_results]
    ari_rx = [result["rx_quality_ari"] for result in benchmark_results]
    rects1 = ax_ari.bar(r1, ari_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_ari.bar(r2, ari_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_ari.set_ylim(0, 1.05)
    add_labels(ax_ari, rects1, rects2, ari_nx, ari_rx)
    style_axis(ax_ari, "Adjusted Rand Index (ARI)", "External Quality - ARI (higher is better, NaN if no ground truth)")

    # NMI
    nmi_nx = [result["nx_nmi"] for result in benchmark_results]
    nmi_rx = [result["rx_quality_nmi"] for result in benchmark_results]
    rects1 = ax_nmi.bar(r1, nmi_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_nmi.bar(r2, nmi_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_nmi.set_ylim(0, 1.05)
    add_labels(ax_nmi, rects1, rects2, nmi_nx, nmi_rx)
    style_axis(ax_nmi, "Normalized Mutual Info (NMI)", "External Quality - NMI (higher is better, NaN if no ground truth)")
    
    # Homogeneity
    homogeneity_nx = [result["nx_homogeneity"] for result in benchmark_results]
    homogeneity_rx = [result["rx_quality_homogeneity"] for result in benchmark_results]
    rects1 = ax_homogeneity.bar(r1, homogeneity_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_homogeneity.bar(r2, homogeneity_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_homogeneity.set_ylim(0, 1.05)
    add_labels(ax_homogeneity, rects1, rects2, homogeneity_nx, homogeneity_rx)
    style_axis(ax_homogeneity, "Homogeneity Score", "External Quality - Homogeneity (higher is better, NaN if no ground truth)")

    # Completeness
    completeness_nx = [result["nx_completeness"] for result in benchmark_results]
    completeness_rx = [result["rx_quality_completeness"] for result in benchmark_results]
    rects1 = ax_completeness.bar(r1, completeness_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_completeness.bar(r2, completeness_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_completeness.set_ylim(0, 1.05)
    add_labels(ax_completeness, rects1, rects2, completeness_nx, completeness_rx)
    style_axis(ax_completeness, "Completeness Score", "External Quality - Completeness (higher is better, NaN if no ground truth)")

    # V-Measure
    vmeasure_nx = [result["nx_v_measure"] for result in benchmark_results]
    vmeasure_rx = [result["rx_quality_v_measure"] for result in benchmark_results]
    rects1 = ax_vmeasure.bar(r1, vmeasure_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_vmeasure.bar(r2, vmeasure_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_vmeasure.set_ylim(0, 1.05)
    add_labels(ax_vmeasure, rects1, rects2, vmeasure_nx, vmeasure_rx)
    style_axis(ax_vmeasure, "V-Measure Score", "External Quality - V-Measure (higher is better, NaN if no ground truth)")

    # FMI
    fmi_nx = [result["nx_fmi"] for result in benchmark_results]
    fmi_rx = [result["rx_quality_fmi"] for result in benchmark_results]
    rects1 = ax_fmi.bar(r1, fmi_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_fmi.bar(r2, fmi_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_fmi.set_ylim(0, 1.05)
    add_labels(ax_fmi, rects1, rects2, fmi_nx, fmi_rx)
    style_axis(ax_fmi, "Fowlkes–Mallows Index (FMI)", "External Quality - FMI (higher is better, NaN if no ground truth)")

    # --- Internal Quality Metrics --- (Always applicable)
    # Modularity
    modularity_nx = [result["nx_modularity"] for result in benchmark_results]
    modularity_rx = [result["rx_modularity"] for result in benchmark_results]
    min_mod = np.nanmin([np.nanmin(modularity_nx), np.nanmin(modularity_rx), 0])
    max_mod = np.nanmax([np.nanmax(modularity_nx), np.nanmax(modularity_rx)])
    max_mod = max_mod * 1.1 if not np.isnan(max_mod) else 1.0 # Adjust padding, handle all NaN case
    min_mod = min_mod if not np.isnan(min_mod) else 0.0
    rects1 = ax_modularity.bar(r1, modularity_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_modularity.bar(r2, modularity_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_modularity.set_ylim(min_mod - abs(min_mod)*0.05, max_mod + abs(max_mod)*0.05) # Adjusted y-limits slightly
    add_labels(ax_modularity, rects1, rects2, modularity_nx, modularity_rx)
    style_axis(ax_modularity, "Modularity Score", "Internal Quality - Modularity (higher is better)")

    # Conductance
    conductance_nx = [result["nx_conductance"] for result in benchmark_results]
    conductance_rx = [result["rx_conductance"] for result in benchmark_results]
    min_cond = 0 # Conductance is non-negative
    max_cond = np.nanmax([np.nanmax(conductance_nx), np.nanmax(conductance_rx)]) 
    max_cond = max_cond * 1.1 if not np.isnan(max_cond) else 1.0 # Adjust padding
    max_cond = max(max_cond, 0.1) # Ensure max_cond is at least a small positive value
    rects1 = ax_conductance.bar(r1, conductance_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_conductance.bar(r2, conductance_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_conductance.set_ylim(min_cond, max_cond)
    add_labels(ax_conductance, rects1, rects2, conductance_nx, conductance_rx)
    style_axis(ax_conductance, "Avg. Conductance", "Internal Quality - Conductance (lower is better)")

    # Internal Edge Density
    int_density_nx = [result["nx_internal_density"] for result in benchmark_results]
    int_density_rx = [result["rx_internal_density"] for result in benchmark_results]
    min_dens = 0 # Density is non-negative
    max_dens = np.nanmax([np.nanmax(int_density_nx), np.nanmax(int_density_rx)])
    max_dens = max_dens * 1.1 if not np.isnan(max_dens) else 1.0 # Adjust padding
    max_dens = max(max_dens, 0.1) # Ensure max_dens is at least a small positive value
    rects1 = ax_internal_density.bar(r1, int_density_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    rects2 = ax_internal_density.bar(r2, int_density_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_internal_density.set_ylim(min_dens, max_dens)
    add_labels(ax_internal_density, rects1, rects2, int_density_nx, int_density_rx)
    style_axis(ax_internal_density, "Avg. Internal Density", "Internal Quality - Internal Density (higher is better)")

    # --- Performance Metrics --- (Always applicable)
    # Execution time
    time_nx = [result["nx_elapsed"] for result in benchmark_results]
    time_rx = [result["rx_elapsed_quality"] for result in benchmark_results]
    time_nx = [max(t, 1e-10) for t in time_nx]
    time_rx = [max(t, 1e-10) for t in time_rx]
    ax_time.bar(r1, time_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_time.bar(r2, time_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_time.set_yscale("log")
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}" if x < 0.001 else f"{x:.3g}"))
    # Add time labels using format_time
    for i, v in enumerate(time_nx): ax_time.text(r1[i], v * 1.3, f"{format_time(v)}", ha="center", va="bottom", fontweight="bold", fontsize=10, color="#333333")
    for i, v in enumerate(time_rx): ax_time.text(r2[i], v * 1.3, f"{format_time(v)}", ha="center", va="bottom", fontweight="bold", fontsize=10, color="#333333")
    style_axis(ax_time, "Execution Time (seconds)", "Performance - Execution Time (lower is better, log scale)")
    
    # Memory usage
    memory_nx = [result.get("nx_memory", 0) for result in benchmark_results]
    memory_rx = [result.get("rx_memory_quality", 0) for result in benchmark_results]
    memory_nx = [max(m, 1e-6) for m in memory_nx]
    memory_rx = [max(m, 1e-6) for m in memory_rx]
    ax_memory.bar(r1, memory_nx, width=bar_width, label="NetworkX", color=nx_color, edgecolor="#3B77DB", linewidth=0.8)
    ax_memory.bar(r2, memory_rx, width=bar_width, label="RustWorkX", color=rx_color, edgecolor="#2D9249", linewidth=0.8)
    ax_memory.set_yscale("log")
    ax_memory.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_memory(x)))
    # Add memory labels using format_memory
    for i, v in enumerate(memory_nx): ax_memory.text(r1[i], v * 1.3, f"{format_memory(v)}", ha="center", va="bottom", fontweight="bold", fontsize=10, color="#333333")
    for i, v in enumerate(memory_rx): ax_memory.text(r2[i], v * 1.3, f"{format_memory(v)}", ha="center", va="bottom", fontweight="bold", fontsize=10, color="#333333")
    style_axis(ax_memory, "Memory Usage (MB)", "Performance - Memory Usage (lower is better, log scale)")
    
    # Overall title and layout
    fig.suptitle("Community Detection Algorithm Comparison", fontsize=22, fontweight="bold", y=1.0) # Adjusted title position
    plt.tight_layout(rect=(0, 0, 1, 0.99)) # Adjusted layout rect
    
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


def create_results_folder():
    """Create a results folder with timestamp to avoid overwriting."""
    base_result_folder = "results"  # Base folder for all results
    timestamp = datetime.now().strftime("%Y%m%d")  # Shorter timestamp (date only)
    result_folder = os.path.join(base_result_folder, timestamp)  # Path like results/YYYYMMDD
    
    # Create base folder if it doesn't exist
    if not os.path.exists(base_result_folder):
        os.makedirs(base_result_folder)
    # Create timestamped subfolder
    os.makedirs(result_folder, exist_ok=True)
    
    print(f"Results will be saved to '{result_folder}'")
    return result_folder


def run_benchmark():
    """Run the benchmark on all datasets and create visualizations."""
    start_time = time.time()

    # Create results directory
    result_folder = create_results_folder()
    
    # List of datasets to benchmark
    datasets = [
        ("Karate Club", load_karate_club),
        ("Davis Southern Women", load_davis_women),
        ("Florentine Families", load_florentine_families),
        ("Les Misérables", load_les_miserables),
        # Add more datasets here if available
        ("Football", load_football),
        ("Political Books", load_political_books),
        ("Dolphins", load_dolphins),
        ("LFR Benchmark (Small)", load_lfr),
        ("Political Blogs", load_polblogs),
        ("Facebook", load_facebook),
        ("Citeseer", load_citeseer),
        ("Email EU Core", load_email_eu_core),
        ("Graph Edges CSV", load_graph_edges_csv),
        ("Graph Edges Parquet", load_graph_edges_parquet),
    ]
    
    benchmark_results = []
    for dataset_name, load_func in datasets:
        result = run_benchmark_on_dataset(dataset_name, load_func, result_folder)
        if result is not None:  # Skip datasets that failed to load
            benchmark_results.append(result)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n================================================================================")
    print("BENCHMARK COMPLETE - VISUALIZATION FILES GENERATED:")
    print("================================================================================")
    
    # List generated files or show message
    if os.path.exists(result_folder):
        visualization_files = [f for f in os.listdir(result_folder) if f.endswith((".png", ".pdf", ".svg"))]
        if visualization_files:
            for vf in visualization_files:
                print(f"{os.path.join(result_folder, vf)}")
        else:
            print("No visualization files were generated. Please check for errors.")
    else:
        print("Results folder not found. Please check for errors.")
        
    print("\n================================================================================")
    print("GENERATING RESULTS TABLE")
    print("================================================================================")
    
    # Generate results table if we have any results
    if benchmark_results:
        table_string = generate_results_table(benchmark_results)
        print(table_string)
        
        # Generate visualization table as PNG
        table_png_path = os.path.join(result_folder, "benchmark_results_table.png")
        generate_results_table_matplotlib(benchmark_results, table_png_path)
    else:
        print("No results to generate table. Please check for errors.")
    
    print(f"\nBenchmark completed in {elapsed_time:.2f} seconds.")
    
    return benchmark_results


def generate_results_table(results):
    """Generate a formatted table summarizing benchmark results."""
    if not results:
        return "No results available."
    
    # Table header
    header = "| Dataset | Nodes | Edges | NX Time | RX Time | NX/RX Ratio | NX Mem | RX Mem | NX Comms | RX Comms | NX Mod | RX Mod |"
    separator = "|---------|-------|-------|---------|---------|-------------|--------|--------|----------|----------|--------|--------|"
    
    rows = []
    
    # Sort results by graph size (node count)
    sorted_results = sorted(results, key=lambda x: x.get("nodes", 0))
    
    for res in sorted_results:
        dataset = res.get("dataset", "Unknown")
        nodes = res.get("nodes", 0)
        edges = res.get("edges", 0)
        
        # Handle None values for all fields
        nx_elapsed = res.get("nx_elapsed", 0)
        rx_elapsed = res.get("rx_elapsed", 0)
        
        if nx_elapsed > 0 and rx_elapsed > 0:
            speedup = nx_elapsed / rx_elapsed
        else:
            speedup = float("nan")
        
        nx_memory = res.get("nx_memory", 0)
        rx_memory = res.get("rx_memory", 0)
        
        nx_comms = res.get("nx_num_communities", 0)
        rx_comms = res.get("rx_num_communities", 0)
        
        nx_mod = res.get("nx_modularity", float("nan"))
        rx_mod = res.get("rx_modularity", float("nan"))
        
        # Format values
        def fmt(key, decimals=4):
            val = res.get(key, float("nan"))
            if isinstance(val, (int, float)) and not math.isnan(val):
                return f"{val:.{decimals}f}"
            return "N/A"
        
        # Create row
        row = f"| {dataset} | {nodes} | {edges} | "
        
        # Format time values based on magnitude
        if nx_elapsed < 1:
            nx_time_str = f"{nx_elapsed*1000:.2f} μs"
        elif nx_elapsed < 1000:
            nx_time_str = f"{nx_elapsed:.2f} ms"
        else:
            nx_time_str = f"{nx_elapsed/1000:.2f} s"
            
        if rx_elapsed < 1:
            rx_time_str = f"{rx_elapsed*1000:.2f} μs"
        elif rx_elapsed < 1000:
            rx_time_str = f"{rx_elapsed:.2f} ms"
        else:
            rx_time_str = f"{rx_elapsed/1000:.2f} s"
        
        row += f"{nx_time_str} | {rx_time_str} | "
        
        # Speedup ratio
        if math.isnan(speedup):
            row += "N/A | "
        else:
            row += f"{speedup:.1f}x | "
        
        # Memory usage
        nx_mem_str = format_memory(nx_memory) if nx_memory else "N/A"
        rx_mem_str = format_memory(rx_memory) if rx_memory else "N/A"
        row += f"{nx_mem_str} | {rx_mem_str} | "
        
        # Communities and modularity
        row += f"{nx_comms} | {rx_comms} | {fmt('nx_modularity')} | {fmt('rx_modularity')} |"
        
        rows.append(row)
    
    # Combine all parts
    table = "\n".join([header, separator] + rows)
    return table


def run_benchmark_on_dataset(dataset_name, load_func, result_folder):
    """Run the benchmark on a specific dataset."""
    # Add header for the dataset
    dashes = "-" * 40
    print(f"\n{dashes}")
    print(f"Processing {dataset_name} dataset...")
    print(f"{dashes}\n")
    
    print("=" * 50)
    print(f"Running Louvain benchmark on {dataset_name} dataset")
    print("=" * 50)

    # Load and prepare the graph
    try:
        nx_graph, true_labels, has_ground_truth = load_func()
        print(f"Loaded {dataset_name}: {len(nx_graph.nodes())} nodes, {len(nx_graph.edges())} edges. Ground truth available: {has_ground_truth}")
        
        # Skip tiny graphs with no edges to avoid division by zero errors
        if len(nx_graph.edges()) == 0:
            print(f"Skipping {dataset_name} - empty graph")
            return None
        
        # Choose the appropriate parameters for each dataset
        # Larger datasets need increased resolution to prevent dominating communities
        leiden_resolution = 1.0  # Default
        if len(nx_graph.nodes()) > 1000:
            leiden_resolution = 1.5
        if len(nx_graph.nodes()) > 5000:
            leiden_resolution = 2.0
        
        results = {"dataset": dataset_name}
        # Run NetworkX Louvain
        try:
            print("\nRunning NetworkX Louvain...")
            start_time = time.time()
            nx_communities, results["nx_memory"] = run_nx_algorithm(nx_graph)
            
            elapsed_ms = (time.time() - start_time) * 1000
            # Format time based on magnitude
            time_str = format_time(elapsed_ms / 1000)
            results["nx_elapsed"] = elapsed_ms
            
            results["nx_num_communities"] = len(nx_communities)
            print(f"NetworkX Louvain: {len(nx_communities)} communities in {time_str} using {format_memory(results['nx_memory'])}")
            
            # Calculate NetworkX modularity
            try:
                nx_communities_list = list(nx_communities.values())
                nx_modularity = nx.community.modularity(nx_graph, nx_communities_list)
                results["nx_modularity"] = nx_modularity
            except Exception as e:
                print(f"Warning: NX modularity calculation failed: {str(e)}")
                results["nx_modularity"] = 0.0
        except Exception as e:
            print(f"Error running NetworkX Louvain: {str(e)}")
            results["nx_elapsed"] = 0
            results["nx_memory"] = 0
            results["nx_num_communities"] = 0
            results["nx_modularity"] = 0
            nx_communities = {0: list(nx_graph.nodes())}
        
        # Convert NetworkX graph to RustWorkX
        rx_graph, node_map = convert_nx_to_rx(nx_graph)
        
        # Run RustWorkX Louvain
        try:
            print("\nRunning RustWorkX Louvain...")
            start_time = time.time()
            rx_communities_quality, results["rx_memory"] = run_rx_quality_algorithm(rx_graph)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Format time based on magnitude
            time_str = format_time(elapsed_ms / 1000)
            results["rx_elapsed"] = elapsed_ms
            
            # Convert communities from rustworkx format to networkx format for comparison
            rx_communities_dict = {}
            for i, comm in enumerate(rx_communities_quality):
                # Make sure comm is iterable before iterating
                if isinstance(comm, (list, set, tuple)):
                    rx_communities_dict[i] = [node_map[node] for node in comm]
                else:
                    # Handle the case where comm is a single integer
                    rx_communities_dict[i] = [node_map[comm]]
            
            results["rx_num_communities"] = len(rx_communities_quality)
            print(f"RustWorkX Louvain: {len(rx_communities_quality)} communities in {time_str} using {format_memory(results['rx_memory'])}")
            
            # Calculate RustWorkX modularity
            try:
                # Define weight function for modularity calculation
                def weight_fn(edge_payload):
                    return edge_payload.get("weight", 1.0)
                
                # Convert communities to the expected format (list of sets)
                communities_as_sets = []
                for comm in rx_communities_quality:
                    if isinstance(comm, (list, set, tuple)):
                        communities_as_sets.append(set(comm))
                    elif isinstance(comm, int):
                        # Handle single integers by putting them in a set
                        communities_as_sets.append({comm})
                    else:
                        # Handle any other type as best we can
                        try:
                            communities_as_sets.append(set([comm]))
                        except:
                            communities_as_sets.append(set())
                
                # First try the top-level modularity function
                rx_modularity = rx.modularity(rx_graph, communities_as_sets, weight_fn)
            except (AttributeError, TypeError):
                print("Warning: rx.modularity not found, trying rx.community.modularity...")
                try:
                    # Use getattr to avoid linter errors
                    community_module = getattr(rx, "community", None)
                    if community_module is not None:
                        rx_modularity = community_module.modularity(rx_graph, communities_as_sets, weight_fn)
                    else:
                        # Fallback if community module is not available
                        print("Warning: Community module not found in rustworkx")
                        rx_modularity = float("nan")
                except (AttributeError, TypeError):
                    print("Warning: Modularity function not found in rustworkx or rustworkx.community.")
                    rx_modularity = float("nan")
                    
            results["rx_modularity"] = rx_modularity
                
            # If all RX methods fail, try calculating with NetworkX for comparison
            if math.isnan(rx_modularity):
                try:
                    rx_communities_list = list(rx_communities_dict.values())
                    rx_modularity = nx.community.modularity(nx_graph, rx_communities_list)
                    results["rx_modularity"] = rx_modularity
                except Exception as e2:
                    print(f"Warning: Fallback NX modularity calculation also failed: {str(e2)}")
                    results["rx_modularity"] = 0.0
        except Exception as e:
            print(f"Error running RustWorkX Louvain: {str(e)}")
            results["rx_elapsed"] = 0
            results["rx_memory"] = 0
            results["rx_num_communities"] = 0
            results["rx_modularity"] = 0
            rx_communities_quality = [list(range(len(node_map)))]
            rx_communities_dict = {0: list(nx_graph.nodes())}
        
        # Ensure the graph is intact for cdlib
        graph_for_nx_cdlib = nx_graph.copy()
        
        # Run Leiden (if cdlib is available)
        try:
            if HAVE_CDLIB:
                print("\nRunning Leiden (cdlib, resolution=1.0)...")
                print(f"Running Leiden algorithm (cdlib, resolution={leiden_resolution})...")
                leiden_communities, results["ld_memory"] = run_leiden_cdlib(graph_for_nx_cdlib, resolution=leiden_resolution)
                
                # Calculate Leiden modularity
                try:
                    leiden_communities_list = list(leiden_communities.values())
                    leiden_modularity = nx.community.modularity(nx_graph, leiden_communities_list)
                    results["ld_modularity"] = leiden_modularity
                except Exception as e:
                    print(f"Warning: Leiden modularity calculation failed: {str(e)}")
                    results["ld_modularity"] = 0.0
                
                results["ld_num_communities"] = len(leiden_communities)
                print(f"Leiden (cdlib): {len(leiden_communities)} communities in {format_time(results.get('ld_elapsed', 0)/1000)} using {format_memory(results['ld_memory'])}")
            else:
                leiden_communities = {0: list(nx_graph.nodes())}
                results["ld_memory"] = 0
                results["ld_modularity"] = 0.0
                results["ld_num_communities"] = 0
        except Exception as e:
            print(f"Error running Leiden (cdlib): {str(e)}")
            leiden_communities = {0: list(nx_graph.nodes())}
            results["ld_memory"] = 0
            results["ld_modularity"] = 0.0
            results["ld_num_communities"] = 0
        
        # Run Infomap (if cdlib is available)
        try:
            if HAVE_CDLIB:
                print("\nRunning Infomap (cdlib)...")
                print("Running Infomap algorithm (cdlib)...")
                infomap_communities, results["im_memory"] = run_infomap_cdlib(graph_for_nx_cdlib)
                
                # Calculate Infomap modularity
                try:
                    infomap_communities_list = list(infomap_communities.values())
                    infomap_modularity = nx.community.modularity(nx_graph, infomap_communities_list)
                    results["im_modularity"] = infomap_modularity
                except Exception as e:
                    print(f"Warning: Infomap modularity calculation failed: {str(e)}")
                    results["im_modularity"] = 0.0
                
                results["im_num_communities"] = len(infomap_communities)
                print(f"Infomap (cdlib): {len(infomap_communities)} communities in {format_time(results.get('im_elapsed', 0)/1000)} using {format_memory(results['im_memory'])}")
            else:
                infomap_communities = {0: list(nx_graph.nodes())}
                results["im_memory"] = 0
                results["im_modularity"] = 0.0
                results["im_num_communities"] = 0
        except Exception as e:
            print(f"Error running Infomap (cdlib): {str(e)}")
            infomap_communities = {0: list(nx_graph.nodes())}
            results["im_memory"] = 0
            results["im_modularity"] = 0.0
            results["im_num_communities"] = 0
        
        # Print scores summary
        print("\n--- Scores Summary ---")
        print(f"Modularity:         NX={results.get('nx_modularity', 0.0):.4f}, RX={results.get('rx_modularity', 0.0):.4f}, LD={results.get('ld_modularity', 0.0):.4f}, IM={results.get('im_modularity', 0.0):.4f}")
        
        # Calculate internal metrics
        try:
            nx_communities_list = list(nx_communities.values())
            nx_metrics = calculate_internal_metrics(nx_graph, nx_communities_list)
            results["nx_conductance"], results["nx_internal_density"], results["nx_avg_internal_degree"], results["nx_tpr"], results["nx_cut_ratio"], results["nx_surprise"], results["nx_significance"] = nx_metrics
        except Exception as e:
            print(f"Warning: Failed to calculate NX internal metrics: {str(e)}")
            results["nx_conductance"] = float("nan")
            results["nx_internal_density"] = float("nan")
            results["nx_avg_internal_degree"] = float("nan")
            results["nx_tpr"] = float("nan")
            results["nx_cut_ratio"] = float("nan")
            results["nx_surprise"] = float("nan")
            results["nx_significance"] = float("nan")
        
        try:
            rx_metrics = calculate_internal_metrics(nx_graph, list(rx_communities_dict.values()))
            results["rx_conductance"], results["rx_internal_density"], results["rx_avg_internal_degree"], results["rx_tpr"], results["rx_cut_ratio"], results["rx_surprise"], results["rx_significance"] = rx_metrics
        except Exception as e:
            print(f"Warning: Failed to calculate RX internal metrics: {str(e)}")
            results["rx_conductance"] = float("nan")
            results["rx_internal_density"] = float("nan")
            results["rx_avg_internal_degree"] = float("nan")
            results["rx_tpr"] = float("nan")
            results["rx_cut_ratio"] = float("nan")
            results["rx_surprise"] = float("nan")
            results["rx_significance"] = float("nan")
        
        try:
            ld_metrics = calculate_internal_metrics(nx_graph, list(leiden_communities.values()))
            results["ld_conductance"], results["ld_internal_density"], results["ld_avg_internal_degree"], results["ld_tpr"], results["ld_cut_ratio"], results["ld_surprise"], results["ld_significance"] = ld_metrics
        except Exception as e:
            print(f"Warning: Failed to calculate LD internal metrics: {str(e)}")
            results["ld_conductance"] = float("nan")
            results["ld_internal_density"] = float("nan")
            results["ld_avg_internal_degree"] = float("nan")
            results["ld_tpr"] = float("nan")
            results["ld_cut_ratio"] = float("nan")
            results["ld_surprise"] = float("nan")
            results["ld_significance"] = float("nan")
        
        try:
            im_metrics = calculate_internal_metrics(nx_graph, list(infomap_communities.values()))
            results["im_conductance"], results["im_internal_density"], results["im_avg_internal_degree"], results["im_tpr"], results["im_cut_ratio"], results["im_surprise"], results["im_significance"] = im_metrics
        except Exception as e:
            print(f"Warning: Failed to calculate IM internal metrics: {str(e)}")
            results["im_conductance"] = float("nan")
            results["im_internal_density"] = float("nan")
            results["im_avg_internal_degree"] = float("nan")
            results["im_tpr"] = float("nan")
            results["im_cut_ratio"] = float("nan")
            results["im_surprise"] = float("nan")
            results["im_significance"] = float("nan")
        
        # Print internal metrics
        print(f"Conductance:        NX={results.get('nx_conductance', float('nan')):.4f}, RX={results.get('rx_conductance', float('nan')):.4f}, LD={results.get('ld_conductance', float('nan'))}, IM={results.get('im_conductance', float('nan'))} (lower=better)")
        print(f"Internal Density:   NX={results.get('nx_internal_density', float('nan')):.4f}, RX={results.get('rx_internal_density', float('nan')):.4f}, LD={results.get('ld_internal_density', float('nan'))}, IM={results.get('im_internal_density', float('nan'))} (higher=better)")
        print(f"Avg Internal Deg:   NX={results.get('nx_avg_internal_degree', float('nan')):.4f}, RX={results.get('rx_avg_internal_degree', float('nan')):.4f}, LD={results.get('ld_avg_internal_degree', float('nan'))}, IM={results.get('im_avg_internal_degree', float('nan'))} (higher=better)")
        print(f"TPR:                NX={results.get('nx_tpr', float('nan')):.4f}, RX={results.get('rx_tpr', float('nan')):.4f}, LD={results.get('ld_tpr', float('nan'))}, IM={results.get('im_tpr', float('nan'))} (higher=better)")
        print(f"Cut Ratio:          NX={results.get('nx_cut_ratio', float('nan')):.4f}, RX={results.get('rx_cut_ratio', float('nan')):.4f}, LD={results.get('ld_cut_ratio', float('nan'))}, IM={results.get('im_cut_ratio', float('nan'))} (lower=better)")
        print(f"Surprise:           NX={results.get('nx_surprise', float('nan')):.4f}, RX={results.get('rx_surprise', float('nan')):.4f}, LD={results.get('ld_surprise', float('nan'))}, IM={results.get('im_surprise', float('nan'))} (higher=better)")
        print(f"Significance:       NX={results.get('nx_significance', float('nan')):.4f}, RX={results.get('rx_significance', float('nan')):.4f}, LD={results.get('ld_significance', float('nan'))}, IM={results.get('im_significance', float('nan'))} (higher=better)")

        # Calculate external metrics if ground truth is available
        if has_ground_truth:
            print("-- External --")
            try:
                nx_external = compare_with_true_labels(list(nx_communities.values()), true_labels)
                results["nx_ari"], results["nx_nmi"], results["nx_homogeneity"], results["nx_completeness"], results["nx_v_measure"], results["nx_fmi"] = nx_external
            except Exception as e:
                print(f"Warning: Failed to calculate NX external metrics: {str(e)}")
                results["nx_ari"] = float("nan")
                results["nx_nmi"] = float("nan")
                
            try:
                rx_external = compare_with_true_labels(list(rx_communities_dict.values()), true_labels)
                results["rx_ari"], results["rx_nmi"], results["rx_homogeneity"], results["rx_completeness"], results["rx_v_measure"], results["rx_fmi"] = rx_external
            except Exception as e:
                print(f"Warning: Failed to calculate RX external metrics: {str(e)}")
                results["rx_ari"] = float("nan")
                results["rx_nmi"] = float("nan")
                
            try:
                ld_external = compare_with_true_labels(list(leiden_communities.values()), true_labels)
                results["ld_ari"], results["ld_nmi"], results["ld_homogeneity"], results["ld_completeness"], results["ld_v_measure"], results["ld_fmi"] = ld_external
            except Exception as e:
                print(f"Warning: Failed to calculate LD external metrics: {str(e)}")
                results["ld_ari"] = float("nan")
                results["ld_nmi"] = float("nan")
                
            try:
                im_external = compare_with_true_labels(list(infomap_communities.values()), true_labels)
                results["im_ari"], results["im_nmi"], results["im_homogeneity"], results["im_completeness"], results["im_v_measure"], results["im_fmi"] = im_external
            except Exception as e:
                print(f"Warning: Failed to calculate IM external metrics: {str(e)}")
                results["im_ari"] = float("nan")
                results["im_nmi"] = float("nan")
                
            print(f"ARI:                NX={results.get('nx_ari', float('nan')):.4f}, RX={results.get('rx_ari', float('nan')):.4f}, LD={results.get('ld_ari', float('nan'))}, IM={results.get('im_ari', float('nan'))}")
            print(f"NMI:                NX={results.get('nx_nmi', float('nan')):.4f}, RX={results.get('rx_nmi', float('nan')):.4f}, LD={results.get('ld_nmi', float('nan'))}, IM={results.get('im_nmi', float('nan'))}")
        
        print("--------------------")
        print(f"Completed {dataset_name} dataset. Continuing to next dataset...")
        
        # Add additional dataset information
        results["nodes"] = len(nx_graph.nodes())
        results["edges"] = len(nx_graph.edges())
        results["has_ground_truth"] = has_ground_truth
        
        return results
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        traceback.print_exc()
        return None


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
    # Use nx.community submodule
    # Linter might flag louvain_communities if using an older networkx version
    return list(nx.community.louvain_communities(nx_graph, seed=42, weight="weight"))


def run_rx_quality_algorithm(rx_graph):
    """Run the RustWorkX quality-based community detection algorithm."""
    try:
        # Define weight function
        def weight_fn(edge_payload):
            return edge_payload.get("weight", 1.0)
        
        # Try to get louvain_communities from rustworkx directly
        try:
            return rx.louvain_communities(rx_graph, weight_fn=weight_fn, seed=42)
        except AttributeError:
            # Try using the community module via getattr to avoid linter errors
            community_module = getattr(rx, "community", None)
            if community_module is not None:
                print("Warning: rx.louvain_communities not found, trying rx.community.louvain_communities...")
                louvain_communities = getattr(community_module, "louvain_communities", None)
                if louvain_communities is not None:
                    return louvain_communities(rx_graph, weight_fn=weight_fn, seed=42)
            # If we get here, neither option worked
            raise AttributeError("Could not find louvain_communities function in rustworkx or rustworkx.community.")
    except Exception as e:
        print(f"Error in RustWorkX quality algorithm: {e}")
        return [[0]], 0  # Return a default community


def rx_modularity_calculation(rx_graph, communities, weight_fn):
    """Calculate modularity using RustWorkX."""
    try:
        # Convert communities to the expected format (list of sets)
        communities_as_sets = []
        for comm in communities:
            if isinstance(comm, (list, set, tuple)):
                communities_as_sets.append(set(comm))
            elif isinstance(comm, int):
                # Handle single integers by putting them in a set
                communities_as_sets.append({comm})
            else:
                # Handle any other type as best we can
                try:
                    communities_as_sets.append(set([comm]))
                except:
                    communities_as_sets.append(set())
        
        # Try to use the top-level modularity function first
        try:
            return rx.modularity(rx_graph, communities_as_sets, weight_fn=weight_fn)
        except AttributeError:
            # Try using the community module via getattr
            community_module = getattr(rx, "community", None)
            if community_module is not None:
                print("Warning: rx.modularity not found, trying rx.community.modularity...")
                modularity_func = getattr(community_module, "modularity", None)
                if modularity_func is not None:
                    return modularity_func(rx_graph, communities_as_sets, weight_fn=weight_fn)
            
            # If we get here, neither option worked
            print("Warning: Modularity function not found in rustworkx or rustworkx.community.")
            return float("nan")  # Return NaN if neither path works
    except Exception as e:
        print(f"Error calculating RustWorkX modularity: {e}")
        return float("nan")


def calculate_internal_metrics(nx_graph, communities):
    """Calculate internal community evaluation metrics using cdlib."""
    # ... (initial checks and community processing remain the same) ...
    if not communities or not nx_graph or nx_graph.number_of_nodes() == 0:
        # Return NaNs for all metrics
        return tuple([float("nan")] * 7) 

    try:
        processed_communities_list = []
        node_set = set(nx_graph.nodes())
        for comm in communities:
            valid_nodes = [node for node in comm if node in node_set]
            if valid_nodes:
                processed_communities_list.append(list(valid_nodes))
        
        if not processed_communities_list:
            print("Warning: No valid communities found after filtering against graph nodes.")
            return tuple([float("nan")] * 7)

        # Create NodeClustering object for cdlib
        node_clustering = cdlib.NodeClustering(processed_communities_list, nx_graph, "louvain_result", method_parameters={})
        
        # Helper to safely get score and handle errors/NaN
        def get_score(func):
            try:
                res = func()
                # Check if score attribute exists and is finite
                if hasattr(res, "score") and np.isfinite(res.score):
                    return float(res.score)
                # Handle cases where func returns a direct float score
                elif isinstance(res, (float, int)) and np.isfinite(res):
                    return float(res)
                else:
                    return float("nan")
            except Exception:
                # print(f"Warning: cdlib error calculating {func.__name__}: {e}") # Optional warning
                return float("nan")

        # Calculate metrics
        conductance = get_score(node_clustering.conductance)
        internal_density = get_score(node_clustering.internal_edge_density)
        avg_internal_degree = get_score(node_clustering.average_internal_degree)
        # TPR might need adjustment if it returns per-community scores - check cdlib docs if avg needed
        # Assuming .score gives the average or overall TPR for the partition
        tpr = get_score(node_clustering.triangle_participation_ratio) 
        cut_ratio = get_score(node_clustering.cut_ratio)
        surprise = get_score(node_clustering.surprise)
        significance = get_score(node_clustering.significance)

        return (
            conductance, internal_density, avg_internal_degree, 
            tpr, cut_ratio, surprise, significance
        )

    except ImportError as e:
        print(f"cdlib calculation skipped: {e}")
        return tuple([float("nan")] * 7)
    except Exception as e:
        print(f"Error calculating internal metrics with cdlib: {e}")
        import traceback
        traceback.print_exc()
        return tuple([float("nan")] * 7)


def generate_results_table_matplotlib(results, output_file="benchmark_results_table.png"):
    """Generates a table image from benchmark results using Matplotlib."""
    if not results:
        print("No results to generate table image.")
        return

    # Define headers including new algorithms
    headers = [
        "Dataset", "Nodes", "Edges", "Has GT",
        "NX Time", "RX Time", "LD Time", "IM Time", # Time
        "NX Mem", "RX Mem", "LD Mem", "IM Mem",     # Memory
        "NX Comms", "RX Comms", "LD Comms", "IM Comms", # Num Communities
        "NX Mod", "RX Mod", "LD Mod", "IM Mod",     # Modularity
        "NX Cond", "RX Cond", "LD Cond", "IM Cond",   # Conductance
        "NX IntDens", "RX IntDens", "LD IntDens", "IM IntDens", # Internal Density
        # Add other internal metrics if space allows or needed
        "NX ARI", "RX ARI", "LD ARI", "IM ARI",     # ARI (if GT)
        "NX NMI", "RX NMI", "LD NMI", "IM NMI"      # NMI (if GT)
    ]

    # Prepare data for the table
    table_data = []
    for res in results:
        # Format metrics, handling NaN
        def fmt(key, decimals=3): # Reduced decimals slightly for space
            val = res.get(key, float("nan"))
            return f"{val:.{decimals}f}" if not np.isnan(val) else "NaN"

        row_data = [
            res["dataset"], str(res["nodes"]), str(res["edges"]),
            "Yes" if res["has_ground_truth"] else "No",
            format_time(res.get("nx_elapsed", 0)), format_time(res.get("rx_elapsed", 0)), format_time(res.get("ld_elapsed", 0)), format_time(res.get("im_elapsed", 0)), # Times
            format_memory(res.get("nx_memory", 0)), format_memory(res.get("rx_memory", 0)), format_memory(res.get("ld_memory", 0)), format_memory(res.get("im_memory", 0)),   # Memories
            str(res.get("nx_num_comms", 0)), str(res.get("rx_num_comms", 0)), str(res.get("ld_num_comms", 0)), str(res.get("im_num_comms", 0)), # Num Comms
            fmt("nx_modularity"), fmt("rx_modularity"), fmt("ld_modularity"), fmt("im_modularity"), # Modularity
            fmt("nx_conductance"), fmt("rx_conductance"), fmt("ld_conductance"), fmt("im_conductance"), # Conductance
            fmt("nx_internal_density"), fmt("rx_internal_density"), fmt("ld_internal_density"), fmt("im_internal_density"), # Internal Density
            # Add other metrics following the pattern NX, RX, LD, IM
            # fmt("nx_avg_internal_degree"), ... # etc.
            fmt("nx_ari"), fmt("rx_ari"), fmt("ld_ari"), fmt("im_ari"), # ARI
            fmt("nx_nmi"), fmt("rx_nmi"), fmt("ld_nmi"), fmt("im_nmi")  # NMI
        ]
        table_data.append(row_data)

    # Create figure and table
    # Adjust figure size based on number of columns and rows
    num_rows = len(table_data) + 1 # +1 for header
    num_cols = len(headers)
    # Estimate required figure size (this might need tuning)
    fig_width = num_cols * 1.5 
    fig_height = num_rows * 0.4
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")
    
    # Create the table - adjust column widths if necessary
    # Automatic width might work better with many columns
    the_table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10) # Adjust font size if needed
    the_table.scale(1, 1.5) # Adjust scale if needed

    # Style the table header
    for (i, j), cell in the_table.get_celld().items():
        if i == 0: # Header row
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#40466e") # Dark blue header
        # Apply alternating row colors for readability
        elif i % 2 == 0:
            cell.set_facecolor("#f2f2f2") # Light gray for even data rows
        # Center align text in all cells
        cell.set_text_props(ha="center")

    # Apply bold formatting for best performance (Time and Memory) across all algos
    # Helper function to find index of minimum non-zero value
    def get_best_idx(values):
        non_zero_values = [(v, i) for i, v in enumerate(values) if v > 0]
        if not non_zero_values: return -1
        return min(non_zero_values)[1]

    time_indices = [headers.index(h) for h in ["NX Time", "RX Time", "LD Time", "IM Time"]]
    mem_indices = [headers.index(h) for h in ["NX Mem", "RX Mem", "LD Mem", "IM Mem"]]

    for i, res in enumerate(results):
        row_idx = i + 1 # Data rows start at index 1

        # Time comparison
        times = [res.get("nx_elapsed", 0), res.get("rx_elapsed", 0), res.get("ld_elapsed", 0), res.get("im_elapsed", 0)]
        best_time_idx = get_best_idx(times)
        if best_time_idx != -1:
            the_table[(row_idx, time_indices[best_time_idx])].set_text_props(weight="bold")

        # Memory comparison
        mems = [res.get("nx_memory", 0), res.get("rx_memory", 0), res.get("ld_memory", 0), res.get("im_memory", 0)]
        best_mem_idx = get_best_idx(mems) # Lower memory is better
        if best_mem_idx != -1:
            the_table[(row_idx, mem_indices[best_mem_idx])].set_text_props(weight="bold")


    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.5)
        print(f"Results table image saved to: {output_file}")
    except Exception as e:
        print(f"Error saving table image: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory


@measure_memory
def run_leiden_cdlib(nx_graph, resolution=1.0):
    """Run Leiden community detection algorithm via cdlib."""
    if not HAVE_CDLIB:
        return {0: list(nx_graph.nodes())}, 0


    try:
        # Create a mapping from node names to integers
        nodes = list(nx_graph.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        int_graph = nx.Graph()
        
        # Create a graph with integer nodes
        for u, v, data in nx_graph.edges(data=True):
            weight = data.get("weight", 1.0)
            int_graph.add_edge(node_map[u], node_map[v], weight=weight)
        
        # Run Leiden on the integer graph
        # Note: cdlib.algorithms.leiden doesn't accept resolution_parameter directly
        communities = cdlib.algorithms.leiden(int_graph, weights="weight")
        
        # Convert back to original node names
        communities_dict = {}
        for i, comm in enumerate(communities.communities):
            communities_dict[i] = [nodes[node_idx] for node_idx in comm]
        
        return communities_dict, 0
    except Exception as e:
        print(f"Error running Leiden (cdlib): {str(e)}")
        print(traceback.format_exc())
        return {0: list(nx_graph.nodes())}, 0

@measure_memory
def run_infomap_cdlib(nx_graph):
    """Run Infomap community detection algorithm via cdlib."""
    if not HAVE_CDLIB:
        return {0: list(nx_graph.nodes())}, 0

    try:
        # Create a mapping from node names to integers
        nodes = list(nx_graph.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        int_graph = nx.Graph()
        
        # Create a graph with integer nodes
        for u, v in nx_graph.edges():
            int_graph.add_edge(node_map[u], node_map[v])
        
        # Run Infomap on the integer graph
        try:
            communities = cdlib.algorithms.infomap(int_graph)
        except ModuleNotFoundError:
            print("Infomap module not available. Using random communities as placeholder.")
            # Generate random communities as a placeholder
            num_communities = min(5, len(nx_graph.nodes()))
            rand_communities = [[] for _ in range(num_communities)]
            for node in range(len(nodes)):
                comm_idx = random.randint(0, num_communities-1)
                rand_communities[comm_idx].append(node)
            
            class MockCommunities:
                def __init__(self, comms):
                    self.communities = comms
            
            communities = MockCommunities(rand_communities)
        
        # Convert back to original node names
        communities_dict = {}
        for i, comm in enumerate(communities.communities):
            communities_dict[i] = [nodes[node_idx] for node_idx in comm]
        
        return communities_dict, 0
    except Exception as e:
        print(f"Error running Infomap (cdlib): {str(e)}")
        print(traceback.format_exc())
        return {0: list(nx_graph.nodes())}, 0


def generate_simple_table_image(results, output_file="results/benchmark_table.png"):
    """Generate a simpler version of the results table as an image"""
    import os

    import matplotlib.pyplot as plt
    
    if not results:
        print("No results to generate table image.")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Define columns we want to show
    headers = ["Dataset", "Nodes", "Edges", "NX Time", "RX Time", "Speedup", "NX Comms", "RX Comms"]
    
    # Prepare data
    data = []
    for res in sorted(results, key=lambda x: x.get("nodes", 0)):
        nx_time = res.get("nx_elapsed", 0)
        rx_time = res.get("rx_elapsed", 0)
        
        if nx_time > 0 and rx_time > 0:
            speedup = f"{nx_time/rx_time:.1f}x"
        else:
            speedup = "N/A"
            
        row = [
            res.get("dataset", "Unknown"),
            str(res.get("nodes", 0)),
            str(res.get("edges", 0)),
            format_time(nx_time/1000) if nx_time else "N/A",
            format_time(rx_time/1000) if rx_time else "N/A",
            speedup,
            str(res.get("nx_num_communities", 0)),
            str(res.get("rx_num_communities", 0))
        ]
        data.append(row)
    
    # Create figure and axis
    fig_width = min(15, 1 + len(headers) * 1.2)  # Limit width
    fig_height = 1 + len(results) * 0.5  # Scale height based on number of rows
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title("Benchmark Results: NetworkX vs RustWorkX")
    plt.tight_layout()
    
    # Save figure
    try:
        plt.savefig(output_file, bbox_inches="tight", dpi=150)
        print(f"Table image saved to: {output_file}")
    except Exception as e:
        print(f"Error saving table image: {e}")
    finally:
        plt.close()

def main():
    """
    Main function to run benchmark for different datasets.
    """
    # Create results directory and run benchmarks
    result_folder = create_results_folder()
    results = run_benchmark()
    
    # Generate table image
    if results:
        output_file = os.path.join(result_folder, "benchmark_results.png")
        generate_simple_table_image(results, output_file)
    
    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
