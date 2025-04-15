import os
import shutil
import time

import networkx as nx
import numpy as np
import polars as pl

# Import format_time from the utils module
from benchmark_utils import format_time


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
    """Load graph from CSV, calculate similarity weight from distance."""
    print("Loading graph from datasets/graph_edges.csv...")
    try:
        df = pl.read_csv("datasets/graph_edges.csv")
        G = nx.Graph()
        for row in df.iter_rows(named=True):
            src = row["src"]
            dst = row["dst"]
            distance = float(row.get("distance", 2.0)) # Default to max distance if missing
            # Calculate similarity: 1 - d/2 for cosine distance [0, 2] -> similarity [0, 1]
            # Ensure weight is positive
            similarity = max(1e-9, 1.0 - distance / 2.0) 
            G.add_edge(src, dst, weight=similarity) # Use similarity as weight
            
        nodes = list(G.nodes())
        edges = list(G.edges())
        print(f"Created graph with {len(nodes)} nodes and {len(edges)} edges (using similarity weights)")
        true_labels = [0] * len(nodes)
        print("Warning: No ground truth available for this dataset.")
        return G, true_labels, False
    except Exception as e:
        print(f"Error loading graph from CSV: {e}")
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False


def load_graph_edges_parquet():
    """Load graph from Parquet, calculate similarity weight from distance."""
    print("Loading graph from datasets/graph_edges_big.parquet...")
    try:
        df = pl.read_parquet("datasets/graph_edges_big.parquet")
        G = nx.Graph()
        for row in df.iter_rows(named=True):
            src = row["src"]
            dst = row["dst"]
            distance = float(row.get("distance", 2.0)) # Default to max distance if missing
            # Calculate similarity: 1 - d/2 for cosine distance [0, 2] -> similarity [0, 1]
            # Ensure weight is positive
            similarity = max(1e-9, 1.0 - distance / 2.0) 
            G.add_edge(src, dst, weight=similarity) # Use similarity as weight

        nodes = list(G.nodes())
        edges = list(G.edges())
        print(f"Created graph with {len(nodes)} nodes and {len(edges)} edges (using similarity weights)")
        true_labels = [0] * len(nodes)
        print("Warning: No ground truth available for this dataset.")
        return G, true_labels, False
    except Exception as e:
        print(f"Error loading graph from Parquet: {e}")
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False


def load_graph_edges_9m_parquet():
    """Load graph from the 9M Parquet file, calculate similarity weight from distance."""
    print("Loading graph from datasets/graph_edges_9m.parquet...")
    try:
        df = pl.read_parquet("datasets/graph_edges_9m.parquet")
        G = nx.Graph()
        for row in df.iter_rows(named=True):
            src = row["src"]
            dst = row["dst"]
            distance = float(row.get("distance", 2.0)) # Default to max distance if missing
            # Calculate similarity: 1 - d/2 for cosine distance [0, 2] -> similarity [0, 1]
            # Ensure weight is positive
            similarity = max(1e-9, 1.0 - distance / 2.0)
            G.add_edge(src, dst, weight=similarity) # Use similarity as weight

        nodes = list(G.nodes())
        edges = list(G.edges())
        print(f"Created graph with {len(nodes)} nodes and {len(edges)} edges (using similarity weights)")
        true_labels = [0] * len(nodes)
        print("Warning: No ground truth available for this dataset.")
        return G, true_labels, False
    except Exception as e:
        print(f"Error loading graph from 9M Parquet: {e}")
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False


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