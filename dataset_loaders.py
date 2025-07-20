import logging
import os
import shutil
import time
from typing import Any

import networkx as nx
import numpy as np
import polars as pl

# Import format_time from the utils module
from benchmark_utils import format_time

# Setup logger for this module
logger = logging.getLogger(__name__)
# BasicConfig should be handled by the main script (benchmark_community.py)

# Define a common return type for simple loaders
SimpleLoaderReturn = tuple[nx.Graph, list[int], bool]
# Define a common return type for dict-based loaders
DictLoaderReturn = dict[str, nx.Graph | dict[Any, int] | dict[str, Any] | str | None]


def load_karate_club() -> SimpleLoaderReturn:
    """Load Zachary's Karate Club dataset with ground truth communities.

    The ground truth is derived from the 'club' attribute of the nodes.
    Edges are assigned a default weight of 1.0.

    Returns:
        A tuple containing:
            - nx.Graph: The Karate Club graph.
            - list[int]: A list of true community labels for each node.
            - bool: True, indicating ground truth is available.
    """
    G: nx.Graph = nx.karate_club_graph()
    true_labels: list[int] = [1 if G.nodes[node]["club"] == "Mr. Hi" else 0 for node in G.nodes()]
    for u, v in G.edges():
        G.edges[u, v]["weight"] = 1.0
    return G, true_labels, True


def load_davis_women() -> SimpleLoaderReturn:
    """Load Davis Southern Women dataset with ground truth communities.

    No explicit ground truth is provided by NetworkX for this dataset.
    Dummy labels (all nodes in one group) are created.

    Returns:
        A tuple containing:
            - nx.Graph: The Davis Southern Women graph.
            - list[int]: A list of dummy true community labels (all zeros).
            - bool: True, but the ground truth is artificial.
    """
    G: nx.Graph = nx.davis_southern_women_graph()
    true_labels: list[int] = [0] * len(G.nodes())
    logger.warning("Davis Southern Women graph loaded with dummy ground truth (all nodes in one group).")
    for u, v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_florentine_families() -> SimpleLoaderReturn:
    """Load Florentine Families dataset with ground truth communities.

    No standard ground truth is provided by NetworkX for this dataset.
    Dummy labels (all nodes in one group) are created.

    Returns:
        A tuple containing:
            - nx.Graph: The Florentine Families graph.
            - list[int]: A list of dummy true community labels (all zeros).
            - bool: True, but the ground truth is artificial.
    """
    G: nx.Graph = nx.florentine_families_graph()
    true_labels: list[int] = [0] * len(G.nodes())
    logger.warning("Florentine Families graph loaded with dummy ground truth (all nodes in one group).")
    for u, v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_les_miserables() -> SimpleLoaderReturn:
    """Load Les Misérables dataset with ground truth communities.

    Node labels are remapped to integers, and communities are derived
    from predefined character groupings. Edges are assigned a default
    weight of 1.0.

    Returns:
        A tuple containing:
            - nx.Graph: The Les Misérables graph with integer node IDs.
            - list[int]: A list of true community labels for each node.
            - bool: True, indicating ground truth is available.
    """
    G_orig: nx.Graph = nx.les_miserables_graph()
    
    old_groups: dict[str, int] = {
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
    
    pos: dict[Any, np.ndarray] = nx.spring_layout(G_orig, k=1/np.sqrt(len(G_orig.nodes())), iterations=50, seed=42)
    H: nx.Graph = nx.Graph()
    mapping: dict[Any, int] = {old: i for i, old in enumerate(G_orig.nodes())}
    
    for old_node, new_node_idx in mapping.items():
        H.add_node(new_node_idx, pos=pos.get(old_node), value=old_groups.get(old_node, -1))

    for u_orig, v_orig in G_orig.edges():
        H.add_edge(mapping[u_orig], mapping[v_orig], weight=1.0)

    true_labels: list[int] = [H.nodes[node_idx]["value"] for node_idx in H.nodes()]
    return H, true_labels, True


def load_football() -> SimpleLoaderReturn:
    """Load American College Football dataset with ground truth communities.

    Assumes the file 'datasets/football.gml' exists.
    Edges are assigned weight 1.0 if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The American College Football graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/football.gml' is not found.
    """
    file_path: str = "datasets/football.gml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")
    
    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node]["value"] for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_political_books() -> SimpleLoaderReturn:
    """Load Political Books dataset with ground truth communities.

    Assumes the file 'datasets/polbooks.gml' exists.
    Edges are assigned weight 1.0 if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The Political Books graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/polbooks.gml' is not found.
    """
    file_path: str = "datasets/polbooks.gml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")

    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node]["value"] for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_wiki_news_sim(**kwargs) -> DictLoaderReturn:
    """Load the Wiki News Similarity dataset from a Parquet file.

    This is a wrapper around `load_graph_edges_gt_clusters` for a specific
    large dataset with ground truth communities. It accepts and ignores
    extra keyword arguments like `dtype_spec` for compatibility with the
    benchmark runner.

    Returns:
        DictLoaderReturn: A dictionary containing the graph, ground truth,
                          and other metadata.
    """
    # The benchmark runner might pass dtype_spec, which we ignore.
    _ = kwargs  # Explicitly mark as unused
    file_path = "datasets/wiki_news_edges_sim_thresh_0_9.parquet"
    return load_graph_edges_gt_clusters(
        file_path=file_path,
        name="Wiki News Sim",
        dtype_spec={"src": str, "dst": str}
    )


def load_dolphins() -> SimpleLoaderReturn:
    """Load Dolphins Social Network dataset with ground truth communities.

    Assumes file 'datasets/dolphins.gml' exists and that node attribute 'value'
    contains the ground truth community. Edges are assigned weight 1.0
    if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The Dolphins Social Network graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/dolphins.gml' is not found.
    """
    file_path: str = "datasets/dolphins.gml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")
        
    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node]["value"] for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_polblogs() -> SimpleLoaderReturn:
    """Load Political Blogs dataset with ground truth communities.

    Assumes file 'datasets/polblogs.gml' exists and that node attribute 'value'
    contains the ground truth community. Edges are assigned weight 1.0
    if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The Political Blogs graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/polblogs.gml' is not found.
    """
    file_path: str = "datasets/polblogs.gml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")

    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node]["value"] for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_cora(data_dir: str = "datasets/cora", name: str = "Cora") -> SimpleLoaderReturn:
    """Load Cora citation network with ground truth communities.

    Assumes file 'datasets/cora.gml' exists and that node attribute 'value'
    contains the class label. Edges are assigned weight 1.0
    if not specified.

    Args:
        data_dir: Directory containing 'cora.gml'.
        name: Name of the dataset (used for logging, not functionally).

    Returns:
        A tuple containing:
            - nx.Graph: The Cora citation graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If '{data_dir}/cora.gml' is not found.
    """
    file_path: str = os.path.join(data_dir, "cora.gml")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")

    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node]["value"] for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_facebook() -> SimpleLoaderReturn:
    """Load Facebook Ego Networks / Social Circles dataset with ground truth communities.

    Assumes file 'datasets/facebook.gml' exists and that node attribute 'value'
    contains the community label. Edges are assigned weight 1.0
    if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The Facebook graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/facebook.gml' is not found.
    """
    file_path: str = "datasets/facebook.gml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")

    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node]["value"] for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_citeseer() -> SimpleLoaderReturn:
    """Load Citeseer dataset with ground truth communities.

    Assumes file 'datasets/citeseer.gml' exists and that node attribute 'value'
    (or default 0) contains the ground truth community. Edges are assigned weight 1.0
    if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The Citeseer graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/citeseer.gml' is not found.
    """
    file_path: str = "datasets/citeseer.gml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")

    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node].get("value", 0) for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_email_eu_core() -> SimpleLoaderReturn:
    """Load Email EU Core dataset with ground truth communities.

    Assumes file 'datasets/email_eu_core.gml' exists and that node attribute 'value'
    (or default 0) contains the ground truth community. Edges are assigned weight 1.0
    if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The Email EU Core graph.
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/email_eu_core.gml' is not found.
    """
    file_path: str = "datasets/email_eu_core.gml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please ensure it exists.")

    G: nx.Graph = nx.read_gml(file_path, label="id")
    true_labels: list[int] = [G.nodes[node].get("value", 0) for node in G.nodes()]
    for u,v in G.edges():
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_graph_edges_csv() -> SimpleLoaderReturn:
    """Load graph from CSV, calculate similarity weight from distance.

    The CSV is expected to have 'src', 'dst', and optionally 'distance' columns.
    Edge weights are calculated as similarity = 1 - (distance / 2.0), ensuring
    a minimum weight of 1e-9. If 'distance' is missing, it defaults to 2.0.
    No ground truth is loaded; dummy labels are created.

    Returns:
        A tuple containing:
            - nx.Graph: The loaded graph.
            - list[int]: A list of dummy true community labels (all zeros).
            - bool: False, indicating no actual ground truth is available.
            
    Raises:
        FileNotFoundError: If 'datasets/graph_edges.csv' is not found.
        pl.exceptions.ComputeError: If Polars encounters issues reading the CSV.
    """
    file_path: str = "datasets/graph_edges.csv"
    logger.info(f"Loading graph from {file_path}...")
    
    # Import os for CPU optimization
    import os
    original_threads = os.environ.get("POLARS_MAX_THREADS")
    cpu_count = os.cpu_count() or 4
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Use lazy CSV reading for better performance
        lf = pl.scan_csv(file_path)
        
        # Check if distance column exists
        schema = lf.schema
        required_cols = ["src", "dst"]
        distance_col = "distance"
        
        if distance_col in schema:
            required_cols.append(distance_col)
        
        # Select only needed columns and apply transformations
        lf = lf.select(required_cols)
        
        # Handle missing distance and compute similarity weights
        if distance_col not in lf.columns:
            lf = lf.with_columns(pl.lit(2.0, dtype=pl.Float32).alias(distance_col))
        
        lf = lf.with_columns([
            pl.col(distance_col).fill_null(2.0).cast(pl.Float32)
        ]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32))
            ).alias("weight")
        )
        
        # Collect with optimizations
        logger.info(f"  Processing CSV with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,
            predicate_pushdown=True,
            projection_pushdown=True,
            simplify_expression=True
        )
        
        # Build graph efficiently
        G: nx.Graph = nx.Graph()
        edge_data = df.select(["src", "dst", "weight"]).rows()
        G.add_weighted_edges_from(edge_data)

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        logger.info(f"Created graph with {num_nodes} nodes and {num_edges} edges from CSV.")
        
        true_labels: list[int]
        if num_nodes > 0:
            true_labels = [0] * num_nodes
            logger.warning("No ground truth available for this CSV dataset. Using dummy labels.")
        else:
            true_labels = []
            logger.warning("Graph loaded from CSV is empty. No labels generated.")
            
        return G, true_labels, False
        
    except Exception as e:
        logger.error(f"Error loading graph from CSV {file_path}: {e}", exc_info=True)
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False
    finally:
        # Restore thread settings
        if original_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = original_threads
        elif "POLARS_MAX_THREADS" in os.environ:
            del os.environ["POLARS_MAX_THREADS"]


def load_graph_edges_parquet() -> SimpleLoaderReturn:
    """Load graph from Parquet, calculate similarity weight from distance.

    Expects 'src', 'dst', and optionally 'distance' columns.
    Edge weights are similarity = 1 - (distance / 2.0), min 1e-9.
    'distance' defaults to 2.0 if missing. No ground truth; dummy labels created.

    Returns:
        A tuple containing:
            - nx.Graph: The loaded graph.
            - list[int]: Dummy true community labels (all zeros).
            - bool: False (no actual ground truth).
            
    Raises:
        FileNotFoundError: If 'datasets/graph_edges_big.parquet' is not found.
        pl.exceptions.ComputeError: If Polars has issues reading the Parquet.
    """
    file_path: str = "datasets/graph_edges_big.parquet"
    logger.info(f"Loading graph from {file_path}...")
    
    # Import os for CPU optimization
    import os
    original_threads = os.environ.get("POLARS_MAX_THREADS")
    cpu_count = os.cpu_count() or 4
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        # Use lazy Parquet reading for better performance
        lf = pl.scan_parquet(file_path)
        schema = lf.schema
        
        required_cols = ["src", "dst"]
        distance_col = "distance"
        
        if distance_col in schema:
            required_cols.append(distance_col)
        
        # Select only needed columns
        lf = lf.select(required_cols)
        
        # Handle missing distance and compute similarity weights
        if distance_col not in lf.columns:
            lf = lf.with_columns(pl.lit(2.0, dtype=pl.Float32).alias(distance_col))
        
        lf = lf.with_columns([
            pl.col(distance_col).fill_null(2.0).cast(pl.Float32)
        ]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32))
            ).alias("weight")
        )
        
        # Collect with optimizations
        logger.info(f"  Processing Parquet with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,
            predicate_pushdown=True,
            projection_pushdown=True,
            simplify_expression=True,
            slice_pushdown=True,
            comm_subplan_elim=True,
            comm_subexpr_elim=True
        )
        
        # Build graph efficiently with chunking for large datasets
        G: nx.Graph = nx.Graph()
        total_rows = len(df)
        chunk_size = 1_000_000
        
        if total_rows > chunk_size:
            logger.info(f"  Processing {total_rows} edges in chunks...")
            for i in range(0, total_rows, chunk_size):
                chunk = df.slice(i, min(chunk_size, total_rows - i))
                edges_chunk = chunk.select(["src", "dst", "weight"]).rows()
                G.add_weighted_edges_from(edges_chunk)
        else:
            edge_data = df.select(["src", "dst", "weight"]).rows()
            G.add_weighted_edges_from(edge_data)

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        logger.info(f"Created graph with {num_nodes} nodes and {num_edges} edges from Parquet.")

        true_labels: list[int]
        if num_nodes > 0:
            true_labels = [0] * num_nodes
            logger.warning("No ground truth for this Parquet dataset. Using dummy labels.")
        else:
            true_labels = []
            logger.warning("Graph from Parquet is empty. No labels.")
            
        return G, true_labels, False
        
    except Exception as e:
        logger.error(f"Error loading graph from Parquet {file_path}: {e}", exc_info=True)
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False
    finally:
        # Restore thread settings
        if original_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = original_threads
        elif "POLARS_MAX_THREADS" in os.environ:
            del os.environ["POLARS_MAX_THREADS"]


def load_graph_edges_9m_parquet() -> SimpleLoaderReturn:
    """Load graph from the 9M Parquet file, calculate similarity weight from distance.

    Expects 'src', 'dst', and optionally 'distance' columns.
    Edge weights are similarity = 1 - (distance / 2.0), min 1e-9.
    'distance' defaults to 2.0 if missing. No ground truth; dummy labels created.

    Returns:
        A tuple containing:
            - nx.Graph: The loaded graph.
            - list[int]: Dummy true community labels (all zeros).
            - bool: False (no actual ground truth).

    Raises:
        FileNotFoundError: If 'datasets/graph_edges_9m.parquet' is not found.
        pl.exceptions.ComputeError: If Polars has issues reading the Parquet.
    """
    file_path: str = "datasets/graph_edges_9m.parquet"
    logger.info(f"Loading graph from {file_path}...")
    
    # Import os for CPU optimization  
    import os
    original_threads = os.environ.get("POLARS_MAX_THREADS")
    cpu_count = os.cpu_count() or 4
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Use lazy Parquet reading for better performance on large files
        lf = pl.scan_parquet(file_path)
        schema = lf.schema
        
        required_cols = ["src", "dst"]
        distance_col = "distance"
        
        if distance_col in schema:
            required_cols.append(distance_col)
        
        # Select only needed columns
        lf = lf.select(required_cols)
        
        # Handle missing distance and compute similarity weights
        if distance_col not in lf.columns:
            lf = lf.with_columns(pl.lit(2.0, dtype=pl.Float32).alias(distance_col))
        
        lf = lf.with_columns([
            pl.col(distance_col).fill_null(2.0).cast(pl.Float32)
        ]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32))
            ).alias("weight")
        )
        
        # Collect with optimizations for large 9M dataset
        logger.info(f"  Processing 9M Parquet with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,
            predicate_pushdown=True,
            projection_pushdown=True,
            simplify_expression=True,
            slice_pushdown=True,
            comm_subplan_elim=True,
            comm_subexpr_elim=True
        )
        
        # Build graph efficiently with chunking for 9M dataset
        G: nx.Graph = nx.Graph()
        total_rows = len(df)
        chunk_size = 500_000  # Smaller chunks for 9M dataset
        
        logger.info(f"  Processing {total_rows} edges in {chunk_size}-row chunks...")
        for i in range(0, total_rows, chunk_size):
            chunk = df.slice(i, min(chunk_size, total_rows - i))
            edges_chunk = chunk.select(["src", "dst", "weight"]).rows()
            G.add_weighted_edges_from(edges_chunk)
            if (i // chunk_size + 1) % 10 == 0:  # Progress logging every 10 chunks
                logger.info(f"    Processed {i + len(chunk)} / {total_rows} edges...")

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        logger.info(f"Created graph with {num_nodes} nodes and {num_edges} edges from 9M Parquet.")
        
        true_labels: list[int]
        if num_nodes > 0:
            true_labels = [0] * num_nodes
            logger.warning("No ground truth for 9M Parquet dataset. Using dummy labels.")
        else:
            true_labels = []
            logger.warning("Graph from 9M Parquet is empty. No labels.")

        return G, true_labels, False
        
    except Exception as e:
        logger.error(f"Error loading graph from 9M Parquet {file_path}: {e}", exc_info=True)
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False
    finally:
        # Restore thread settings
        if original_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = original_threads
        elif "POLARS_MAX_THREADS" in os.environ:
            del os.environ["POLARS_MAX_THREADS"]


def load_lfr(n: int = 250, mu: float = 0.1, name: str = "LFR Benchmark") -> SimpleLoaderReturn:
    """Generate a synthetic LFR benchmark graph with ground truth communities.

    Uses `nx.LFR_benchmark_graph`. Node attribute "community" (a frozenset)
    is used to derive integer community labels (min element of the set).

    Args:
        n: Number of nodes.
        mu: Mixing parameter (fraction of inter-community edges for a node).
        name: Name for logging.

    Returns:
        A tuple containing:
            - nx.Graph: The LFR benchmark graph.
            - list[int]: True community labels.
            - bool: True (ground truth available).
    """
    tau1: int = 3      # Power law exponent for degree distribution
    tau2: int = 1.5    # Power law exponent for community size distribution
    avg_degree: int = 10
    min_comm: int = 50
    # Ensure max_community is at least min_community and reasonable for n
    max_comm: int = max(min_comm, n // 5 if n // 5 >= min_comm else min_comm + 1)
    if n < min_comm : # Adjust if n is too small for min_community
        min_comm = max(1, n // 2 if n > 1 else 1)
        max_comm = n
        logger.warning(f"LFR: n ({n}) < min_community (50). Adjusted min_community to {min_comm}, max_community to {max_comm}.")


    logger.info(f"Generating {name} graph (n={n}, mu={mu}, tau1={tau1}, tau2={tau2}, avg_deg={avg_degree}, min_comm={min_comm}, max_comm={max_comm})...")
    start_time: float = time.time()
    
    G: nx.Graph = nx.LFR_benchmark_graph(
        n, tau1, tau2, mu,
        average_degree=avg_degree,
        min_community=min_comm,
        max_community=max_comm,
        seed=42
    )
    gen_time: float = time.time() - start_time
    logger.info(f"Generated {name} ({len(G.nodes())} nodes, {len(G.edges())} edges) in {format_time(gen_time)}")
    
    # Ensure all nodes have the "community" attribute
    true_labels: list[int] = []
    for node_id in G.nodes():
        community_set = G.nodes[node_id].get("community")
        if community_set is None or not isinstance(community_set, frozenset) or not community_set:
            logger.warning(f"Node {node_id} in LFR graph missing valid 'community' attribute. Assigning to default label -1.")
            true_labels.append(-1) # Default label for problematic nodes
        else:
            true_labels.append(min(list(community_set))) # Get min element from the frozenset of community IDs

    for u,v in G.edges(): # Ensure weights for LFR if not added by default
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
             
    return G, true_labels, True


def load_large_synthetic() -> SimpleLoaderReturn:
    """Generate a large synthetic graph with many communities.

    Creates a graph with 15 predefined communities of varying sizes.
    Intra-community connection probability is higher than inter-community.

    Returns:
        A tuple containing:
            - nx.Graph: The SBM graph.
            - list[int]: True community labels.
            - bool: True (ground truth available).
    """
    logger.info("Generating large synthetic SBM graph...")
    num_nodes_total: int = 1500 
    sizes: list[int] = [50, 80, 100, 120, 70, 90, 110, 60, 150, 130, 100, 80, 120, 140, 100]
    
    # Ensure total size matches num_nodes_total if it was specified, otherwise sum sizes.
    if sum(sizes) != num_nodes_total:
        logger.warning(f"Sum of SBM community sizes ({sum(sizes)}) does not match target total nodes ({num_nodes_total}). Using sum of sizes.")
        # num_nodes_total = sum(sizes) # This line is not needed as sizes define the graph

    p_in: float = 0.1  # Intra-community connection probability
    p_out: float = 0.001 # Inter-community connection probability

    probs: np.ndarray = np.full((len(sizes), len(sizes)), p_out)
    np.fill_diagonal(probs, p_in)

    G: nx.Graph = nx.stochastic_block_model(sizes, probs, seed=42)

    true_labels: list[int] = []
    for i, size_val in enumerate(sizes):
        true_labels.extend([i] * size_val)

    logger.info(f"Generated SBM graph with {len(G.nodes())} nodes, {len(G.edges())} edges, and {len(sizes)} communities.")
    for u,v in G.edges(): # Ensure weights
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
    return G, true_labels, True


def load_snap_text_dataset(name: str, edge_file_base: str, community_file_base: str) -> SimpleLoaderReturn:
    """Generic function to load SNAP text datasets (ungraph.txt, all.cmty.txt).

    Handles unzipping .gz files if necessary.
    Nodes are expected to be integers. Communities are derived from the
    community file, with overlapping nodes assigned to the last community ID.

    Args:
        name: Name of the dataset (for logging).
        edge_file_base: Base name for the edge file (e.g., "com-orkut").
        community_file_base: Base name for the community file.

    Returns:
        A tuple containing:
            - nx.Graph: The loaded graph.
            - list[int]: True community labels.
            - bool: True (ground truth available).
            
    Raises:
        FileNotFoundError: If required dataset files (or their .gz versions)
                           are not found in the 'datasets/' directory.
    """
    dataset_dir: str = "datasets"
    edge_file_gz: str = os.path.join(dataset_dir, f"{edge_file_base}.ungraph.txt.gz")
    community_file_gz: str = os.path.join(dataset_dir, f"{community_file_base}.all.cmty.txt.gz")
    edge_file: str = os.path.join(dataset_dir, f"{edge_file_base}.ungraph.txt")
    community_file: str = os.path.join(dataset_dir, f"{community_file_base}.all.cmty.txt")

    for gz_path, plain_path in [(edge_file_gz, edge_file), (community_file_gz, community_file)]:
        if not os.path.exists(plain_path):
            if os.path.exists(gz_path):
                logger.info(f"Unzipping {gz_path} to {plain_path}...")
                import gzip
                with gzip.open(gz_path, "rb") as f_in, open(plain_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                logger.info(f"Unzipped {gz_path} successfully.")
            else:
                raise FileNotFoundError(
                    f"{name} dataset file not found: neither {plain_path} nor {gz_path} exists. "
                    f"Please download from SNAP and place in '{dataset_dir}'."
                )

    logger.info(f"Loading {name} graph from {edge_file}...")
    start_time_graph: float = time.time()
    G: nx.Graph = nx.read_edgelist(edge_file, comments="#", create_using=nx.Graph(), nodetype=int)
    load_time_graph: float = time.time() - start_time_graph
    logger.info(f"Loaded {name} graph ({len(G.nodes())} N, {len(G.edges())} E) in {format_time(load_time_graph)}.")

    logger.info(f"Loading {name} communities from {community_file}...")
    start_time_comm: float = time.time()
    raw_communities: dict[int, set[int]] = {}
    nodes_in_any_community: set[int] = set()
    
    with open(community_file) as f:
        for comm_idx, line in enumerate(f):
            try:
                node_ids_in_line: list[int] = [int(n_str) for n_str in line.strip().split()]
                if node_ids_in_line:
                    current_comm_set: set[int] = set(node_ids_in_line)
                    raw_communities[comm_idx] = current_comm_set
                    nodes_in_any_community.update(current_comm_set)
            except ValueError:
                logger.warning(f"Skipping invalid line in {community_file} (line {comm_idx+1}): {line.strip()}")
                continue
    
    load_time_comm: float = time.time() - start_time_comm
    num_raw_communities: int = len(raw_communities)
    logger.info(f"Loaded {num_raw_communities} raw community groups in {format_time(load_time_comm)}.")

    graph_nodes_set: set[int] = set(G.nodes())
    nodes_not_in_graph_but_in_comm: set[int] = nodes_in_any_community - graph_nodes_set
    if nodes_not_in_graph_but_in_comm:
        logger.warning(f"{len(nodes_not_in_graph_but_in_comm)} nodes found in community file but not in graph edgelist for {name}.")

    # Create true_labels for nodes present in the graph G
    # Nodes in G but not in any community will get a default label.
    # Nodes in communities but not G are ignored for labeling G's nodes.
    
    node_list_graph: list[int] = list(G.nodes()) # Nodes for which we need labels
    node_to_final_idx: dict[int, int] = {node_id: i for i, node_id in enumerate(node_list_graph)}
    num_graph_nodes: int = len(node_list_graph)
    true_labels_list: list[int] = [-1] * num_graph_nodes # Default label for 'unassigned'

    # Assign community IDs. If overlapping, last seen comm_idx wins.
    # This uses the original comm_idx from the file as the label.
    actual_labels_assigned_count: int = 0
    for comm_idx, nodes_in_comm_set in raw_communities.items():
        for node_id in nodes_in_comm_set:
            if node_id in node_to_final_idx: # If this node from community file is in our graph
                final_node_idx: int = node_to_final_idx[node_id]
                if true_labels_list[final_node_idx] == -1: # First time assigning a label to this graph node
                    actual_labels_assigned_count +=1
                true_labels_list[final_node_idx] = comm_idx 

    # Handle graph nodes that were not in any community
    unassigned_nodes_count: int = true_labels_list.count(-1)
    if unassigned_nodes_count > 0:
        max_assigned_label_val: int = -1
        for lbl in true_labels_list:
            if lbl > max_assigned_label_val:
                max_assigned_label_val = lbl
        
        next_available_label: int = max_assigned_label_val + 1
        for i in range(num_graph_nodes):
            if true_labels_list[i] == -1:
                true_labels_list[i] = next_available_label
        logger.info(f"{unassigned_nodes_count} graph nodes were not in any listed community for {name}; assigned them to new label {next_available_label}.")
    
    logger.info(f"Assigned community labels to {actual_labels_assigned_count} graph nodes based on {name} community file.")
    for u,v in G.edges(): # Ensure weights
        if "weight" not in G.edges[u,v]:
             G.edges[u,v]["weight"] = 1.0
             
    return G, true_labels_list, True


def load_orkut() -> SimpleLoaderReturn:
    """Load the Orkut dataset from SNAP text files.

    Assumes files are downloaded and unzipped by download_datasets.py script.
    It expects datasets/com-orkut.ungraph.txt and datasets/com-orkut.all.cmty.txt

    Returns:
        A tuple as per `load_snap_text_dataset`.
            
    Raises:
        FileNotFoundError: If dataset files are not found.
    """
    return load_snap_text_dataset("Orkut", "com-orkut", "com-orkut")


def load_livejournal() -> SimpleLoaderReturn:
    """Load the LiveJournal dataset from SNAP text files.

    Assumes files are downloaded and unzipped by download_datasets.py script.
    It expects datasets/com-lj.ungraph.txt and datasets/com-lj.all.cmty.txt

    Returns:
        A tuple as per `load_snap_text_dataset`.
            
    Raises:
        FileNotFoundError: If dataset files are not found.
    """
    return load_snap_text_dataset("LiveJournal", "com-lj", "com-lj")


def load_graph_edges_gt_clusters(
    file_path: str = "datasets/graph_edges_gt_clusters.parquet",
    name: str = "Graph Edges GT Clusters",
    dtype_spec: dict[str, Any] | None = None
) -> DictLoaderReturn:
    """Load a graph and its ground truth communities from a single Parquet file.

    The Parquet file is expected to contain edge information ('src', 'dst', 'distance')
    and ground truth cluster information for the 'src' node ('src_cluster').

    Args:
        file_path: Path to the main Parquet file.
                         Expects 'src', 'dst', 'distance', 'src_cluster' columns.
        name: Name of the dataset.
        dtype_spec: Optional dictionary specifying dtypes for columns,
                     e.g., {'src': str, 'dst': str}. 'src' dtype will also apply
                     to the node IDs in ground truth derived from 'src' column.

    Returns:
        dict: A dictionary containing:
            - "graph": networkx.Graph object.
            - "ground_truth": Dictionary {node_id: community_id}.
            - "info": Dictionary with dataset statistics.
            - "error": String message if an error occurred, else None.
    """
    logger.info(f"Loading {name} from {file_path}...")
    start_time: float = time.time()
    
    # Import os at function level to avoid undefined variable in finally block
    import os
    
    edge_id_col_1: str = "src"
    edge_id_col_2: str = "dst"
    distance_col: str = "distance"
    gt_cluster_source_col: str = "src_cluster" # Column for GT cluster ID

    def _create_error_return(err_msg: str) -> DictLoaderReturn:
        return {
            "graph": None, "ground_truth": None,
            "info": {"name": name, "load_time_s": time.time() - start_time, "nodes":0, "edges":0, "has_ground_truth":False, "num_gt_clusters":0, "num_nodes_in_gt":0},
            "error": err_msg
        }

    if not os.path.exists(file_path):
        return _create_error_return(f"Edge file not found: {file_path}")

    try:
        # Force Polars to use all CPU cores for maximum performance
        original_threads = os.environ.get("POLARS_MAX_THREADS")
        cpu_count = os.cpu_count() or 4
        os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
        
        # Configure for optimal parallel processing in Polars 1.30.0+
        pl.Config.set_streaming_chunk_size(500_000)  # Smaller chunks = more parallelism
        
        # Enable aggressive memory optimization for large datasets
        original_verbose = pl.Config.state().get("verbose", False)
        pl.Config.set_verbose(False)  # Reduce memory overhead from logging
        
        # Use lazy loading and streaming for memory efficiency
        required_cols = [edge_id_col_1, edge_id_col_2, gt_cluster_source_col]
        
        # Check schema without loading data - scan_parquet is already optimized
        lf = pl.scan_parquet(file_path)
        schema = lf.schema
        
        if distance_col in schema:
            required_cols.append(distance_col)
        
        # Check essential columns early
        essential_cols = [edge_id_col_1, edge_id_col_2, gt_cluster_source_col]
        if not all(col in schema for col in essential_cols):
            missing = [col for col in essential_cols if col not in schema]
            return _create_error_return(f"Required columns missing in {file_path}: {missing}")
        
        # Build lazy computation pipeline for maximum efficiency - select only needed columns
        lf = lf.select(required_cols)
        
        # Apply dtype casting in lazy mode
        if dtype_spec:
            cast_exprs = []
            for col_name, target_type in dtype_spec.items():
                if col_name in required_cols:
                    if target_type is str:
                        cast_exprs.append(pl.col(col_name).cast(pl.Utf8))
            if cast_exprs:
                lf = lf.with_columns(cast_exprs)
        
        # Handle distance column and compute weights in lazy pipeline
        if distance_col not in lf.columns:
            lf = lf.with_columns(pl.lit(2.0).alias(distance_col))
        
        # Complete pipeline: nulls, types, and weight calculation with parallel processing
        lf = lf.with_columns([
            pl.col(distance_col).fill_null(2.0).cast(pl.Float32),  # Use Float32 for better memory efficiency
            pl.col(gt_cluster_source_col).fill_null(-1).cast(pl.Int32)
        ]).with_columns(
            # Use parallel computation for weight calculation - optimized for vectorization
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32), 
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32))
            ).alias("weight")
        )
        
        # Execute with parallel processing for maximum speed
        # Note: Using collect() with proper parallelization settings for Polars 1.30.0+
        logger.info(f"  Starting parallel processing with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,     # Enable type coercion optimizations
            predicate_pushdown=True, # Push filters down to file reading
            projection_pushdown=True, # Push column selection down to file reading
            simplify_expression=True, # Simplify expressions for better performance
            slice_pushdown=True,     # Push row limits down to file reading
            comm_subplan_elim=True,  # Eliminate common subplans
            comm_subexpr_elim=True   # Eliminate common subexpressions
        )
        logger.info(f"  Data processing completed. Building graph from {len(df)} edges...")
        
        # Initialize graph
        G: nx.Graph = nx.Graph()
        
        # Process large datasets in chunks to minimize memory usage
        chunk_size = 1_000_000  # Process 1M rows at a time
        total_rows = len(df)
        
        if total_rows > chunk_size:
            # Process in chunks for memory efficiency
            for i in range(0, total_rows, chunk_size):
                end_idx = min(i + chunk_size, total_rows)
                chunk = df.slice(i, end_idx - i)
                
                # Extract edges as tuples directly - fastest for NetworkX
                edges_chunk = chunk.select([edge_id_col_1, edge_id_col_2, "weight"]).rows()
                G.add_weighted_edges_from(edges_chunk)
        else:
            # Small dataset - process all at once
            edge_data = df.select([edge_id_col_1, edge_id_col_2, "weight"]).rows()
            G.add_weighted_edges_from(edge_data)

        # Build ground truth efficiently with deduplication
        gt_df = (df
                 .filter(pl.col(gt_cluster_source_col) != -1)
                 .select([edge_id_col_1, gt_cluster_source_col])
                 .unique(subset=[edge_id_col_1], keep="last"))  # Keep last occurrence
        
        # Convert to dict efficiently using Polars rows()
        gt_data_dict: dict[Any, int] = dict(gt_df.rows())

        num_nodes_val: int = G.number_of_nodes()
        num_edges_val: int = G.number_of_edges()
        
        num_gt_clusters_val: int = 0
        num_nodes_in_gt_val: int = 0
        has_gt: bool = False
        if gt_data_dict:
            num_gt_clusters_val = len(set(gt_data_dict.values()))
            num_nodes_in_gt_val = len(gt_data_dict)
            has_gt = True
            logger.info(f"  Ground truth from {file_path}: {num_gt_clusters_val} clusters for {num_nodes_in_gt_val} nodes.")
        else:
            logger.warning(f"  No ground truth data derived from {file_path}.")

        load_time_val: float = time.time() - start_time
        logger.info(f"  {name} loaded: {num_nodes_val} N, {num_edges_val} E. Took {format_time(load_time_val)}.")

        info_dict: dict[str, Any] = {
            "name": name, "nodes": num_nodes_val, "edges": num_edges_val,
            "has_ground_truth": has_gt, "num_gt_clusters": num_gt_clusters_val,
            "num_nodes_in_gt": num_nodes_in_gt_val, "load_time_s": load_time_val
        }
        return {"graph": G if num_nodes_val > 0 else None, "ground_truth": gt_data_dict if has_gt else None, "info": info_dict, "error": None}

    except Exception as e:
        logger.error(f"Error loading {name} from {file_path}: {e}", exc_info=True)
        return _create_error_return(str(e))
    finally:
        # Restore original settings
        if original_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = original_threads
        elif "POLARS_MAX_THREADS" in os.environ:
            del os.environ["POLARS_MAX_THREADS"]
        
        # Restore verbose setting if it was changed
        try:
            pl.Config.set_verbose(original_verbose)
        except:
            pass  # Ignore if original_verbose is not defined due to early error


def load_graph_edges_llm_clusters(
    file_path: str = "datasets/graph_edges_llm_clusters.parquet",
    name: str = "Graph Edges LLM Clusters",
    dtype_spec: dict[str, Any] | None = None
) -> DictLoaderReturn:
    """Load LLM-generated clusters graph and its ground truth communities from Parquet.

    This is a wrapper around `load_graph_edges_gt_clusters`, configured for
    the LLM cluster dataset file.

    Args:
        file_path: Path to the Parquet file (e.g.,
            'datasets/graph_edges_llm_clusters.parquet').
        name: Name of the dataset for logging.
        dtype_spec: Optional dictionary for column dtypes.

    Returns:
        A dictionary as per `load_graph_edges_gt_clusters`.
    """
    return load_graph_edges_gt_clusters(file_path=file_path, name=name, dtype_spec=dtype_spec)


def load_graph_edges_no_gt_polars(
    file_path: str = "datasets/graph_edges_no_gt.parquet",
    name: str = "Graph Edges No GT",
    dtype_spec: dict[str, Any] | None = None
) -> DictLoaderReturn:
    """Load a graph without ground truth from a Parquet file.

    Expects 'src', 'dst', and optionally 'distance' columns. Edge weights
    are similarity = 1 - (distance / 2.0), min 1e-9. 'distance'
    defaults to 2.0 if missing.

    Args:
        file_path: Path to the Parquet file.
        name: Name of the dataset for logging.
        dtype_spec: Optional dictionary for column dtypes.

    Returns:
        A dictionary with keys:
            - "graph" (nx.Graph | None): Loaded graph or None on error.
            - "ground_truth" (None): Always None for this loader.
            - "info" (dict[str, Any]): Dataset statistics and load time.
            - "error" (str | None): Error message if any, else None.
    """
    logger.info(f"Loading {name} (no GT) from {file_path}...")
    start_time_load_no_gt: float = time.time()
    
    # Import os for CPU optimization
    import os
    original_threads = os.environ.get("POLARS_MAX_THREADS")
    cpu_count = os.cpu_count() or 4
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    
    edge_id_col_1: str = "src"
    edge_id_col_2: str = "dst"
    distance_col: str = "distance"

    def _create_error_return_no_gt(err_msg: str) -> DictLoaderReturn:
        return {
            "graph": None, "ground_truth": None,
            "info": {"name": name, "load_time_s": time.time() - start_time_load_no_gt, "nodes":0, "edges":0, "has_ground_truth":False, "num_gt_clusters":0, "num_nodes_in_gt":0},
            "error": err_msg
        }

    try:
        if not os.path.exists(file_path):
            return _create_error_return_no_gt(f"File not found: {file_path}")

        # Use lazy Parquet reading for better performance
        lf = pl.scan_parquet(file_path)
        schema = lf.schema
        
        required_cols = [edge_id_col_1, edge_id_col_2]
        
        if distance_col in schema:
            required_cols.append(distance_col)
        
        # Check essential columns
        if not all(col in schema for col in [edge_id_col_1, edge_id_col_2]):
            missing_essential = [col for col in [edge_id_col_1, edge_id_col_2] if col not in schema]
            return _create_error_return_no_gt(f"Essential columns {missing_essential} missing in {file_path}")

        # Select only needed columns
        lf = lf.select(required_cols)
        
        # Apply dtype casting in lazy mode
        if dtype_spec:
            cast_exprs = []
            for col_name, target_type in dtype_spec.items():
                if col_name in required_cols:
                    if target_type is str:
                        cast_exprs.append(pl.col(col_name).cast(pl.Utf8))
            if cast_exprs:
                lf = lf.with_columns(cast_exprs)
        
        # Handle missing distance and compute similarity weights
        if distance_col not in lf.columns:
            lf = lf.with_columns(pl.lit(2.0, dtype=pl.Float32).alias(distance_col))
        
        lf = lf.with_columns([
            pl.col(distance_col).fill_null(2.0).cast(pl.Float32)
        ]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32))
            ).alias("weight")
        )
        
        # Collect with optimizations
        logger.info(f"  Processing {name} with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,
            predicate_pushdown=True,
            projection_pushdown=True,
            simplify_expression=True,
            slice_pushdown=True,
            comm_subplan_elim=True,
            comm_subexpr_elim=True
        )
        
        # Build graph efficiently with chunking if needed
        G_no_gt: nx.Graph = nx.Graph()
        total_rows = len(df)
        chunk_size = 1_000_000
        
        if total_rows > chunk_size:
            logger.info(f"  Processing {total_rows} edges in chunks...")
            for i in range(0, total_rows, chunk_size):
                chunk = df.slice(i, min(chunk_size, total_rows - i))
                edges_chunk = chunk.select([edge_id_col_1, edge_id_col_2, "weight"]).rows()
                G_no_gt.add_weighted_edges_from(edges_chunk)
        else:
            edge_data = df.select([edge_id_col_1, edge_id_col_2, "weight"]).rows()
            G_no_gt.add_weighted_edges_from(edge_data)

        num_nodes_val_no_gt: int = G_no_gt.number_of_nodes()
        num_edges_val_no_gt: int = G_no_gt.number_of_edges()

        load_time_val_no_gt: float = time.time() - start_time_load_no_gt
        logger.info(f"  {name} loaded: {num_nodes_val_no_gt} N, {num_edges_val_no_gt} E. Took {format_time(load_time_val_no_gt)}.")

        info_dict_no_gt: dict[str, Any] = {
            "name": name, "nodes": num_nodes_val_no_gt, "edges": num_edges_val_no_gt,
            "has_ground_truth": False, "num_gt_clusters": 0,
            "num_nodes_in_gt": 0, "load_time_s": load_time_val_no_gt
        }
        return {"graph": G_no_gt if num_nodes_val_no_gt > 0 else None, "ground_truth": None, "info": info_dict_no_gt, "error": None}

    except Exception as e:
        logger.error(f"Error loading {name} from {file_path}: {e}", exc_info=True)
        return _create_error_return_no_gt(str(e))
    finally:
        # Restore thread settings
        if original_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = original_threads
        elif "POLARS_MAX_THREADS" in os.environ:
            del os.environ["POLARS_MAX_THREADS"]

# Example usage (for testing individual loaders)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO) # Add basicConfig for standalone testing
#     # Test a simple loader
#     # G_karate, labels_karate, has_gt_karate = load_karate_club()
#     # logger.info(f"Karate Club: Nodes={G_karate.number_of_nodes()}, Edges={G_karate.number_of_edges()}, Has GT={has_gt_karate}, Labels Ex: {labels_karate[:5]}")

#     # Test a dict loader
#     # dataset_result_gt = load_graph_edges_gt_clusters()
#     # if dataset_result_gt["error"]:
#     #     logger.error(f"Error loading GT dataset: {dataset_result_gt['error']}")
#     # elif dataset_result_gt["graph"] is not None:
#     #     G_gt = dataset_result_gt["graph"]
#     #     logger.info(f"GT Dataset Info: {dataset_result_gt['info']}")
#     #     if dataset_result_gt["ground_truth"]:
#     #          logger.info(f"GT Sample (first 5): {list(dataset_result_gt['ground_truth'].items())[:5]}")


#     # dataset_result_no_gt = load_graph_edges_no_gt_polars()
#     # if dataset_result_no_gt["error"]:
#     #      logger.error(f"Error loading No-GT dataset: {dataset_result_no_gt['error']}")
#     # elif dataset_result_no_gt["graph"] is not None:
#     #      G_no_gt = dataset_result_no_gt["graph"]
#     #      logger.info(f"No-GT Dataset Info: {dataset_result_no_gt['info']}")
#     pass # End of example main 