# ruff: noqa: E501
import logging
import os
import time
from typing import Any

import networkx as nx
import polars as pl

from benchmark_utils import format_time
from dataset_loaders_shared import (
    DictLoaderReturn,
    SimpleLoaderReturn,
    _add_weighted_edges_from_df,
)

logger = logging.getLogger(__name__)


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

        lf = lf.with_columns([pl.col(distance_col).fill_null(2.0).cast(pl.Float32)]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32)),
            ).alias("weight")
        )

        # Collect with optimizations
        logger.info(f"  Processing CSV with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,
            predicate_pushdown=True,
            projection_pushdown=True,
            simplify_expression=True,
        )

        # Build graph efficiently
        G: nx.Graph = nx.Graph()
        _add_weighted_edges_from_df(df, G)

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

        lf = lf.with_columns([pl.col(distance_col).fill_null(2.0).cast(pl.Float32)]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32)),
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
            comm_subexpr_elim=True,
        )

        # Build graph efficiently with chunking for large datasets
        G: nx.Graph = nx.Graph()
        _add_weighted_edges_from_df(df, G, chunk_size=1_000_000)

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

        lf = lf.with_columns([pl.col(distance_col).fill_null(2.0).cast(pl.Float32)]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32)),
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
            comm_subexpr_elim=True,
        )

        # Build graph efficiently with chunking for 9M dataset
        G: nx.Graph = nx.Graph()
        _add_weighted_edges_from_df(
            df,
            G,
            chunk_size=500_000,
            progress_every=10,
            progress_prefix="  ",
        )

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


def load_graph_edges_gt_clusters(
    file_path: str = "datasets/graph_edges_gt_clusters.parquet",
    name: str = "Graph Edges GT Clusters",
    dtype_spec: dict[str, Any] | None = None,
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

    edge_id_col_1: str = "src"
    edge_id_col_2: str = "dst"
    distance_col: str = "distance"
    gt_cluster_source_col: str = "src_cluster"  # Column for GT cluster ID

    def _create_error_return(err_msg: str) -> DictLoaderReturn:
        return {
            "graph": None,
            "ground_truth": None,
            "info": {
                "name": name,
                "load_time_s": time.time() - start_time,
                "nodes": 0,
                "edges": 0,
                "has_ground_truth": False,
                "num_gt_clusters": 0,
                "num_nodes_in_gt": 0,
            },
            "error": err_msg,
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
        lf = lf.with_columns(
            [
                pl.col(distance_col)
                .fill_null(2.0)
                .cast(pl.Float32),  # Use Float32 for better memory efficiency
                pl.col(gt_cluster_source_col).fill_null(-1).cast(pl.Int32),
            ]
        ).with_columns(
            # Use parallel computation for weight calculation - optimized for vectorization
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32)),
            ).alias("weight")
        )

        # Execute with parallel processing for maximum speed
        # Note: Using collect() with proper parallelization settings for Polars 1.30.0+
        logger.info(f"  Starting parallel processing with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,  # Enable type coercion optimizations
            predicate_pushdown=True,  # Push filters down to file reading
            projection_pushdown=True,  # Push column selection down to file reading
            simplify_expression=True,  # Simplify expressions for better performance
            slice_pushdown=True,  # Push row limits down to file reading
            comm_subplan_elim=True,  # Eliminate common subplans
            comm_subexpr_elim=True,  # Eliminate common subexpressions
        )
        logger.info(f"  Data processing completed. Building graph from {len(df)} edges...")

        # Initialize graph
        G: nx.Graph = nx.Graph()
        _add_weighted_edges_from_df(
            df,
            G,
            source_col=edge_id_col_1,
            dst_col=edge_id_col_2,
            chunk_size=1_000_000,
            progress_prefix="  ",
        )

        # Build ground truth efficiently with deduplication
        gt_df = (
            df.filter(pl.col(gt_cluster_source_col) != -1)
            .select([edge_id_col_1, gt_cluster_source_col])
            .unique(subset=[edge_id_col_1], keep="last")
        )  # Keep last occurrence

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
            logger.info(
                f"  Ground truth from {file_path}: {num_gt_clusters_val} clusters for {num_nodes_in_gt_val} nodes."
            )
        else:
            logger.warning(f"  No ground truth data derived from {file_path}.")

        load_time_val: float = time.time() - start_time
        logger.info(
            f"  {name} loaded: {num_nodes_val} N, {num_edges_val} E. Took {format_time(load_time_val)}."
        )

        info_dict: dict[str, Any] = {
            "name": name,
            "nodes": num_nodes_val,
            "edges": num_edges_val,
            "has_ground_truth": has_gt,
            "num_gt_clusters": num_gt_clusters_val,
            "num_nodes_in_gt": num_nodes_in_gt_val,
            "load_time_s": load_time_val,
        }
        return {
            "graph": G if num_nodes_val > 0 else None,
            "ground_truth": gt_data_dict if has_gt else None,
            "info": info_dict,
            "error": None,
        }

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
        except Exception:
            pass  # Ignore if original_verbose is not defined due to early error


def load_graph_edges_llm_clusters(
    file_path: str = "datasets/graph_edges_llm_clusters.parquet",
    name: str = "Graph Edges LLM Clusters",
    dtype_spec: dict[str, Any] | None = None,
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


def load_wiki_news_edges() -> SimpleLoaderReturn:
    """Load graph from Wiki News edges Parquet file, calculate similarity weight from distance.

    Expects 'src', 'dst', and optionally 'distance' columns.
    Edge weights are similarity = 1 - (distance / 2.0), min 1e-9.
    'distance' defaults to 2.0 if missing. No ground truth; dummy labels created.

    Returns:
        A tuple containing:
            - nx.Graph: The loaded graph.
            - list[int]: Dummy true community labels (all zeros).
            - bool: False (no actual ground truth).

    Raises:
        FileNotFoundError: If 'datasets/wiki_news_edges_sim_thresh_0_9.parquet' is not found.
        pl.exceptions.ComputeError: If Polars has issues reading the Parquet.
    """
    file_path: str = "datasets/wiki_news_edges_sim_thresh_0_9.parquet"
    logger.info(f"Loading Wiki News graph from {file_path}...")

    # Import os for CPU optimization

    original_threads = os.environ.get("POLARS_MAX_THREADS")
    cpu_count = os.cpu_count() or 4
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Use lazy Parquet reading for better performance
        lf = pl.scan_parquet(file_path)
        schema = lf.collect_schema()
        schema_names = set(schema.names())

        required_cols = ["src", "dst"]
        distance_col = "distance"
        similarity_col = "similarity"
        src_cluster_col = "src_cluster"
        dst_cluster_col = "dst_cluster"

        if similarity_col in schema_names:
            required_cols.append(similarity_col)
        elif distance_col in schema_names:
            required_cols.append(distance_col)
        # Cluster columns (if present) for GT extraction
        if src_cluster_col in schema_names:
            required_cols.append(src_cluster_col)
        if dst_cluster_col in schema_names:
            required_cols.append(dst_cluster_col)

        # Select only needed columns
        lf = lf.select(required_cols)

        # Compute weight column
        if similarity_col in lf.columns:
            # Use provided similarity directly as weight, enforce minimum 1e-9
            lf = lf.with_columns(
                pl.max_horizontal(
                    pl.lit(1e-9, dtype=pl.Float32),
                    pl.col(similarity_col).fill_null(1.0).cast(pl.Float32),
                ).alias("weight")
            )
        else:
            # Fall back to converting distance -> similarity
            if distance_col not in lf.columns:
                lf = lf.with_columns(pl.lit(2.0, dtype=pl.Float32).alias(distance_col))
            lf = lf.with_columns([pl.col(distance_col).fill_null(2.0).cast(pl.Float32)]).with_columns(
                pl.max_horizontal(
                    pl.lit(1e-9, dtype=pl.Float32),
                    (
                        pl.lit(1.0, dtype=pl.Float32)
                        - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32)
                    ),
                ).alias("weight")
            )

        # Collect with optimizations
        logger.info(f"  Processing Wiki News Parquet with {cpu_count} CPU cores...")
        df = lf.collect(
            type_coercion=True,
            predicate_pushdown=True,
            projection_pushdown=True,
            simplify_expression=True,
            slice_pushdown=True,
            comm_subplan_elim=True,
            comm_subexpr_elim=True,
        )

        # Build graph efficiently with chunking for large datasets
        G: nx.Graph = nx.Graph()
        _add_weighted_edges_from_df(
            df,
            G,
            source_col="src",
            dst_col="dst",
            chunk_size=1_000_000,
            progress_every=5,
            progress_prefix="    ",
        )

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        logger.info(
            f"Created Wiki News graph with {num_nodes} nodes and {num_edges} edges from Parquet."
        )

        # Build ground truth from cluster columns if present
        gt_map: dict[Any, int] = {}
        df_cols = set(df.columns)
        if src_cluster_col in df_cols:
            for s, sc in df.select(["src", src_cluster_col]).rows():
                if sc is not None:
                    gt_map[s] = int(sc)
        if dst_cluster_col in df_cols:
            for d, dc in df.select(["dst", dst_cluster_col]).rows():
                if dc is not None:
                    gt_map[d] = int(dc)

        if gt_map:
            logger.info(f"  Derived ground truth for {len(gt_map)} nodes from cluster columns.")
            return G, gt_map, True
        else:
            # Fallback: dummy labels
            true_labels: list[int]
            if num_nodes > 0:
                true_labels = [0] * num_nodes
                logger.warning(
                    "No ground truth columns found for Wiki News dataset. Using dummy labels."
                )
            else:
                true_labels = []
                logger.warning("Wiki News graph is empty. No labels.")
            return G, true_labels, False

    except Exception as e:
        logger.error(f"Error loading Wiki News graph from {file_path}: {e}", exc_info=True)
        G = nx.Graph()
        G.add_node(0)
        return G, [0], False
    finally:
        # Restore thread settings
        if original_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = original_threads
        elif "POLARS_MAX_THREADS" in os.environ:
            del os.environ["POLARS_MAX_THREADS"]


def load_graph_edges_no_gt_polars(
    file_path: str = "datasets/graph_edges_no_gt.parquet",
    name: str = "Graph Edges No GT",
    dtype_spec: dict[str, Any] | None = None,
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

    original_threads = os.environ.get("POLARS_MAX_THREADS")
    cpu_count = os.cpu_count() or 4
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)

    edge_id_col_1: str = "src"
    edge_id_col_2: str = "dst"
    distance_col: str = "distance"

    def _create_error_return_no_gt(err_msg: str) -> DictLoaderReturn:
        return {
            "graph": None,
            "ground_truth": None,
            "info": {
                "name": name,
                "load_time_s": time.time() - start_time_load_no_gt,
                "nodes": 0,
                "edges": 0,
                "has_ground_truth": False,
                "num_gt_clusters": 0,
                "num_nodes_in_gt": 0,
            },
            "error": err_msg,
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
            return _create_error_return_no_gt(
                f"Essential columns {missing_essential} missing in {file_path}"
            )

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

        lf = lf.with_columns([pl.col(distance_col).fill_null(2.0).cast(pl.Float32)]).with_columns(
            pl.max_horizontal(
                pl.lit(1e-9, dtype=pl.Float32),
                (pl.lit(1.0, dtype=pl.Float32) - pl.col(distance_col) * pl.lit(0.5, dtype=pl.Float32)),
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
            comm_subexpr_elim=True,
        )

        # Build graph efficiently with chunking if needed
        G_no_gt: nx.Graph = nx.Graph()
        _add_weighted_edges_from_df(
            df,
            G_no_gt,
            source_col=edge_id_col_1,
            dst_col=edge_id_col_2,
            chunk_size=1_000_000,
            progress_prefix="  ",
        )

        num_nodes_val_no_gt: int = G_no_gt.number_of_nodes()
        num_edges_val_no_gt: int = G_no_gt.number_of_edges()

        load_time_val_no_gt: float = time.time() - start_time_load_no_gt
        logger.info(
            f"  {name} loaded: {num_nodes_val_no_gt} N, {num_edges_val_no_gt} E. Took {format_time(load_time_val_no_gt)}."
        )

        info_dict_no_gt: dict[str, Any] = {
            "name": name,
            "nodes": num_nodes_val_no_gt,
            "edges": num_edges_val_no_gt,
            "has_ground_truth": False,
            "num_gt_clusters": 0,
            "num_nodes_in_gt": 0,
            "load_time_s": load_time_val_no_gt,
        }
        return {
            "graph": G_no_gt if num_nodes_val_no_gt > 0 else None,
            "ground_truth": None,
            "info": info_dict_no_gt,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error loading {name} from {file_path}: {e}", exc_info=True)
        return _create_error_return_no_gt(str(e))
    finally:
        # Restore thread settings
        if original_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = original_threads
        elif "POLARS_MAX_THREADS" in os.environ:
            del os.environ["POLARS_MAX_THREADS"]
