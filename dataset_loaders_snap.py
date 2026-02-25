# ruff: noqa: E501
import logging
import os
import shutil
import time

import networkx as nx

from benchmark_utils import format_time
from dataset_loaders_shared import SimpleLoaderReturn

logger = logging.getLogger(__name__)


def load_snap_text_dataset(
    name: str, edge_file_base: str, community_file_base: str
) -> SimpleLoaderReturn:
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
    logger.info(
        f"Loaded {name} graph ({len(G.nodes())} N, {len(G.edges())} E) in {format_time(load_time_graph)}."
    )

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
                logger.warning(
                    f"Skipping invalid line in {community_file} (line {comm_idx + 1}): {line.strip()}"
                )
                continue

    load_time_comm: float = time.time() - start_time_comm
    num_raw_communities: int = len(raw_communities)
    logger.info(f"Loaded {num_raw_communities} raw community groups in {format_time(load_time_comm)}.")

    graph_nodes_set: set[int] = set(G.nodes())
    nodes_not_in_graph_but_in_comm: set[int] = nodes_in_any_community - graph_nodes_set
    if nodes_not_in_graph_but_in_comm:
        logger.warning(
            f"{len(nodes_not_in_graph_but_in_comm)} nodes found in community file but not in graph edgelist for {name}."
        )

    # Create true_labels for nodes present in the graph G
    # Nodes in G but not in any community will get a default label.
    # Nodes in communities but not G are ignored for labeling G's nodes.

    node_list_graph: list[int] = list(G.nodes())  # Nodes for which we need labels
    node_to_final_idx: dict[int, int] = {node_id: i for i, node_id in enumerate(node_list_graph)}
    num_graph_nodes: int = len(node_list_graph)
    true_labels_list: list[int] = [-1] * num_graph_nodes  # Default label for 'unassigned'

    # Assign community IDs. If overlapping, last seen comm_idx wins.
    # This uses the original comm_idx from the file as the label.
    actual_labels_assigned_count: int = 0
    for comm_idx, nodes_in_comm_set in raw_communities.items():
        for node_id in nodes_in_comm_set:
            if node_id in node_to_final_idx:  # If this node from community file is in our graph
                final_node_idx: int = node_to_final_idx[node_id]
                if (
                    true_labels_list[final_node_idx] == -1
                ):  # First time assigning a label to this graph node
                    actual_labels_assigned_count += 1
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
        logger.info(
            f"{unassigned_nodes_count} graph nodes were not in any listed community for {name}; assigned them to new label {next_available_label}."
        )

    logger.info(
        f"Assigned community labels to {actual_labels_assigned_count} graph nodes based on {name} community file."
    )
    for u, v in G.edges():  # Ensure weights
        if "weight" not in G.edges[u, v]:
            G.edges[u, v]["weight"] = 1.0

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
