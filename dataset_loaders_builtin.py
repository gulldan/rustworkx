# ruff: noqa: E501
import logging
import os
from typing import Any

import networkx as nx
import numpy as np

from dataset_loaders_shared import (
    SimpleLoaderReturn,
    _ensure_default_edge_weight,
    _load_gml_with_ground_truth,
)

logger = logging.getLogger(__name__)


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
    _ensure_default_edge_weight(G)
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
    _ensure_default_edge_weight(G)
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
        "Myriel": 0,
        "Napoleon": 0,
        "MlleBaptistine": 0,
        "MmeMagloire": 0,
        "CountessDeLo": 0,
        "Geborand": 0,
        "Champtercier": 0,
        "Cravatte": 0,
        "Count": 0,
        "OldMan": 0,
        "Valjean": 1,
        "Labarre": 1,
        "Marguerite": 1,
        "MmeDeR": 1,
        "Isabeau": 1,
        "Gervais": 1,
        "Tholomyes": 2,
        "Listolier": 2,
        "Fameuil": 2,
        "Blacheville": 2,
        "Favourite": 2,
        "Dahlia": 2,
        "Zephine": 2,
        "Fantine": 2,
        "MmeThenardier": 3,
        "Thenardier": 3,
        "Cosette": 3,
        "Javert": 4,
        "Fauchelevent": 4,
        "Bamatabois": 4,
        "Perpetue": 4,
        "Simplice": 4,
        "Scaufflaire": 4,
        "Woman1": 4,
        "Judge": 4,
        "Champmathieu": 4,
        "Brevet": 4,
        "Chenildieu": 4,
        "Cochepaille": 4,
        "Pontmercy": 5,
        "Boulatruelle": 5,
        "Eponine": 5,
        "Anzelma": 5,
        "Woman2": 5,
        "MotherInnocent": 6,
        "Gribier": 6,
        "Jondrette": 7,
        "MmeBurgon": 7,
        "Gavroche": 7,
        "Gillenormand": 8,
        "Magnon": 8,
        "MlleGillenormand": 8,
        "MmePontmercy": 8,
        "MlleVaubois": 8,
        "LtGillenormand": 8,
        "Marius": 8,
        "BaronessT": 8,
        "Mabeuf": 9,
        "Enjolras": 9,
        "Combeferre": 9,
        "Prouvaire": 9,
        "Feuilly": 9,
        "Courfeyrac": 9,
        "Bahorel": 9,
        "Bossuet": 9,
        "Joly": 9,
        "Grantaire": 9,
        "MotherPlutarch": 9,
        "Gueulemer": 10,
        "Babet": 10,
        "Claquesous": 10,
        "Montparnasse": 10,
        "Toussaint": 11,
        "Child1": 11,
        "Child2": 11,
        "Brujon": 11,
        "MmeHucheloup": 11,
    }

    pos: dict[Any, np.ndarray] = nx.spring_layout(
        G_orig, k=1 / np.sqrt(len(G_orig.nodes())), iterations=50, seed=42
    )
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
    return _load_gml_with_ground_truth(file_path, lambda attrs: attrs["value"])


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
    return _load_gml_with_ground_truth(file_path, lambda attrs: attrs["value"])


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
    return _load_gml_with_ground_truth(file_path, lambda attrs: attrs["value"])


def load_polblogs() -> SimpleLoaderReturn:
    """Load Political Blogs dataset with ground truth communities.

    Assumes file 'datasets/polblogs.gml' exists and that node attribute 'value'
    contains the ground truth community. Edges are assigned weight 1.0
    if not specified.

    Returns:
        A tuple containing:
            - nx.Graph: The Political Blogs graph (undirected).
            - list[int]: A list of true community labels.
            - bool: True, indicating ground truth is available.

    Raises:
        FileNotFoundError: If 'datasets/polblogs.gml' is not found.
    """
    file_path: str = "datasets/polblogs.gml"
    return _load_gml_with_ground_truth(
        file_path,
        lambda attrs: attrs["value"],
        to_graph_edges_fn=lambda path: nx.Graph(nx.read_gml(path, label="id")),
    )


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
    return _load_gml_with_ground_truth(file_path, lambda attrs: attrs["value"])


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
    return _load_gml_with_ground_truth(file_path, lambda attrs: attrs["value"])


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
    return _load_gml_with_ground_truth(file_path, lambda attrs: attrs.get("value", 0))


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
    return _load_gml_with_ground_truth(file_path, lambda attrs: attrs.get("value", 0))
