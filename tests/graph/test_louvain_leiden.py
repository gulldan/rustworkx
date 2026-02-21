# Licensed under the Apache License, Version 2.0 (the "License");
import networkx as nx
import pytest
import rustworkx as rx


def _two_cliques_bridge(n_each=8):
    g = rx.PyGraph()
    g.add_nodes_from(range(n_each))
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges += [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
    edges += [(3, 4)]
    for u, v in edges:
        g.add_edge(u, v, 1.0)
    return g, edges


def _partition_signature(communities):
    return {frozenset(comm) for comm in communities if comm}


def _run_leidenalg_on_weighted_edges(num_nodes, weighted_edges):
    igraph = pytest.importorskip("igraph")
    leidenalg = pytest.importorskip("leidenalg")

    ig_graph = igraph.Graph(
        n=num_nodes,
        edges=[(u, v) for u, v, _ in weighted_edges],
        directed=False,
    )
    ig_graph.es["weight"] = [float(weight) for _, _, weight in weighted_edges]

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=42,
    )
    return [set(comm) for comm in partition]


def _run_rx_leiden_on_weighted_edges(num_nodes, weighted_edges):
    graph = rx.PyGraph(multigraph=True)
    graph.add_nodes_from(range(num_nodes))
    for u, v, weight in weighted_edges:
        graph.add_edge(u, v, float(weight))

    communities = rx.community.leiden_communities(
        graph, weight_fn=lambda edge: float(edge), resolution=1.0, seed=42
    )
    return [set(comm) for comm in communities]


def test_louvain_matches_nx_count():
    g, edges = _two_cliques_bridge()
    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(8))
    nx_g.add_edges_from(edges)

    seed = 42
    nx_comms = nx.community.louvain_communities(nx_g, weight="weight", seed=seed)
    rx_comms = rx.community.louvain_communities(g, weight_fn=None, seed=seed, resolution=1.0)

    assert len(nx_comms) == len(rx_comms)
    assert set().union(*nx_comms) == set(range(8))
    assert set().union(*map(set, rx_comms)) == set(range(8))
    assert {frozenset(c) for c in nx_comms} == {frozenset(c) for c in map(set, rx_comms)}


def test_leiden_basic_connectivity():
    g, _ = _two_cliques_bridge()
    comms = rx.community.leiden_communities(g, resolution=1.0, seed=42)
    # At least one community, all nodes covered
    assert len(comms) >= 1
    assert set().union(*map(set, comms)) == set(range(8))


def test_leiden_matches_leidenalg_on_weighted_graph():
    weighted_edges = [
        (0, 1, 1.5),
        (0, 2, 0.8),
        (1, 2, 1.2),
        (2, 3, 0.4),
        (3, 4, 1.7),
        (3, 5, 1.1),
        (4, 5, 1.3),
        (5, 6, 0.6),
        (6, 7, 1.4),
        (7, 8, 1.6),
        (8, 9, 1.0),
        (6, 9, 0.9),
        (9, 10, 0.7),
        (10, 11, 1.8),
        (11, 12, 1.2),
        (12, 13, 1.0),
        (13, 10, 1.1),
        (2, 10, 0.3),
        (1, 8, 0.2),
        (4, 12, 0.5),
    ]
    rx_partition = _run_rx_leiden_on_weighted_edges(14, weighted_edges)
    leidenalg_partition = _run_leidenalg_on_weighted_edges(14, weighted_edges)

    assert _partition_signature(rx_partition) == _partition_signature(leidenalg_partition)


def test_leiden_matches_leidenalg_with_self_loops():
    weighted_edges = [
        (0, 0, 0.9),
        (1, 1, 1.1),
        (2, 2, 0.7),
        (0, 1, 1.4),
        (1, 2, 1.3),
        (2, 3, 1.2),
        (3, 4, 1.5),
        (4, 5, 1.6),
        (5, 0, 1.0),
        (2, 5, 0.8),
        (1, 4, 0.6),
        (3, 6, 1.7),
        (6, 7, 1.9),
        (7, 8, 1.3),
        (8, 6, 1.0),
    ]
    rx_partition = _run_rx_leiden_on_weighted_edges(9, weighted_edges)
    leidenalg_partition = _run_leidenalg_on_weighted_edges(9, weighted_edges)

    assert _partition_signature(rx_partition) == _partition_signature(leidenalg_partition)
