# Licensed under the Apache License, Version 2.0 (the "License");
import networkx
import pytest
import rustworkx as rx


def _has_strongest() -> bool:
    return hasattr(networkx.community, "asyn_lpa_communities_strongest")


def test_lpa_empty_graph():
    g = rx.PyGraph()
    res = rx.community.asyn_lpa_communities(g)
    assert res == []


def test_lpa_single_node():
    g = rx.PyGraph()
    g.add_node(0)
    res = rx.community.asyn_lpa_communities(g)
    assert len(res) == 1
    assert res[0] == [0]


def test_lpa_two_communities_matches_nx_count():
    # Build two cliques connected by one edge
    g = rx.PyGraph()
    g.add_nodes_from(range(8))
    # clique 0-3
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # clique 4-7
    edges += [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
    # bridge
    edges += [(3, 4)]
    for u, v in edges:
        g.add_edge(u, v, None)

    # Build the same for NetworkX
    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(8))
    nx_g.add_edges_from(edges)

    seed = 42
    nx_comms = list(nx.community.asyn_lpa_communities(nx_g, seed=seed))
    rx_comms = rx.community.asyn_lpa_communities(g, seed=seed)

    assert len(nx_comms) == len(rx_comms)
    # All nodes covered
    assert set().union(*nx_comms) == set(range(8))
    assert set().union(*map(set, rx_comms)) == set(range(8))


def test_lpa_weighted_bridge():
    # Two triangles with weak bridge should yield 2 communities
    g = rx.PyGraph()
    g.add_nodes_from(range(6))
    # left triangle
    g.add_edge(0, 1, {"weight": 2.0})
    g.add_edge(1, 2, {"weight": 2.0})
    g.add_edge(2, 0, {"weight": 2.0})
    # right triangle
    g.add_edge(3, 4, {"weight": 2.0})
    g.add_edge(4, 5, {"weight": 2.0})
    g.add_edge(5, 3, {"weight": 2.0})
    # weak bridge
    g.add_edge(2, 3, {"weight": 0.25})

    res = rx.community.asyn_lpa_communities(g, weight="weight", seed=42)
    assert len(res) == 2
    sets = list(map(set, res))
    assert set(range(0, 3)) in sets
    assert set(range(3, 6)) in sets


def test_lpa_missing_weight_defaults_to_1():
    g = rx.PyGraph()
    g.add_nodes_from(range(4))
    g.add_edge(0, 1, {"other": 1.0})
    g.add_edge(1, 2, {"other": 1.0})
    g.add_edge(2, 3, {"other": 1.0})
    # Should not crash and should cover all nodes
    res = rx.community.asyn_lpa_communities(g, weight="weight", seed=42)
    assert set().union(*map(set, res)) == set(range(4))


def test_lpa_strongest_empty_graph():
    g = rx.PyGraph()
    res = rx.community.asyn_lpa_communities_strongest(g)
    assert res == []


def test_lpa_strongest_single_node():
    g = rx.PyGraph()
    g.add_node(0)
    res = rx.community.asyn_lpa_communities_strongest(g)
    assert res == [[0]]


@pytest.mark.skipif(
    not _has_strongest(),
    reason="NetworkX does not provide asyn_lpa_communities_strongest",
)
def test_lpa_strongest_matches_reference_on_weighted_graph():
    g = rx.PyGraph()
    g.add_nodes_from(range(4))
    # edges: duplicate weights to ensure tie handling matches reference implementation
    g.add_edge(0, 1, {"weight": 3.0})
    g.add_edge(0, 2, {"weight": 1.0})
    g.add_edge(1, 3, {"weight": 3.0})
    g.add_edge(2, 3, {"weight": 0.5})

    # Build equivalent networkx graph
    nx_g = networkx.Graph()
    nx_g.add_nodes_from(range(4))
    nx_g.add_edge(0, 1, weight=3.0)
    nx_g.add_edge(0, 2, weight=1.0)
    nx_g.add_edge(1, 3, weight=3.0)
    nx_g.add_edge(2, 3, weight=0.5)

    seed = 5
    rx_result = rx.community.asyn_lpa_communities_strongest(g, weight="weight", seed=seed)
    nx_result = list(networkx.community.asyn_lpa_communities_strongest(nx_g, weight="weight", seed=seed))

    assert sorted(map(sorted, rx_result)) == sorted(map(sorted, nx_result))


@pytest.mark.skipif(
    not _has_strongest(),
    reason="NetworkX does not provide asyn_lpa_communities_strongест",
)
def test_lpa_strongest_tie_breaking_unweighted():
    g = rx.PyGraph()
    g.add_nodes_from(range(5))
    edges = [(0, 1), (0, 2), (0, 3), (3, 4)]
    g.add_edges_from((u, v, None) for u, v in edges)

    nx_g = networkx.Graph()
    nx_g.add_nodes_from(range(5))
    nx_g.add_edges_from(edges)

    seed = 7
    rx_result = rx.community.asyn_lpa_communities_strongest(g, seed=seed)
    nx_result = list(networkx.community.asyn_lpa_communities_strongest(nx_g, seed=seed))

    assert sorted(map(sorted, rx_result)) == sorted(map(sorted, nx_result))
