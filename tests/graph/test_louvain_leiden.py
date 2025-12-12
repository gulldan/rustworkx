# Licensed under the Apache License, Version 2.0 (the "License");
import networkx as nx
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


def test_leiden_basic_connectivity():
    g, _ = _two_cliques_bridge()
    comms = rx.community.leiden_communities(g, resolution=1.0, seed=42)
    # At least one community, all nodes covered
    assert len(comms) >= 1
    assert set().union(*map(set, comms)) == set(range(8))
