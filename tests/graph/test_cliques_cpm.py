# Licensed under the Apache License, Version 2.0 (the "License");
import networkx as nx
import rustworkx as rx


def _to_rx_graph(nx_graph: nx.Graph) -> rx.PyGraph:
    graph = rx.PyGraph()
    graph.add_nodes_from(range(nx_graph.number_of_nodes()))
    graph.add_edges_from_no_data(list(nx_graph.edges()))
    return graph


def _partition_set(communities):
    return {frozenset(c) for c in communities if c}


def test_find_maximal_cliques_matches_networkx_karate():
    nx_graph = nx.karate_club_graph()
    rx_graph = _to_rx_graph(nx_graph)

    nx_cliques = _partition_set(nx.find_cliques(nx_graph))
    rx_cliques = _partition_set(rx.community.find_maximal_cliques(rx_graph))

    assert rx_cliques == nx_cliques


def test_find_maximal_cliques_ignores_self_loops():
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(3))
    # Self-loops should not affect maximal clique enumeration.
    nx_graph.add_edges_from([(0, 1), (1, 1), (1, 2), (2, 2)])
    rx_graph = _to_rx_graph(nx_graph)

    nx_cliques = _partition_set(nx.find_cliques(nx_graph))
    rx_cliques = _partition_set(rx.community.find_maximal_cliques(rx_graph))

    assert rx_cliques == nx_cliques


def test_cpm_matches_networkx_karate_k3():
    nx_graph = nx.karate_club_graph()
    rx_graph = _to_rx_graph(nx_graph)

    nx_comms = _partition_set(nx.community.k_clique_communities(nx_graph, 3))
    rx_comms = _partition_set(rx.community.cpm_communities(rx_graph, 3))

    assert rx_comms == nx_comms


def test_cpm_ignores_self_loops():
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(4))
    nx_graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 2), (3, 3)])
    rx_graph = _to_rx_graph(nx_graph)

    nx_comms = _partition_set(nx.community.k_clique_communities(nx_graph, 3))
    rx_comms = _partition_set(rx.community.cpm_communities(rx_graph, 3))

    assert rx_comms == nx_comms
