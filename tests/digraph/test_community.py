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

import unittest

import networkx
import pytest
import rustworkx
from rustworkx import PyDiGraph


def _has_strongest() -> bool:
    return hasattr(networkx.community, "asyn_lpa_communities_strongest")


class TestDiGraphCommunity(unittest.TestCase):
    """Tests for community detection algorithms on directed graphs."""

    def test_empty_digraph(self):
        """Test asyn_lpa_communities on an empty directed graph."""
        g = PyDiGraph()
        res = rustworkx.community.asyn_lpa_communities(g)
        self.assertEqual(len(res), 0)

    def test_single_node_digraph(self):
        """Test asyn_lpa_communities on a single-node directed graph."""
        g = PyDiGraph()
        g.add_node(0)
        res = rustworkx.community.asyn_lpa_communities(g)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], [0])

    def test_disconnected_nodes_digraph(self):
        """Test asyn_lpa_communities on disconnected nodes in directed graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(5))
        res = rustworkx.community.asyn_lpa_communities(g)
        self.assertEqual(len(res), 5)
        for i in range(5):
            self.assertIn([i], res)

    def test_simple_communities_digraph(self):
        """Test asyn_lpa_communities on a directed graph with two obvious communities."""
        g = PyDiGraph()
        g.add_nodes_from(range(8))

        # Community 1: 0->1->2->3->0 (cycle)
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 3, None)
        g.add_edge(3, 0, None)

        # Community 2: 4->5->6->7->4 (cycle)
        g.add_edge(4, 5, None)
        g.add_edge(5, 6, None)
        g.add_edge(6, 7, None)
        g.add_edge(7, 4, None)

        # Weak connection between communities
        g.add_edge(3, 4, None)

        res = rustworkx.community.asyn_lpa_communities(g, seed=42)

        # Should have at least one community
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(8)))

    def test_weighted_digraph(self):
        """Test asyn_lpa_communities with weighted edges in directed graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(6))

        # Create a weighted directed graph where weights influence community formation
        g.add_edge(0, 1, {"weight": 3.0})
        g.add_edge(1, 2, {"weight": 3.0})
        g.add_edge(2, 0, {"weight": 3.0})

        g.add_edge(3, 4, {"weight": 3.0})
        g.add_edge(4, 5, {"weight": 3.0})
        g.add_edge(5, 3, {"weight": 3.0})

        # Weak connection between communities
        g.add_edge(2, 3, {"weight": 0.1})

        res = rustworkx.community.asyn_lpa_communities(g, weight="weight", seed=42)

        # Should have at least one community
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(6)))

    def test_directed_chain(self):
        """Test asyn_lpa_communities on a directed chain."""
        g = PyDiGraph()
        g.add_nodes_from(range(5))

        # Create a chain: 0->1->2->3->4
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 3, None)
        g.add_edge(3, 4, None)

        res = rustworkx.community.asyn_lpa_communities(g, seed=42)

        # Should have at least one community
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(5)))

    def test_directed_star(self):
        """Test asyn_lpa_communities on a directed star graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(5))

        # Create a star: center node 0, all others point to it
        g.add_edge(1, 0, None)
        g.add_edge(2, 0, None)
        g.add_edge(3, 0, None)
        g.add_edge(4, 0, None)

        res = rustworkx.community.asyn_lpa_communities(g, seed=42)

        # Should have at least one community
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(5)))

    def test_directed_cycle(self):
        """Test asyn_lpa_communities on a directed cycle."""
        g = PyDiGraph()
        g.add_nodes_from(range(6))

        # Create a cycle: 0->1->2->3->4->5->0
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 3, None)
        g.add_edge(3, 4, None)
        g.add_edge(4, 5, None)
        g.add_edge(5, 0, None)

        res = rustworkx.community.asyn_lpa_communities(g, seed=42)

        # Should have at least one community
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(6)))

    def test_directed_multiple_edges(self):
        """Test asyn_lpa_communities with multiple edges between same nodes in directed graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(4))

        # Add multiple edges between 0 and 1
        g.add_edge(0, 1, {"weight": 1.0})
        g.add_edge(0, 1, {"weight": 1.0})
        g.add_edge(0, 1, {"weight": 1.0})

        # Single edge between 1 and 2
        g.add_edge(1, 2, {"weight": 1.0})

        # Multiple edges between 2 and 3
        g.add_edge(2, 3, {"weight": 1.0})
        g.add_edge(2, 3, {"weight": 1.0})

        res = rustworkx.community.asyn_lpa_communities(g, weight="weight", seed=42)

        # Should form communities based on edge weights (summed)
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(4)))

    def test_directed_isolated_nodes(self):
        """Test asyn_lpa_communities with isolated nodes in directed graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(5))

        # Add edges only between nodes 0, 1, 2
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 0, None)

        # Nodes 3 and 4 are isolated

        res = rustworkx.community.asyn_lpa_communities(g, seed=42)

        # Should have 3 communities: {0,1,2}, {3}, {4}
        self.assertEqual(len(res), 3)

        # Convert to sets for easy comparison
        comm_sets = [set(comm) for comm in res]

        # Check that isolated nodes are in their own communities
        self.assertIn({3}, comm_sets)
        self.assertIn({4}, comm_sets)

        # Check that connected nodes form a community
        connected_comm = set(range(3))
        self.assertIn(connected_comm, comm_sets)

    def test_directed_edge_weight_dict_format(self):
        """Test asyn_lpa_communities with edge weights stored as dictionaries in directed graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(4))

        # Add edges with weights in different dictionary formats
        g.add_edge(0, 1, {"weight": 2.5})
        g.add_edge(1, 2, {"weight": 1.0})
        g.add_edge(2, 3, {"weight": 3.0})

        res = rustworkx.community.asyn_lpa_communities(g, weight="weight", seed=42)

        # Should complete successfully
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(4)))

    def test_directed_edge_weight_object_format(self):
        """Test asyn_lpa_communities with edge weights stored as objects with mapping protocol in directed graph."""

        class WeightObject:
            def __init__(self, weight):
                self.weight = weight

            def __getitem__(self, key):
                if key == "weight":
                    return self.weight
                raise KeyError(key)

        g = PyDiGraph()
        g.add_nodes_from(range(4))

        # Add edges with weight objects
        g.add_edge(0, 1, WeightObject(2.0))
        g.add_edge(1, 2, WeightObject(1.5))
        g.add_edge(2, 3, WeightObject(3.0))

        res = rustworkx.community.asyn_lpa_communities(g, weight="weight", seed=42)

        # Should complete successfully
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(4)))

    def test_directed_no_weight_attribute(self):
        """Test asyn_lpa_communities when weight attribute doesn't exist in directed graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(4))

        # Add edges without weight attribute
        g.add_edge(0, 1, {"other": 1.0})
        g.add_edge(1, 2, {"other": 1.0})
        g.add_edge(2, 3, {"other": 1.0})

        # Should use default weight of 1.0
        res = rustworkx.community.asyn_lpa_communities(g, weight="weight", seed=42)

        # Should complete successfully
        self.assertGreater(len(res), 0)

        # All nodes should be in some community
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(4)))

    def test_directed_max_iterations(self):
        """Test asyn_lpa_communities with max_iterations parameter on directed graph."""
        g = PyDiGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 3, None)
        g.add_edge(3, 0, None)

        # Should complete within 10 iterations
        res = rustworkx.community.asyn_lpa_communities(g, max_iterations=10, seed=42)
        self.assertGreater(len(res), 0)  # Should have at least one community

    def test_directed_networkx_comparison(self):
        """Test that rustworkx results are reasonable compared to NetworkX for directed graphs."""
        # Note: NetworkX ignores edge directions for directed graphs,
        # but our implementation only considers outgoing edges.
        # So we test that both complete successfully but don't expect exact matches.

        # Create a directed graph
        nx_g = networkx.DiGraph()
        nx_g.add_nodes_from(range(6))
        nx_g.add_edge(0, 1)
        nx_g.add_edge(1, 2)
        nx_g.add_edge(2, 0)
        nx_g.add_edge(3, 4)
        nx_g.add_edge(4, 5)
        nx_g.add_edge(5, 3)
        nx_g.add_edge(2, 3)

        g = PyDiGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 0, None)
        g.add_edge(3, 4, None)
        g.add_edge(4, 5, None)
        g.add_edge(5, 3, None)
        g.add_edge(2, 3, None)

        # Run both algorithms with the same seed
        seed = 42
        nx_communities = list(nx.community.asyn_lpa_communities(nx_g, seed=seed))
        rx_communities = rustworkx.community.asyn_lpa_communities(g, seed=seed)

        # Both should complete successfully
        self.assertGreater(len(nx_communities), 0)
        self.assertGreater(len(rx_communities), 0)

        # All nodes should be in some community in both cases
        nx_all_nodes = set()
        for comm in nx_communities:
            nx_all_nodes.update(comm)
        self.assertEqual(nx_all_nodes, set(range(6)))

        rx_all_nodes = set()
        for comm in rx_communities:
            rx_all_nodes.update(comm)
        self.assertEqual(rx_all_nodes, set(range(6)))

    def test_strongest_lpa_empty_digraph(self):
        g = PyDiGraph()
        res = rustworkx.community.asyn_lpa_communities_strongest(g)
        self.assertEqual(res, [])

    def test_strongest_lpa_single_node(self):
        g = PyDiGraph()
        g.add_node(0)
        res = rustworkx.community.asyn_lpa_communities_strongest(g)
        self.assertEqual(res, [[0]])

    @pytest.mark.skipif(
        not _has_strongest(),
        reason="NetworkX does not provide asyn_lpa_communities_strongest",
    )
    def test_strongest_lpa_matches_reference_weighted(self):
        g = PyDiGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, {"weight": 4.0})
        g.add_edge(0, 2, {"weight": 2.0})
        g.add_edge(1, 3, {"weight": 4.0})
        g.add_edge(2, 3, {"weight": 1.0})

        nx_g = networkx.DiGraph()
        nx_g.add_nodes_from(range(4))
        nx_g.add_edge(0, 1, weight=4.0)
        nx_g.add_edge(0, 2, weight=2.0)
        nx_g.add_edge(1, 3, weight=4.0)
        nx_g.add_edge(2, 3, weight=1.0)

        seed = 11
        rx_res = rustworkx.community.asyn_lpa_communities_strongest(g, weight="weight", seed=seed)
        nx_res = list(
            networkx.community.asyn_lpa_communities_strongest(nx_g, weight="weight", seed=seed)
        )

        self.assertEqual(sorted(map(sorted, rx_res)), sorted(map(sorted, nx_res)))

    @pytest.mark.skipif(
        not _has_strongest(),
        reason="NetworkX does not provide asyn_lpa_communities_strongest",
    )
    def test_strongest_lpa_unweighted_ties(self):
        g = PyDiGraph()
        g.add_nodes_from(range(5))
        g.add_edge(0, 1, None)
        g.add_edge(0, 2, None)
        g.add_edge(0, 3, None)
        g.add_edge(3, 4, None)

        nx_g = networkx.DiGraph()
        nx_g.add_nodes_from(range(5))
        nx_g.add_edge(0, 1)
        nx_g.add_edge(0, 2)
        nx_g.add_edge(0, 3)
        nx_g.add_edge(3, 4)

        seed = 19
        rx_res = rustworkx.community.asyn_lpa_communities_strongest(g, seed=seed)
        nx_res = list(networkx.community.asyn_lpa_communities_strongest(nx_g, seed=seed))

        self.assertEqual(sorted(map(sorted, rx_res)), sorted(map(sorted, nx_res)))
