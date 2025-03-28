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

import os
import unittest

import rustworkx
from rustworkx import PyGraph


class TestCommunity(unittest.TestCase):
    """Tests for community detection algorithms."""

    def test_empty_graph(self):
        """Test community detection on an empty graph."""
        g = PyGraph()
        res = rustworkx.label_propagation_communities(g)
        self.assertEqual(len(res), 0)

    def test_single_node(self):
        """Test community detection on a single-node graph."""
        g = PyGraph()
        g.add_node(0)
        res = rustworkx.label_propagation_communities(g)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], [0])

    def test_disconnected_nodes(self):
        """Test community detection on disconnected nodes."""
        g = PyGraph()
        g.add_nodes_from(range(5))
        res = rustworkx.label_propagation_communities(g)
        self.assertEqual(len(res), 5)
        for i in range(5):
            self.assertIn([i], res)

    def test_simple_communities(self):
        """Test community detection on a graph with two obvious communities."""
        g = PyGraph()
        g.add_nodes_from(range(8))
        # Create two distinct communities
        # Community 1: nodes 0-3 fully connected
        g.add_edge(0, 1, None)
        g.add_edge(0, 2, None)
        g.add_edge(0, 3, None)
        g.add_edge(1, 2, None)
        g.add_edge(1, 3, None)
        g.add_edge(2, 3, None)
        
        # Community 2: nodes 4-7 fully connected
        g.add_edge(4, 5, None)
        g.add_edge(4, 6, None)
        g.add_edge(4, 7, None)
        g.add_edge(5, 6, None)
        g.add_edge(5, 7, None)
        g.add_edge(6, 7, None)
        
        # Add one edge between communities
        g.add_edge(3, 4, None)
        
        res = rustworkx.label_propagation_communities(g)
        self.assertEqual(len(res), 2)
        
        # Sort the communities by size and make sure each is properly formed
        communities = sorted(res, key=len)
        # Convert to sets for easy comparison
        comm_sets = [set(comm) for comm in communities]
        
        # Should have two communities: {0,1,2,3} and {4,5,6,7}
        self.assertIn(set(range(0, 4)), comm_sets)
        self.assertIn(set(range(4, 8)), comm_sets)

    def test_not_implemented_for_directed(self):
        """Test that an error is raised for directed graphs."""
        g = rustworkx.PyDiGraph()
        with self.assertRaises(NotImplementedError):
            rustworkx.label_propagation_communities(g)
            
    def test_karate_club(self):
        """Test community detection on the classic Zachary karate club network."""
        g = PyGraph()
        g.add_nodes_from(range(34))  # 34 members
        
        # Add edges based on the classic Zachary karate club dataset
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
            (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
            (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
            (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
            (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32),
            (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32),
            (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32),
            (23, 33), (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33),
            (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33),
            (31, 32), (31, 33), (32, 33)
        ]
        
        for i, j in edges:
            g.add_edge(i, j, None)
            
        communities = rustworkx.label_propagation_communities(g)
        
        # Check that we find at least 2 communities (the known split)
        self.assertGreaterEqual(len(communities), 2)
        
        # In the real Zachary karate club, the club split into two factions
        # Owner faction and instructor faction. We should check if nodes 0 and 33
        # are in different communities
        
        # Find which communities contain nodes 0 and 33
        node0_comm = None
        node33_comm = None
        
        for i, comm in enumerate(communities):
            if 0 in comm:
                node0_comm = i
            if 33 in comm:
                node33_comm = i
        
        # Check they're in different communities
        self.assertIsNotNone(node0_comm)
        self.assertIsNotNone(node33_comm)
        self.assertNotEqual(node0_comm, node33_comm) 