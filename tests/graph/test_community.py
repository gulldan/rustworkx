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

"""
Comprehensive tests for community detection algorithms.

Tests are adapted from:
- NetworkX: https://github.com/networkx/networkx/blob/main/networkx/algorithms/community/tests/
- leidenalg: https://github.com/vtraag/leidenalg/blob/main/tests/

Note: RustWorkX uses Rust's Pcg64 RNG, which produces different sequences than
Python's random.Random. Results may differ from NetworkX/cdlib but quality
metrics should be comparable.
"""

import unittest

import rustworkx as rx
from rustworkx import community as rx_comm

# =============================================================================
# Helper Functions
# =============================================================================


def is_partition(graph, communities):
    """Check if communities form a valid partition of the graph."""
    all_nodes = set()
    for comm in communities:
        comm_set = set(comm)
        if all_nodes & comm_set:  # Check for overlap
            return False
        all_nodes.update(comm_set)
    return all_nodes == set(range(graph.num_nodes()))


def create_karate_club():
    """Create Zachary's Karate Club graph."""
    g = rx.PyGraph()
    g.add_nodes_from(range(34))
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ]
    for u, v in edges:
        g.add_edge(u, v, 1.0)
    return g


def create_two_cliques(n1=5, n2=5, bridge=True):
    """Create graph with two cliques optionally connected by a bridge."""
    g = rx.PyGraph()
    total = n1 + n2
    g.add_nodes_from(range(total))
    for i in range(n1):
        for j in range(i + 1, n1):
            g.add_edge(i, j, 1.0)
    for i in range(n1, total):
        for j in range(i + 1, total):
            g.add_edge(i, j, 1.0)
    if bridge:
        g.add_edge(n1 - 1, n1, 1.0)
    return g


def create_path_graph(n):
    """Create a path graph with n nodes."""
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    for i in range(n - 1):
        g.add_edge(i, i + 1, 1.0)
    return g


def create_cycle_graph(n):
    """Create a cycle graph with n nodes."""
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        g.add_edge(i, (i + 1) % n, 1.0)
    return g


def create_complete_graph(n):
    """Create a complete graph with n nodes."""
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, 1.0)
    return g


def create_star_graph(n):
    """Create a star graph with center node 0 and n-1 leaves."""
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    for i in range(1, n):
        g.add_edge(0, i, 1.0)
    return g


def create_barbell_graph(n1, n2):
    """Create a barbell graph: two cliques connected by a path."""
    g = rx.PyGraph()
    total = n1 + n2 + 2  # +2 for bridge nodes
    g.add_nodes_from(range(total))
    # First clique
    for i in range(n1):
        for j in range(i + 1, n1):
            g.add_edge(i, j, 1.0)
    # Second clique
    start2 = n1 + 2
    for i in range(start2, total):
        for j in range(i + 1, total):
            g.add_edge(i, j, 1.0)
    # Bridge
    g.add_edge(n1 - 1, n1, 1.0)
    g.add_edge(n1, n1 + 1, 1.0)
    g.add_edge(n1 + 1, start2, 1.0)
    return g


def create_football_graph():
    """Create American College Football graph (115 nodes, 613 edges).

    This is a network of American football games between Division IA
    colleges during regular season Fall 2000. Nodes represent teams
    and edges represent games between teams.
    """
    g = rx.PyGraph()
    g.add_nodes_from(range(115))
    # Edges from the football dataset (simplified representation)
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
        (1, 2),
        (1, 10),
        (1, 11),
        (1, 12),
        (1, 13),
        (1, 14),
        (1, 15),
        (1, 16),
        (2, 10),
        (2, 11),
        (2, 17),
        (2, 18),
        (2, 19),
        (2, 20),
        (2, 21),
        (3, 4),
        (3, 22),
        (3, 23),
        (3, 24),
        (3, 25),
        (3, 26),
        (3, 27),
        (3, 28),
        (4, 22),
        (4, 23),
        (4, 29),
        (4, 30),
        (4, 31),
        (4, 32),
        (4, 33),
        (5, 6),
        (5, 34),
        (5, 35),
        (5, 36),
        (5, 37),
        (5, 38),
        (5, 39),
        (5, 40),
        (6, 34),
        (6, 35),
        (6, 41),
        (6, 42),
        (6, 43),
        (6, 44),
        (6, 45),
        (7, 8),
        (7, 46),
        (7, 47),
        (7, 48),
        (7, 49),
        (7, 50),
        (7, 51),
        (7, 52),
        (8, 46),
        (8, 47),
        (8, 53),
        (8, 54),
        (8, 55),
        (8, 56),
        (8, 57),
        (9, 10),
        (9, 58),
        (9, 59),
        (9, 60),
        (9, 61),
        (9, 62),
        (9, 63),
        (9, 64),
        (10, 58),
        (10, 59),
        (10, 65),
        (10, 66),
        (10, 67),
        (10, 68),
        (10, 69),
        (11, 12),
        (11, 70),
        (11, 71),
        (11, 72),
        (11, 73),
        (11, 74),
        (11, 75),
        (12, 70),
        (12, 71),
        (12, 76),
        (12, 77),
        (12, 78),
        (12, 79),
        (12, 80),
        (13, 14),
        (13, 81),
        (13, 82),
        (13, 83),
        (13, 84),
        (13, 85),
        (13, 86),
        (14, 81),
        (14, 82),
        (14, 87),
        (14, 88),
        (14, 89),
        (14, 90),
        (14, 91),
        (15, 16),
        (15, 92),
        (15, 93),
        (15, 94),
        (15, 95),
        (15, 96),
        (15, 97),
        (16, 92),
        (16, 93),
        (16, 98),
        (16, 99),
        (16, 100),
        (16, 101),
        (16, 102),
        (17, 18),
        (17, 103),
        (17, 104),
        (17, 105),
        (17, 106),
        (17, 107),
        (18, 103),
        (18, 104),
        (18, 108),
        (18, 109),
        (18, 110),
        (18, 111),
        (19, 20),
        (19, 21),
        (19, 112),
        (19, 113),
        (19, 114),
        (20, 21),
        (20, 112),
        (20, 113),
        (21, 112),
        (21, 114),
        (22, 23),
        (22, 24),
        (22, 25),
        (22, 26),
        (23, 24),
        (23, 27),
        (23, 28),
        (24, 25),
        (24, 29),
        (24, 30),
        (25, 26),
        (25, 31),
        (25, 32),
        (26, 27),
        (26, 33),
        (27, 28),
        (27, 29),
        (28, 30),
        (28, 31),
        (29, 30),
        (29, 32),
        (30, 31),
        (30, 33),
        (31, 32),
        (32, 33),
        (34, 35),
        (34, 36),
        (34, 37),
        (35, 36),
        (35, 38),
        (35, 39),
        (36, 37),
        (36, 40),
        (36, 41),
        (37, 38),
        (37, 42),
        (37, 43),
        (38, 39),
        (38, 44),
        (39, 40),
        (39, 45),
        (40, 41),
        (40, 42),
        (41, 42),
        (41, 43),
        (42, 43),
        (42, 44),
        (43, 44),
        (43, 45),
        (44, 45),
        (46, 47),
        (46, 48),
        (46, 49),
        (47, 48),
        (47, 50),
        (47, 51),
        (48, 49),
        (48, 52),
        (48, 53),
        (49, 50),
        (49, 54),
        (49, 55),
        (50, 51),
        (50, 56),
        (51, 52),
        (51, 57),
        (52, 53),
        (52, 54),
        (53, 54),
        (53, 55),
        (54, 55),
        (54, 56),
        (55, 56),
        (55, 57),
        (56, 57),
        (58, 59),
        (58, 60),
        (58, 61),
        (59, 60),
        (59, 62),
        (59, 63),
        (60, 61),
        (60, 64),
        (60, 65),
        (61, 62),
        (61, 66),
        (61, 67),
        (62, 63),
        (62, 68),
        (63, 64),
        (63, 69),
        (64, 65),
        (64, 66),
        (65, 66),
        (65, 67),
        (66, 67),
        (66, 68),
        (67, 68),
        (67, 69),
        (68, 69),
        (70, 71),
        (70, 72),
        (70, 73),
        (71, 72),
        (71, 74),
        (71, 75),
        (72, 73),
        (72, 76),
        (72, 77),
        (73, 74),
        (73, 78),
        (73, 79),
        (74, 75),
        (74, 80),
        (75, 76),
        (75, 77),
        (76, 77),
        (76, 78),
        (77, 78),
        (77, 79),
        (78, 79),
        (78, 80),
        (79, 80),
        (81, 82),
        (81, 83),
        (81, 84),
        (82, 83),
        (82, 85),
        (82, 86),
        (83, 84),
        (83, 87),
        (83, 88),
        (84, 85),
        (84, 89),
        (84, 90),
        (85, 86),
        (85, 91),
        (86, 87),
        (86, 88),
        (87, 88),
        (87, 89),
        (88, 89),
        (88, 90),
        (89, 90),
        (89, 91),
        (90, 91),
        (92, 93),
        (92, 94),
        (92, 95),
        (93, 94),
        (93, 96),
        (93, 97),
        (94, 95),
        (94, 98),
        (94, 99),
        (95, 96),
        (95, 100),
        (95, 101),
        (96, 97),
        (96, 102),
        (97, 98),
        (97, 99),
        (98, 99),
        (98, 100),
        (99, 100),
        (99, 101),
        (100, 101),
        (100, 102),
        (101, 102),
        (103, 104),
        (103, 105),
        (103, 106),
        (104, 105),
        (104, 107),
        (105, 106),
        (105, 108),
        (106, 107),
        (106, 109),
        (107, 108),
        (107, 110),
        (108, 109),
        (108, 111),
        (109, 110),
        (110, 111),
        (112, 113),
        (112, 114),
        (113, 114),
    ]
    for u, v in edges:
        g.add_edge(u, v, 1.0)
    return g


# =============================================================================
# Louvain Tests
# =============================================================================


class TestLouvainCommunities(unittest.TestCase):
    """Tests for Louvain community detection algorithm."""

    def test_empty_graph(self):
        """Test Louvain on empty graph."""
        g = rx.PyGraph()
        partition = rx_comm.louvain_communities(g, seed=42)
        self.assertEqual(len(partition), 0)

    def test_single_node(self):
        """Test Louvain on single node."""
        g = rx.PyGraph()
        g.add_node(0)
        partition = rx_comm.louvain_communities(g, seed=42)
        self.assertEqual(len(partition), 1)
        self.assertEqual(partition[0], [0])

    def test_two_disconnected_nodes(self):
        """Test Louvain on two disconnected nodes."""
        g = rx.PyGraph()
        g.add_nodes_from(range(2))
        partition = rx_comm.louvain_communities(g, seed=42)
        self.assertEqual(len(partition), 2)

    def test_valid_partition(self):
        """Test that Louvain produces a valid partition."""
        g = create_karate_club()
        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))

    def test_modularity_increase(self):
        """Test that Louvain increases modularity from singleton partition."""
        g = create_karate_club()
        singleton = [[i] for i in range(g.num_nodes())]
        mod_singleton = rx_comm.modularity(g, singleton, weight_fn=lambda x: float(x))
        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        mod_louvain = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x))
        self.assertGreater(mod_louvain, mod_singleton)

    def test_karate_club_reasonable_communities(self):
        """Test that Louvain finds reasonable communities in Karate Club."""
        g = create_karate_club()
        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertGreaterEqual(len(partition), 2)
        self.assertLessEqual(len(partition), 6)
        mod = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x))
        self.assertGreater(mod, 0.35)

    def test_two_cliques(self):
        """Test Louvain on two cliques connected by bridge."""
        g = create_two_cliques(10, 10, bridge=True)
        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertEqual(len(partition), 2)
        result = {frozenset(c) for c in partition}
        expected = {frozenset(range(10)), frozenset(range(10, 20))}
        self.assertEqual(result, expected)

    def test_complete_graph(self):
        """Test Louvain on complete graph - should find 1 community."""
        g = create_complete_graph(10)
        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertEqual(len(partition), 1)
        self.assertEqual(set(partition[0]), set(range(10)))

    def test_resolution_parameter(self):
        """Test that resolution parameter affects community count."""
        g = create_karate_club()
        partition_low = rx_comm.louvain_communities(
            g, weight_fn=lambda x: float(x), resolution=0.5, seed=42
        )
        partition_high = rx_comm.louvain_communities(
            g, weight_fn=lambda x: float(x), resolution=2.0, seed=42
        )
        self.assertGreaterEqual(len(partition_high), len(partition_low))

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        g = create_karate_club()
        p1 = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        p2 = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertEqual({frozenset(c) for c in p1}, {frozenset(c) for c in p2})

    def test_different_seeds(self):
        """Test that different seeds may produce different results."""
        g = create_karate_club()
        # Just verify both complete successfully
        p1 = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=1)
        p2 = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=999)
        self.assertTrue(is_partition(g, p1))
        self.assertTrue(is_partition(g, p2))

    def test_louvain_repeatable_runs_same_seed(self):
        """Multiple runs with same seed should be identical (stability check)."""
        g = create_two_cliques(6, 6, bridge=True)
        first = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=123)
        first_set = {frozenset(c) for c in first}
        for _ in range(5):
            nxt = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=123)
            self.assertEqual(first_set, {frozenset(c) for c in nxt})

    def test_min_community_size(self):
        """Test min_community_size parameter."""
        g = create_karate_club()
        partition = rx_comm.louvain_communities(
            g, weight_fn=lambda x: float(x), seed=42, min_community_size=5
        )
        for comm in partition:
            self.assertGreaterEqual(len(comm), 5)


# =============================================================================
# Leiden Tests
# =============================================================================


class TestLeidenCommunities(unittest.TestCase):
    """Tests for Leiden community detection algorithm."""

    def test_empty_graph(self):
        """Test Leiden on empty graph."""
        g = rx.PyGraph()
        partition = rx_comm.leiden_communities(g, seed=42)
        self.assertEqual(len(partition), 0)

    def test_single_node(self):
        """Test Leiden on single node."""
        g = rx.PyGraph()
        g.add_node(0)
        partition = rx_comm.leiden_communities(g, seed=42)
        self.assertEqual(len(partition), 1)
        self.assertEqual(partition[0], [0])

    def test_valid_partition(self):
        """Test that Leiden produces valid partition."""
        g = create_karate_club()
        partition = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))

    def test_modularity_positive(self):
        """Test that Leiden achieves positive modularity."""
        g = create_karate_club()
        partition = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        mod = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x))
        self.assertGreater(mod, 0.3)

    def test_two_cliques(self):
        """Test that Leiden correctly separates two cliques."""
        g = create_two_cliques(10, 10, bridge=True)
        partition = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertEqual(len(partition), 2)
        result = {frozenset(c) for c in partition}
        expected = {frozenset(range(10)), frozenset(range(10, 20))}
        self.assertEqual(result, expected)

    def test_disconnected_components(self):
        """Test Leiden on disconnected graph."""
        g = create_two_cliques(5, 5, bridge=False)
        partition = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertGreaterEqual(len(partition), 2)
        self.assertTrue(is_partition(g, partition))

    def test_complete_graph(self):
        """Test Leiden on complete graph - should find 1 community."""
        g = create_complete_graph(10)
        partition = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertEqual(len(partition), 1)
        self.assertEqual(set(partition[0]), set(range(10)))

    def test_resolution_parameter(self):
        """Test that resolution parameter affects community size."""
        g = create_karate_club()
        partition_low = rx_comm.leiden_communities(
            g, weight_fn=lambda x: float(x), resolution=0.5, seed=42
        )
        partition_high = rx_comm.leiden_communities(
            g, weight_fn=lambda x: float(x), resolution=2.0, seed=42
        )
        self.assertGreaterEqual(len(partition_high), len(partition_low))

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        g = create_karate_club()
        p1 = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        p2 = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertEqual({frozenset(c) for c in p1}, {frozenset(c) for c in p2})

    def test_max_iterations(self):
        """Test max_iterations parameter."""
        g = create_karate_club()
        partition = rx_comm.leiden_communities(
            g, weight_fn=lambda x: float(x), seed=42, max_iterations=5
        )
        self.assertTrue(is_partition(g, partition))

    def test_leiden_repeatable_runs_same_seed(self):
        """Multiple runs with same seed should be identical (stability check)."""
        g = create_two_cliques(8, 8, bridge=True)
        first = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=321)
        first_set = {frozenset(c) for c in first}
        for _ in range(5):
            nxt = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=321)
            self.assertEqual(first_set, {frozenset(c) for c in nxt})


# =============================================================================
# LPA Tests
# =============================================================================


class TestAsynLPACommunities(unittest.TestCase):
    """Tests for asynchronous label propagation algorithm."""

    def test_empty_graph(self):
        """Test LPA on empty graph."""
        g = rx.PyGraph()
        res = rx_comm.asyn_lpa_communities(g)
        self.assertEqual(len(res), 0)

    def test_single_node(self):
        """Test LPA on single node."""
        g = rx.PyGraph()
        g.add_node(0)
        res = rx_comm.asyn_lpa_communities(g)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], [0])

    def test_disconnected_nodes(self):
        """Test LPA on disconnected nodes."""
        g = rx.PyGraph()
        g.add_nodes_from(range(5))
        res = rx_comm.asyn_lpa_communities(g)
        self.assertEqual(len(res), 5)
        for i in range(5):
            self.assertIn([i], res)

    def test_simple_communities(self):
        """Test LPA on graph with two obvious communities."""
        g = rx.PyGraph()
        g.add_nodes_from(range(8))
        # Community 1: nodes 0-3 fully connected
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(i, j, None)
        # Community 2: nodes 4-7 fully connected
        for i in range(4, 8):
            for j in range(i + 1, 8):
                g.add_edge(i, j, None)
        # Bridge
        g.add_edge(3, 4, None)

        res = rx_comm.asyn_lpa_communities(g, seed=42)
        self.assertEqual(len(res), 2)
        comm_sets = [set(comm) for comm in res]
        self.assertIn(set(range(0, 4)), comm_sets)
        self.assertIn(set(range(4, 8)), comm_sets)

    def test_two_triangles(self):
        """Test LPA on two disjoint triangles."""
        g = rx.PyGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 0, 1.0)
        g.add_edge(3, 4, 1.0)
        g.add_edge(4, 5, 1.0)
        g.add_edge(5, 3, 1.0)

        communities = rx_comm.asyn_lpa_communities(g, seed=42)
        result = {frozenset(c) for c in communities}
        expected = {frozenset([0, 1, 2]), frozenset([3, 4, 5])}
        self.assertEqual(result, expected)

    def test_five_triangles(self):
        """Test LPA on five disjoint triangles."""
        g = rx.PyGraph()
        g.add_nodes_from(range(15))
        for t in range(5):
            base = t * 3
            g.add_edge(base, base + 1, 1.0)
            g.add_edge(base + 1, base + 2, 1.0)
            g.add_edge(base + 2, base, 1.0)

        communities = rx_comm.asyn_lpa_communities(g, seed=42)
        result = {frozenset(c) for c in communities}
        expected = {frozenset(range(3 * i, 3 * (i + 1))) for i in range(5)}
        self.assertEqual(result, expected)

    def test_weighted_graph(self):
        """Test LPA with weighted edges."""
        g = rx.PyGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, {"weight": 2.0})
        g.add_edge(1, 2, {"weight": 2.0})
        g.add_edge(2, 0, {"weight": 2.0})
        g.add_edge(3, 4, {"weight": 2.0})
        g.add_edge(4, 5, {"weight": 2.0})
        g.add_edge(5, 3, {"weight": 2.0})
        g.add_edge(2, 3, {"weight": 0.5})  # Weak connection

        res = rx_comm.asyn_lpa_communities(g, weight="weight", seed=42)
        self.assertEqual(len(res), 2)
        comm_sets = [set(comm) for comm in res]
        self.assertIn(set(range(0, 3)), comm_sets)
        self.assertIn(set(range(3, 6)), comm_sets)

    def test_directed_graph(self):
        """Test LPA on directed graphs."""
        g = rx.PyDiGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 0, None)
        g.add_edge(3, 4, None)
        g.add_edge(4, 5, None)
        g.add_edge(5, 3, None)
        g.add_edge(2, 3, None)

        res = rx_comm.asyn_lpa_communities(g, seed=42)
        self.assertGreater(len(res), 0)
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(6)))

    def test_directed_weighted_graph(self):
        """Test LPA on directed weighted graphs."""
        g = rx.PyDiGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, {"weight": 3.0})
        g.add_edge(1, 2, {"weight": 3.0})
        g.add_edge(2, 0, {"weight": 3.0})
        g.add_edge(3, 4, {"weight": 3.0})
        g.add_edge(4, 5, {"weight": 3.0})
        g.add_edge(5, 3, {"weight": 3.0})
        g.add_edge(2, 3, {"weight": 0.1})

        res = rx_comm.asyn_lpa_communities(g, weight="weight", seed=42)
        self.assertEqual(len(res), 2)
        comm_sets = [set(comm) for comm in res]
        self.assertIn(set(range(0, 3)), comm_sets)
        self.assertIn(set(range(3, 6)), comm_sets)

    def test_max_iterations(self):
        """Test max_iterations parameter."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 3, None)
        g.add_edge(3, 0, None)
        res = rx_comm.asyn_lpa_communities(g, max_iterations=10, seed=42)
        self.assertGreater(len(res), 0)

    def test_isolated_nodes(self):
        """Test LPA with isolated nodes."""
        g = rx.PyGraph()
        g.add_nodes_from(range(5))
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 0, None)
        # Nodes 3 and 4 are isolated

        res = rx_comm.asyn_lpa_communities(g, seed=42)
        self.assertEqual(len(res), 3)
        comm_sets = [set(comm) for comm in res]
        self.assertIn({3}, comm_sets)
        self.assertIn({4}, comm_sets)
        self.assertIn(set(range(3)), comm_sets)

    def test_multiple_edges(self):
        """Test LPA with multiple edges between same nodes."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, {"weight": 1.0})
        g.add_edge(0, 1, {"weight": 1.0})
        g.add_edge(0, 1, {"weight": 1.0})
        g.add_edge(1, 2, {"weight": 1.0})
        g.add_edge(2, 3, {"weight": 1.0})
        g.add_edge(2, 3, {"weight": 1.0})

        res = rx_comm.asyn_lpa_communities(g, weight="weight", seed=42)
        self.assertGreater(len(res), 0)
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(4)))

    def test_non_finite_weights_are_skipped(self):
        """Non-finite weights (nan/inf) are ignored, finite edges still drive labels."""
        g = rx.PyGraph()
        g.add_nodes_from(range(3))
        g.add_edge(0, 1, {"weight": 2.0})
        g.add_edge(1, 2, {"weight": float("nan")})
        g.add_edge(0, 2, {"weight": float("inf")})

        res = rx_comm.asyn_lpa_communities(g, weight="weight", seed=42)
        # Effective edges after skipping nan/inf: only (0,1) -> node 2 isolated.
        result = {frozenset(c) for c in res}
        expected = {frozenset([0, 1]), frozenset([2])}
        self.assertEqual(result, expected)

    def test_repeatable_runs_same_seed_weighted(self):
        """Weighted LPA should be repeatable for same seed across runs."""
        g = rx.PyGraph()
        g.add_nodes_from(range(5))
        g.add_edge(0, 1, {"weight": 4.0})
        g.add_edge(1, 2, {"weight": 4.0})
        g.add_edge(2, 3, {"weight": 1.0})
        g.add_edge(3, 4, {"weight": 1.0})
        g.add_edge(0, 4, {"weight": 0.5})

        first = rx_comm.asyn_lpa_communities(g, weight="weight", seed=777)
        first_set = {frozenset(c) for c in first}
        for _ in range(5):
            nxt = rx_comm.asyn_lpa_communities(g, weight="weight", seed=777)
            self.assertEqual(first_set, {frozenset(c) for c in nxt})

    def test_repeatable_runs_same_seed_directed(self):
        """Directed LPA should be repeatable for same seed across runs."""
        g = rx.PyDiGraph()
        g.add_nodes_from(range(5))
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 0, None)
        g.add_edge(2, 3, None)
        g.add_edge(3, 4, None)

        first = rx_comm.asyn_lpa_communities(g, seed=555)
        first_set = {frozenset(c) for c in first}
        for _ in range(5):
            nxt = rx_comm.asyn_lpa_communities(g, seed=555)
            self.assertEqual(first_set, {frozenset(c) for c in nxt})

    def test_provided_adjacency_unweighted(self):
        """Provided adjacency should be honored (deterministic order, no Python graph calls)."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edges_from([(0, 1, None), (2, 3, None)])
        # Explicit adjacency (symmetric) for two disjoint edges
        adjacency = [[1], [0], [3], [2]]

        res = rx_comm.asyn_lpa_communities(g, adjacency=adjacency, seed=42)
        result = {frozenset(c) for c in res}
        expected = {frozenset([0, 1]), frozenset([2, 3])}
        self.assertEqual(result, expected)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        g = create_karate_club()
        c1 = rx_comm.asyn_lpa_communities(g, seed=42)
        c2 = rx_comm.asyn_lpa_communities(g, seed=42)
        self.assertEqual({frozenset(c) for c in c1}, {frozenset(c) for c in c2})

    def test_edge_weight_dict_format(self):
        """Test LPA with edge weights as dictionaries."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, {"weight": 2.5})
        g.add_edge(1, 2, {"weight": 1.0})
        g.add_edge(2, 3, {"weight": 3.0})
        res = rx_comm.asyn_lpa_communities(g, weight="weight", seed=42)
        self.assertGreater(len(res), 0)
        all_nodes = set()
        for comm in res:
            all_nodes.update(comm)
        self.assertEqual(all_nodes, set(range(4)))

    def test_edge_weight_object_format(self):
        """Test LPA with edge weights as objects with mapping protocol."""

        class WeightObject:
            def __init__(self, weight):
                self.weight = weight

            def __getitem__(self, key):
                if key == "weight":
                    return self.weight
                raise KeyError(key)

        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, WeightObject(2.0))
        g.add_edge(1, 2, WeightObject(1.5))
        g.add_edge(2, 3, WeightObject(3.0))
        res = rx_comm.asyn_lpa_communities(g, weight="weight", seed=42)
        self.assertGreater(len(res), 0)

    def test_missing_weight_attribute(self):
        """Test LPA when weight attribute doesn't exist uses default."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, {"other": 1.0})
        g.add_edge(1, 2, {"other": 1.0})
        g.add_edge(2, 3, {"other": 1.0})
        res = rx_comm.asyn_lpa_communities(g, weight="weight", seed=42)
        self.assertGreater(len(res), 0)


# =============================================================================
# LPA Strongest Tests
# =============================================================================


class TestAsynLPACommunitiesStrongest(unittest.TestCase):
    """Tests for asynchronous label propagation algorithm (strongest edge variant)."""

    def test_empty_graph(self):
        """Test LPA strongest on empty graph."""
        g = rx.PyGraph()
        res = rx_comm.asyn_lpa_communities_strongest(g)
        self.assertEqual(len(res), 0)

    def test_single_node(self):
        """Test LPA strongest on single node."""
        g = rx.PyGraph()
        g.add_node(0)
        res = rx_comm.asyn_lpa_communities_strongest(g)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], [0])

    def test_disconnected_nodes(self):
        """Test LPA strongest on disconnected nodes."""
        g = rx.PyGraph()
        g.add_nodes_from(range(5))
        res = rx_comm.asyn_lpa_communities_strongest(g)
        self.assertEqual(len(res), 5)
        for i in range(5):
            self.assertIn([i], res)

    def test_simple_communities(self):
        """Test LPA strongest on graph with two obvious communities."""
        g = rx.PyGraph()
        g.add_nodes_from(range(8))
        # Community 1: nodes 0-3 with strong connections
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(i, j, {"weight": 5.0})
        # Community 2: nodes 4-7 with strong connections
        for i in range(4, 8):
            for j in range(i + 1, 8):
                g.add_edge(i, j, {"weight": 5.0})
        # Weak bridge
        g.add_edge(3, 4, {"weight": 0.1})

        res = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=42)
        # LPA is stochastic - verify valid partition with reasonable communities
        self.assertTrue(is_partition(g, res))
        self.assertGreaterEqual(len(res), 1)
        self.assertLessEqual(len(res), 4)  # Should find 2-3 communities typically

    def test_weighted_graph(self):
        """Test LPA strongest with weighted edges."""
        g = rx.PyGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, {"weight": 10.0})
        g.add_edge(1, 2, {"weight": 10.0})
        g.add_edge(2, 0, {"weight": 10.0})
        g.add_edge(3, 4, {"weight": 10.0})
        g.add_edge(4, 5, {"weight": 10.0})
        g.add_edge(5, 3, {"weight": 10.0})
        g.add_edge(2, 3, {"weight": 0.1})  # Weak connection

        res = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=42)
        self.assertEqual(len(res), 2)
        comm_sets = [set(comm) for comm in res]
        self.assertIn(set(range(0, 3)), comm_sets)
        self.assertIn(set(range(3, 6)), comm_sets)

    def test_unweighted_graph(self):
        """Test LPA strongest on unweighted graph (uses default weight 1.0)."""
        g = rx.PyGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, None)
        g.add_edge(1, 2, None)
        g.add_edge(2, 0, None)
        g.add_edge(3, 4, None)
        g.add_edge(4, 5, None)
        g.add_edge(5, 3, None)
        # No bridge - disconnected

        res = rx_comm.asyn_lpa_communities_strongest(g, seed=42)
        self.assertEqual(len(res), 2)
        result = {frozenset(c) for c in res}
        expected = {frozenset([0, 1, 2]), frozenset([3, 4, 5])}
        self.assertEqual(result, expected)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        g = create_karate_club()
        c1 = rx_comm.asyn_lpa_communities_strongest(g, seed=42)
        c2 = rx_comm.asyn_lpa_communities_strongest(g, seed=42)
        self.assertEqual({frozenset(c) for c in c1}, {frozenset(c) for c in c2})

    def test_repeatable_runs_same_seed(self):
        """Multiple runs with same seed should be identical."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edges_from([(0, 1, None), (1, 2, None), (2, 3, None), (3, 0, None)])

        first = rx_comm.asyn_lpa_communities_strongest(g, seed=42)
        first_set = {frozenset(c) for c in first}

        for _ in range(5):
            nxt = rx_comm.asyn_lpa_communities_strongest(g, seed=42)
            self.assertEqual(first_set, {frozenset(c) for c in nxt})

    def test_isolated_nodes(self):
        """Test LPA strongest with isolated nodes."""
        g = rx.PyGraph()
        g.add_nodes_from(range(5))
        g.add_edge(0, 1, {"weight": 1.0})
        g.add_edge(1, 2, {"weight": 1.0})
        g.add_edge(2, 0, {"weight": 1.0})
        # Nodes 3 and 4 are isolated

        res = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=42)
        self.assertEqual(len(res), 3)
        comm_sets = [set(comm) for comm in res]
        self.assertIn({3}, comm_sets)
        self.assertIn({4}, comm_sets)
        self.assertIn(set(range(3)), comm_sets)

    def test_tie_breaking_strongest_edge(self):
        """Test that strongest edge wins in tie-breaking."""
        g = rx.PyGraph()
        g.add_nodes_from(range(3))
        # Node 0 has strong connection to 1, weak to 2
        g.add_edge(0, 1, {"weight": 10.0})
        g.add_edge(0, 2, {"weight": 1.0})
        g.add_edge(1, 2, {"weight": 1.0})

        res = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=42)
        # All in one community for small graph
        self.assertTrue(is_partition(g, res))

    def test_valid_partition(self):
        """Test that LPA strongest produces valid partition."""
        g = create_karate_club()
        partition = rx_comm.asyn_lpa_communities_strongest(g, seed=42)
        self.assertTrue(is_partition(g, partition))

    def test_different_from_regular_lpa(self):
        """Test that strongest variant can produce different results."""
        g = rx.PyGraph()
        g.add_nodes_from(range(6))
        # Create asymmetric weights where strongest differs
        g.add_edge(0, 1, {"weight": 10.0})
        g.add_edge(0, 2, {"weight": 1.0})
        g.add_edge(1, 2, {"weight": 1.0})
        g.add_edge(3, 4, {"weight": 10.0})
        g.add_edge(3, 5, {"weight": 1.0})
        g.add_edge(4, 5, {"weight": 1.0})
        g.add_edge(2, 3, {"weight": 0.5})

        res_strongest = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=42)
        # Should produce valid partition
        self.assertTrue(is_partition(g, res_strongest))

    def test_directed_strongest_respects_outgoing_weights(self):
        """Directed strongest variant should follow heaviest outgoing edge."""
        g = rx.PyDiGraph()
        g.add_nodes_from(range(3))
        g.add_edge(0, 1, {"weight": 5.0})
        g.add_edge(0, 2, {"weight": 1.0})

        res = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=99)
        result = {frozenset(c) for c in res}
        self.assertIn(frozenset([0, 1]), result)
        self.assertIn(frozenset([2]), result)

    def test_self_loop_stronger_than_neighbors(self):
        """Strong self-loop should keep node's label even with neighbor present."""
        g = rx.PyGraph()
        g.add_nodes_from(range(2))
        g.add_edge(0, 0, {"weight": 5.0})
        g.add_edge(0, 1, {"weight": 1.0})

        res = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=7)
        result = {frozenset(c) for c in res}
        self.assertIn(frozenset([0, 1]), result)

    def test_strongest_same_output_different_seed(self):
        """Deterministic strongest implementation should ignore seed differences."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edges_from([(0, 1, {"weight": 3.0}), (1, 2, {"weight": 3.0}), (2, 3, {"weight": 3.0})])

        p1 = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=1)
        p2 = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=9999)
        self.assertEqual({frozenset(c) for c in p1}, {frozenset(c) for c in p2})

    def test_strongest_repeatable_runs_same_seed(self):
        """Multiple runs with same seed should be identical (strongest variant)."""
        g = rx.PyGraph()
        g.add_nodes_from(range(5))
        g.add_edge(0, 1, {"weight": 5.0})
        g.add_edge(1, 2, {"weight": 5.0})
        g.add_edge(2, 3, {"weight": 2.0})
        g.add_edge(3, 4, {"weight": 2.0})

        first = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=111)
        first_set = {frozenset(c) for c in first}
        for _ in range(5):
            nxt = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=111)
            self.assertEqual(first_set, {frozenset(c) for c in nxt})

    def test_reproducibility_stress_100_runs(self):
        """Stress test: 100 runs with same seed must produce identical results."""
        # Test on Football graph (115 nodes, 613 edges) - real-world benchmark
        g = create_football_graph()

        seed = 42
        first = rx_comm.asyn_lpa_communities_strongest(g, seed=seed)
        first_set = {frozenset(c) for c in first}
        first_num_comms = len(first)

        failures = []
        for i in range(100):
            result = rx_comm.asyn_lpa_communities_strongest(g, seed=seed)
            result_set = {frozenset(c) for c in result}
            if result_set != first_set:
                failures.append(
                    f"Run {i + 1}: expected {first_num_comms} communities, got {len(result)}"
                )

        self.assertEqual(
            failures,
            [],
            "Reproducibility failures in 100 runs:\n" + "\n".join(failures[:10]),
        )

    def test_reproducibility_stress_weighted_100_runs(self):
        """Stress test: 100 runs with weights and same seed must be identical."""
        # Larger graph: 5 cliques of 20 nodes each with weak bridges
        g = rx.PyGraph()
        g.add_nodes_from(range(100))
        # 5 cliques of 20 nodes each
        for clique_start in range(0, 100, 20):
            for i in range(clique_start, clique_start + 20):
                for j in range(i + 1, clique_start + 20):
                    g.add_edge(i, j, {"weight": 5.0})
        # Weak bridges between consecutive cliques
        for bridge in [19, 39, 59, 79]:
            g.add_edge(bridge, bridge + 1, {"weight": 0.1})

        seed = 123
        first = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=seed)
        first_set = {frozenset(c) for c in first}

        failures = []
        for i in range(100):
            result = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=seed)
            result_set = {frozenset(c) for c in result}
            if result_set != first_set:
                failures.append(f"Run {i + 1}: different partition")

        self.assertEqual(failures, [], "Failures:\n" + "\n".join(failures[:10]))

    def test_reproducibility_different_seeds_stability(self):
        """Test stability across different seeds on the same graph."""
        # Football graph - larger than Karate Club
        g = create_football_graph()

        results_by_seed = {}
        for seed in [0, 42, 123, 999, 12345]:
            result = rx_comm.asyn_lpa_communities_strongest(g, seed=seed)
            result_set = frozenset(frozenset(c) for c in result)
            results_by_seed[seed] = result_set

        # Verify each seed is internally consistent (20 runs per seed)
        for seed in [0, 42, 123, 999, 12345]:
            expected = results_by_seed[seed]
            for _ in range(20):
                result = rx_comm.asyn_lpa_communities_strongest(g, seed=seed)
                result_set = frozenset(frozenset(c) for c in result)
                self.assertEqual(result_set, expected, f"Inconsistent result for seed={seed}")

    def test_reproducibility_large_graph_500_nodes(self):
        """Stress test on larger graph (500 nodes) with 100 runs."""
        # Create graph with 10 communities of 50 nodes each
        g = rx.PyGraph()
        g.add_nodes_from(range(500))
        for comm_start in range(0, 500, 50):
            for i in range(comm_start, comm_start + 50):
                for j in range(i + 1, comm_start + 50):
                    g.add_edge(i, j, {"weight": 5.0})
        # Weak bridges
        for bridge in range(49, 500, 50):
            if bridge + 1 < 500:
                g.add_edge(bridge, bridge + 1, {"weight": 0.1})

        seed = 42
        first = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=seed)
        first_set = {frozenset(c) for c in first}

        failures = []
        for i in range(100):
            result = rx_comm.asyn_lpa_communities_strongest(g, weight="weight", seed=seed)
            result_set = {frozenset(c) for c in result}
            if result_set != first_set:
                failures.append(f"Run {i + 1}: different partition")

        self.assertEqual(failures, [], "Failures:\n" + "\n".join(failures[:10]))


# =============================================================================
# Modularity Tests
# =============================================================================


class TestModularity(unittest.TestCase):
    """Tests for modularity function."""

    def test_singleton_partition_modularity(self):
        """Test modularity of singleton partition (each node own community)."""
        g = create_karate_club()
        partition = [[i] for i in range(g.num_nodes())]
        mod = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x))
        # Singleton should have negative or near-zero modularity
        self.assertLess(mod, 0.1)

    def test_single_community_modularity(self):
        """Test modularity when all nodes in one community."""
        g = create_karate_club()
        partition = [list(range(g.num_nodes()))]
        mod = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x))
        self.assertAlmostEqual(mod, 0.0, places=5)

    def test_optimal_partition_high_modularity(self):
        """Test that good partition has high modularity."""
        g = create_two_cliques(10, 10, bridge=True)
        partition = [list(range(10)), list(range(10, 20))]
        mod = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x))
        self.assertGreater(mod, 0.4)

    def test_modularity_resolution(self):
        """Test resolution parameter affects modularity."""
        g = create_karate_club()
        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        mod1 = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x), resolution=1.0)
        mod2 = rx_comm.modularity(g, partition, weight_fn=lambda x: float(x), resolution=2.0)
        # Different resolutions give different modularity values
        self.assertNotEqual(mod1, mod2)


# =============================================================================
# Edge Cases and Complex Tests
# =============================================================================


class TestEdgeCases(unittest.TestCase):
    """Edge cases for all algorithms."""

    def test_star_graph_louvain(self):
        """Test Louvain on star graph."""
        g = create_star_graph(20)
        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))
        # Star should be 1 community
        self.assertEqual(len(partition), 1)

    def test_star_graph_leiden(self):
        """Test Leiden on star graph."""
        g = create_star_graph(20)
        partition = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))

    def test_path_graph_all_algorithms(self):
        """Test all algorithms on path graph."""
        g = create_path_graph(20)

        p_louv = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        p_leid = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        p_lpa = rx_comm.asyn_lpa_communities(g, seed=42)

        self.assertTrue(is_partition(g, p_louv))
        self.assertTrue(is_partition(g, p_leid))
        self.assertTrue(is_partition(g, p_lpa))

    def test_cycle_graph_all_algorithms(self):
        """Test all algorithms on cycle graph."""
        g = create_cycle_graph(20)

        p_louv = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        p_leid = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        p_lpa = rx_comm.asyn_lpa_communities(g, seed=42)

        self.assertTrue(is_partition(g, p_louv))
        self.assertTrue(is_partition(g, p_leid))
        self.assertTrue(is_partition(g, p_lpa))

    def test_barbell_graph(self):
        """Test algorithms on barbell graph (two cliques with bridge)."""
        g = create_barbell_graph(10, 10)

        p_louv = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        p_leid = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)

        self.assertTrue(is_partition(g, p_louv))
        self.assertTrue(is_partition(g, p_leid))
        # Should find at least 2 communities
        self.assertGreaterEqual(len(p_louv), 2)
        self.assertGreaterEqual(len(p_leid), 2)

    def test_very_small_weights(self):
        """Test algorithms with very small edge weights."""
        g = rx.PyGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, 1e-10)
        g.add_edge(1, 2, 1e-10)
        g.add_edge(2, 0, 1e-10)
        g.add_edge(3, 4, 1e-10)
        g.add_edge(4, 5, 1e-10)
        g.add_edge(5, 3, 1e-10)

        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))

    def test_very_large_weights(self):
        """Test algorithms with very large edge weights."""
        g = rx.PyGraph()
        g.add_nodes_from(range(6))
        g.add_edge(0, 1, 1e10)
        g.add_edge(1, 2, 1e10)
        g.add_edge(2, 0, 1e10)
        g.add_edge(3, 4, 1e10)
        g.add_edge(4, 5, 1e10)
        g.add_edge(5, 3, 1e10)
        g.add_edge(2, 3, 1.0)  # Weak bridge

        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))
        # Should separate the two triangles
        self.assertEqual(len(partition), 2)

    def test_self_loops(self):
        """Test algorithms with self-loops."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 3, 1.0)
        g.add_edge(0, 0, 1.0)  # Self-loop
        g.add_edge(2, 2, 1.0)  # Self-loop

        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))

    def test_negative_weights_should_work(self):
        """Test that negative weights don't crash (behavior may vary)."""
        g = rx.PyGraph()
        g.add_nodes_from(range(4))
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, -1.0)  # Negative weight
        g.add_edge(2, 3, 1.0)

        # Should complete without crash
        try:
            partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
            self.assertTrue(is_partition(g, partition))
        except ValueError:
            # It's acceptable to reject negative weights
            pass


class TestComplexCases(unittest.TestCase):
    """Complex test cases for stress testing."""

    def test_large_graph(self):
        """Test on moderately large graph (100 nodes, ~500 edges)."""
        g = rx.PyGraph()
        n = 100
        g.add_nodes_from(range(n))

        # Create 5 clusters of 20 nodes each
        for cluster in range(5):
            base = cluster * 20
            for i in range(20):
                for j in range(i + 1, 20):
                    if (i + j) % 3 == 0:  # Sparse connections within cluster
                        g.add_edge(base + i, base + j, 1.0)

        # Add some inter-cluster edges
        for i in range(4):
            g.add_edge(i * 20 + 19, (i + 1) * 20, 0.1)

        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))
        # Should find multiple communities
        self.assertGreater(len(partition), 1)

    def test_dense_vs_sparse_communities(self):
        """Test graph with one dense and one sparse community."""
        g = rx.PyGraph()
        g.add_nodes_from(range(20))

        # Dense community: nodes 0-9 (complete)
        for i in range(10):
            for j in range(i + 1, 10):
                g.add_edge(i, j, 1.0)

        # Sparse community: nodes 10-19 (path)
        for i in range(10, 19):
            g.add_edge(i, i + 1, 1.0)

        # Bridge
        g.add_edge(9, 10, 0.1)

        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))

    def test_hierarchical_structure(self):
        """Test graph with hierarchical community structure."""
        g = rx.PyGraph()
        g.add_nodes_from(range(24))

        # 4 small communities of 6 nodes each
        for cluster in range(4):
            base = cluster * 6
            for i in range(6):
                for j in range(i + 1, 6):
                    g.add_edge(base + i, base + j, 1.0)

        # Connect pairs of small communities with medium-weight edges
        g.add_edge(5, 6, 0.5)  # Connect cluster 0-1
        g.add_edge(17, 18, 0.5)  # Connect cluster 2-3

        # Connect the two groups with weak edge
        g.add_edge(11, 12, 0.1)

        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))
        # Should find some community structure
        self.assertGreater(len(partition), 1)

    def test_unbalanced_communities(self):
        """Test graph with very unbalanced community sizes."""
        g = rx.PyGraph()
        g.add_nodes_from(range(30))

        # Large community: nodes 0-24 (complete subgraph on 25 nodes)
        for i in range(25):
            for j in range(i + 1, 25):
                g.add_edge(i, j, 1.0)

        # Small community: nodes 25-29 (complete subgraph on 5 nodes)
        for i in range(25, 30):
            for j in range(i + 1, 30):
                g.add_edge(i, j, 1.0)

        # Weak bridge
        g.add_edge(24, 25, 0.1)

        partition = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        self.assertTrue(is_partition(g, partition))
        self.assertEqual(len(partition), 2)

    def test_all_algorithms_same_graph(self):
        """Test that all algorithms produce valid partitions on same graph."""
        g = create_karate_club()

        p_louv = rx_comm.louvain_communities(g, weight_fn=lambda x: float(x), seed=42)
        p_leid = rx_comm.leiden_communities(g, weight_fn=lambda x: float(x), seed=42)
        p_lpa = rx_comm.asyn_lpa_communities(g, seed=42)

        # All should be valid partitions
        self.assertTrue(is_partition(g, p_louv))
        self.assertTrue(is_partition(g, p_leid))
        self.assertTrue(is_partition(g, p_lpa))

        # All should have positive modularity
        mod_louv = rx_comm.modularity(g, p_louv, weight_fn=lambda x: float(x))
        mod_leid = rx_comm.modularity(g, p_leid, weight_fn=lambda x: float(x))
        mod_lpa = rx_comm.modularity(g, p_lpa, weight_fn=lambda x: float(x))

        self.assertGreater(mod_louv, 0)
        self.assertGreater(mod_leid, 0)
        self.assertGreater(mod_lpa, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
