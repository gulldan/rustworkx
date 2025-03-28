# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs

import sys
import numpy as np
import numpy.typing as npt

from typing import Generic, Any, Callable, overload
from collections.abc import Iterator, Sequence

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

# Re-Exports of rust native functions in rustworkx.rustworkx
# To workaround limitations in mypy around re-exporting objects from the inner
# rustworkx module we need to explicitly re-export every inner function from
# rustworkx.rustworkx (the root rust module) in the form:
# `from .rustworkx import foo as foo` so that mypy will treat `rustworkx.foo`
# as a valid path.
from . import visit as visit

from .rustworkx import DAGHasCycle as DAGHasCycle
from .rustworkx import DAGWouldCycle as DAGWouldCycle
from .rustworkx import InvalidNode as InvalidNode
from .rustworkx import NoEdgeBetweenNodes as NoEdgeBetweenNodes
from .rustworkx import NoPathFound as NoPathFound
from .rustworkx import NoSuitableNeighbors as NoSuitableNeighbors
from .rustworkx import NullGraph as NullGraph
from .rustworkx import NegativeCycle as NegativeCycle
from .rustworkx import JSONSerializationError as JSONSerializationError
from .rustworkx import JSONDeserializationError as JSONDeserializationError
from .rustworkx import FailedToConverge as FailedToConverge
from .rustworkx import InvalidMapping as InvalidMapping
from .rustworkx import GraphNotBipartite as GraphNotBipartite
from .rustworkx import ColoringStrategy as ColoringStrategy

from .rustworkx import digraph_maximum_bisimulation as digraph_maximum_bisimulation
from .rustworkx import digraph_cartesian_product as digraph_cartesian_product
from .rustworkx import graph_cartesian_product as graph_cartesian_product
from .rustworkx import digraph_eigenvector_centrality as digraph_eigenvector_centrality
from .rustworkx import graph_eigenvector_centrality as graph_eigenvector_centrality
from .rustworkx import digraph_betweenness_centrality as digraph_betweenness_centrality
from .rustworkx import graph_betweenness_centrality as graph_betweenness_centrality
from .rustworkx import digraph_edge_betweenness_centrality as digraph_edge_betweenness_centrality
from .rustworkx import graph_edge_betweenness_centrality as graph_edge_betweenness_centrality
from .rustworkx import digraph_closeness_centrality as digraph_closeness_centrality
from .rustworkx import graph_closeness_centrality as graph_closeness_centrality
from .rustworkx import (
    digraph_newman_weighted_closeness_centrality as digraph_newman_weighted_closeness_centrality,
)
from .rustworkx import (
    graph_newman_weighted_closeness_centrality as graph_newman_weighted_closeness_centrality,
)
from .rustworkx import digraph_katz_centrality as digraph_katz_centrality
from .rustworkx import graph_katz_centrality as graph_katz_centrality
from .rustworkx import digraph_degree_centrality as digraph_degree_centrality
from .rustworkx import graph_degree_centrality as graph_degree_centrality
from .rustworkx import in_degree_centrality as in_degree_centrality
from .rustworkx import out_degree_centrality as out_degree_centrality
from .rustworkx import graph_greedy_color as graph_greedy_color
from .rustworkx import graph_greedy_edge_color as graph_greedy_edge_color
from .rustworkx import graph_is_bipartite as graph_is_bipartite
from .rustworkx import connected_subgraphs as connected_subgraphs
from .rustworkx import digraph_is_bipartite as digraph_is_bipartite
from .rustworkx import graph_two_color as graph_two_color
from .rustworkx import digraph_two_color as digraph_two_color
from .rustworkx import graph_misra_gries_edge_color as graph_misra_gries_edge_color
from .rustworkx import graph_bipartite_edge_color as graph_bipartite_edge_color
from .rustworkx import label_propagation_communities as label_propagation_communities
from .rustworkx import connected_components as connected_components
from .rustworkx import is_connected as is_connected
from .rustworkx import is_strongly_connected as is_strongly_connected
from .rustworkx import is_weakly_connected as is_weakly_connected
from .rustworkx import is_semi_connected as is_semi_connected
from .rustworkx import number_connected_components as number_connected_components
from .rustworkx import number_strongly_connected_components as number_strongly_connected_components
from .rustworkx import number_weakly_connected_components as number_weakly_connected_components
from .rustworkx import node_connected_component as node_connected_component
from .rustworkx import strongly_connected_components as strongly_connected_components
from .rustworkx import weakly_connected_components as weakly_connected_components
from .rustworkx import digraph_adjacency_matrix as digraph_adjacency_matrix
from .rustworkx import graph_adjacency_matrix as graph_adjacency_matrix
from .rustworkx import cycle_basis as cycle_basis
from .rustworkx import articulation_points as articulation_points
from .rustworkx import bridges as bridges
from .rustworkx import biconnected_components as biconnected_components
from .rustworkx import chain_decomposition as chain_decomposition
from .rustworkx import digraph_find_cycle as digraph_find_cycle
from .rustworkx import digraph_complement as digraph_complement
from .rustworkx import graph_complement as graph_complement
from .rustworkx import local_complement as local_complement
from .rustworkx import digraph_all_simple_paths as digraph_all_simple_paths
from .rustworkx import graph_all_simple_paths as graph_all_simple_paths
from .rustworkx import digraph_all_pairs_all_simple_paths as digraph_all_pairs_all_simple_paths
from .rustworkx import graph_all_pairs_all_simple_paths as graph_all_pairs_all_simple_paths
from .rustworkx import digraph_longest_simple_path as digraph_longest_simple_path
from .rustworkx import graph_longest_simple_path as graph_longest_simple_path
from .rustworkx import digraph_core_number as digraph_core_number
from .rustworkx import graph_core_number as graph_core_number
from .rustworkx import stoer_wagner_min_cut as stoer_wagner_min_cut
from .rustworkx import simple_cycles as simple_cycles
from .rustworkx import digraph_isolates as digraph_isolates
from .rustworkx import graph_isolates as graph_isolates
from .rustworkx import collect_runs as collect_runs
from .rustworkx import collect_bicolor_runs as collect_bicolor_runs
from .rustworkx import dag_longest_path as dag_longest_path
from .rustworkx import dag_longest_path_length as dag_longest_path_length
from .rustworkx import dag_weighted_longest_path as dag_weighted_longest_path
from .rustworkx import dag_weighted_longest_path_length as dag_weighted_longest_path_length
from .rustworkx import is_directed_acyclic_graph as is_directed_acyclic_graph
from .rustworkx import topological_sort as topological_sort
from .rustworkx import topological_generations as topological_generations
from .rustworkx import lexicographical_topological_sort as lexicographical_topological_sort
from .rustworkx import transitive_reduction as transitive_reduction
from .rustworkx import layers as layers
from .rustworkx import TopologicalSorter as TopologicalSorter
from .rustworkx import digraph_is_isomorphic as digraph_is_isomorphic
from .rustworkx import graph_is_isomorphic as graph_is_isomorphic
from .rustworkx import digraph_is_subgraph_isomorphic as digraph_is_subgraph_isomorphic
from .rustworkx import graph_is_subgraph_isomorphic as graph_is_subgraph_isomorphic
from .rustworkx import digraph_vf2_mapping as digraph_vf2_mapping
from .rustworkx import graph_vf2_mapping as graph_vf2_mapping
from .rustworkx import digraph_bipartite_layout as digraph_bipartite_layout
from .rustworkx import graph_bipartite_layout as graph_bipartite_layout
from .rustworkx import digraph_circular_layout as digraph_circular_layout
from .rustworkx import graph_circular_layout as graph_circular_layout
from .rustworkx import digraph_random_layout as digraph_random_layout
from .rustworkx import graph_random_layout as graph_random_layout
from .rustworkx import digraph_shell_layout as digraph_shell_layout
from .rustworkx import graph_shell_layout as graph_shell_layout
from .rustworkx import digraph_spiral_layout as digraph_spiral_layout
from .rustworkx import graph_spiral_layout as graph_spiral_layout
from .rustworkx import digraph_spring_layout as digraph_spring_layout
from .rustworkx import graph_spring_layout as graph_spring_layout
from .rustworkx import graph_line_graph as graph_line_graph
from .rustworkx import hits as hits
from .rustworkx import pagerank as pagerank
from .rustworkx import max_weight_matching as max_weight_matching
from .rustworkx import is_matching as is_matching
from .rustworkx import is_maximal_matching as is_maximal_matching
from .rustworkx import is_planar as is_planar
from .rustworkx import directed_gnm_random_graph as directed_gnm_random_graph
from .rustworkx import undirected_gnm_random_graph as undirected_gnm_random_graph
from .rustworkx import directed_gnp_random_graph as directed_gnp_random_graph
from .rustworkx import undirected_gnp_random_graph as undirected_gnp_random_graph
from .rustworkx import directed_sbm_random_graph as directed_sbm_random_graph
from .rustworkx import undirected_sbm_random_graph as undirected_sbm_random_graph
from .rustworkx import random_geometric_graph as random_geometric_graph
from .rustworkx import hyperbolic_random_graph as hyperbolic_random_graph
from .rustworkx import barabasi_albert_graph as barabasi_albert_graph
from .rustworkx import directed_barabasi_albert_graph as directed_barabasi_albert_graph
from .rustworkx import undirected_random_bipartite_graph as undirected_random_bipartite_graph
from .rustworkx import directed_random_bipartite_graph as directed_random_bipartite_graph
from .rustworkx import read_graphml as read_graphml
from .rustworkx import digraph_node_link_json as digraph_node_link_json
from .rustworkx import graph_node_link_json as graph_node_link_json
from .rustworkx import from_node_link_json_file as from_node_link_json_file
from .rustworkx import parse_node_link_json as parse_node_link_json
from .rustworkx import digraph_bellman_ford_shortest_paths as digraph_bellman_ford_shortest_paths
from .rustworkx import graph_bellman_ford_shortest_paths as graph_bellman_ford_shortest_paths
from .rustworkx import (
    digraph_bellman_ford_shortest_path_lengths as digraph_bellman_ford_shortest_path_lengths,
)
from .rustworkx import (
    graph_bellman_ford_shortest_path_lengths as graph_bellman_ford_shortest_path_lengths,
)
from .rustworkx import digraph_dijkstra_shortest_paths as digraph_dijkstra_shortest_paths
from .rustworkx import graph_dijkstra_shortest_paths as graph_dijkstra_shortest_paths
from .rustworkx import (
    digraph_dijkstra_shortest_path_lengths as digraph_dijkstra_shortest_path_lengths,
)
from .rustworkx import graph_dijkstra_shortest_path_lengths as graph_dijkstra_shortest_path_lengths
from .rustworkx import (
    digraph_all_pairs_bellman_ford_path_lengths as digraph_all_pairs_bellman_ford_path_lengths,
)
from .rustworkx import (
    graph_all_pairs_bellman_ford_path_lengths as graph_all_pairs_bellman_ford_path_lengths,
)
from .rustworkx import (
    digraph_all_pairs_bellman_ford_shortest_paths as digraph_all_pairs_bellman_ford_shortest_paths,
)
from .rustworkx import (
    graph_all_pairs_bellman_ford_shortest_paths as graph_all_pairs_bellman_ford_shortest_paths,
)
from .rustworkx import (
    digraph_all_pairs_dijkstra_path_lengths as digraph_all_pairs_dijkstra_path_lengths,
)
from .rustworkx import (
    graph_all_pairs_dijkstra_path_lengths as graph_all_pairs_dijkstra_path_lengths,
)
from .rustworkx import (
    digraph_all_pairs_dijkstra_shortest_paths as digraph_all_pairs_dijkstra_shortest_paths,
)
from .rustworkx import (
    graph_all_pairs_dijkstra_shortest_paths as graph_all_pairs_dijkstra_shortest_paths,
)
from .rustworkx import digraph_astar_shortest_path as digraph_astar_shortest_path
from .rustworkx import graph_astar_shortest_path as graph_astar_shortest_path
from .rustworkx import digraph_k_shortest_path_lengths as digraph_k_shortest_path_lengths
from .rustworkx import graph_k_shortest_path_lengths as graph_k_shortest_path_lengths
from .rustworkx import digraph_has_path as digraph_has_path
from .rustworkx import graph_has_path as graph_has_path
from .rustworkx import (
    digraph_num_shortest_paths_unweighted as digraph_num_shortest_paths_unweighted,
)
from .rustworkx import graph_num_shortest_paths_unweighted as graph_num_shortest_paths_unweighted
from .rustworkx import (
    digraph_unweighted_average_shortest_path_length as digraph_unweighted_average_shortest_path_length,
)
from .rustworkx import (
    graph_unweighted_average_shortest_path_length as graph_unweighted_average_shortest_path_length,
)
from .rustworkx import digraph_distance_matrix as digraph_distance_matrix
from .rustworkx import graph_distance_matrix as graph_distance_matrix
from .rustworkx import digraph_floyd_warshall as digraph_floyd_warshall
from .rustworkx import graph_floyd_warshall as graph_floyd_warshall
from .rustworkx import digraph_floyd_warshall_numpy as digraph_floyd_warshall_numpy
from .rustworkx import graph_floyd_warshall_numpy as graph_floyd_warshall_numpy
from .rustworkx import (
    digraph_floyd_warshall_successor_and_distance as digraph_floyd_warshall_successor_and_distance,
)
from .rustworkx import (
    graph_floyd_warshall_successor_and_distance as graph_floyd_warshall_successor_and_distance,
)
from .rustworkx import find_negative_cycle as find_negative_cycle
from .rustworkx import negative_edge_cycle as negative_edge_cycle
from .rustworkx import digraph_all_shortest_paths as digraph_all_shortest_paths
from .rustworkx import graph_all_shortest_paths as graph_all_shortest_paths
from .rustworkx import digraph_tensor_product as digraph_tensor_product
from .rustworkx import graph_tensor_product as graph_tensor_product
from .rustworkx import graph_token_swapper as graph_token_swapper
from .rustworkx import digraph_transitivity as digraph_transitivity
from .rustworkx import graph_transitivity as graph_transitivity
from .rustworkx import digraph_bfs_search as digraph_bfs_search
from .rustworkx import graph_bfs_search as graph_bfs_search
from .rustworkx import digraph_dfs_search as digraph_dfs_search
from .rustworkx import graph_dfs_search as graph_dfs_search
from .rustworkx import digraph_dijkstra_search as digraph_dijkstra_search
from .rustworkx import graph_dijkstra_search as graph_dijkstra_search
from .rustworkx import digraph_dfs_edges as digraph_dfs_edges
from .rustworkx import graph_dfs_edges as graph_dfs_edges
from .rustworkx import ancestors as ancestors
from .rustworkx import bfs_predecessors as bfs_predecessors
from .rustworkx import bfs_successors as bfs_successors
from .rustworkx import descendants as descendants
from .rustworkx import minimum_spanning_edges as minimum_spanning_edges
from .rustworkx import minimum_spanning_tree as minimum_spanning_tree
from .rustworkx import steiner_tree as steiner_tree
from .rustworkx import metric_closure as metric_closure
from .rustworkx import digraph_union as digraph_union
from .rustworkx import graph_union as graph_union
from .rustworkx import immediate_dominators as immediate_dominators
from .rustworkx import dominance_frontiers as dominance_frontiers
from .rustworkx import NodeIndices as NodeIndices
from .rustworkx import PathLengthMapping as PathLengthMapping
from .rustworkx import PathMapping as PathMapping
from .rustworkx import AllPairsPathLengthMapping as AllPairsPathLengthMapping
from .rustworkx import AllPairsPathMapping as AllPairsPathMapping
from .rustworkx import BFSSuccessors as BFSSuccessors
from .rustworkx import BFSPredecessors as BFSPredecessors
from .rustworkx import EdgeIndexMap as EdgeIndexMap
from .rustworkx import EdgeIndices as EdgeIndices
from .rustworkx import Chains as Chains
from .rustworkx import IndexPartitionBlock as IndexPartitionBlock
from .rustworkx import RelationalCoarsestPartition as RelationalCoarsestPartition
from .rustworkx import EdgeList as EdgeList
from .rustworkx import NodeMap as NodeMap
from .rustworkx import NodesCountMapping as NodesCountMapping
from .rustworkx import Pos2DMapping as Pos2DMapping
from .rustworkx import WeightedEdgeList as WeightedEdgeList
from .rustworkx import CentralityMapping as CentralityMapping
from .rustworkx import EdgeCentralityMapping as EdgeCentralityMapping
from .rustworkx import BiconnectedComponents as BiconnectedComponents
from .rustworkx import ProductNodeMap as ProductNodeMap
from .rustworkx import MultiplePathMapping as MultiplePathMapping
from .rustworkx import AllPairsMultiplePathMapping as AllPairsMultiplePathMapping
from .rustworkx import PyGraph as PyGraph
from .rustworkx import PyDiGraph as PyDiGraph
from .rustworkx import (
    PyDAG, PyDiGraph, PyGraph, OutDegreeView, InDegreeView, DegreeView,
    DijkstraError, CentralityMapping, EdgeCentralityMapping, NetworkXKeyError,
    DisjointSet, ProductNodeMap,

    # Graphs and errors
    InvalidNode, DAGWouldCycle, DAGHasCycle, NoSuitableNeighbors, NoEdgeBetweenNodes,
    NoPathFound, InvalidMapping, NullGraph, NegativeCycle, JSONSerializationError,
    JSONDeserializationError, FailedToConverge, GraphNotBipartite,

    # Connectivity
    number_connected_components, connected_components, is_connected, node_connected_component,
    number_weakly_connected_components, weakly_connected_components, is_weakly_connected,
    number_strongly_connected_components, strongly_connected_components, is_strongly_connected,
    is_semi_connected,

    # Biconnected
    biconnected_components, articulation_points, is_biconnected, biconnected_component,

    # Traversal algorithms
    bfs_successors, bfs_predecessors, graph_dijkstra_search, digraph_dijkstra_search,
    graph_bfs_search, digraph_bfs_search, edge_bfs, digraph_find_cycle, graph_dfs_search,
    digraph_dfs_search, digraph_dfs_edges, graph_dfs_edges,

    # Shortest path algorithms
    digraph_dijkstra_shortest_paths, graph_dijkstra_shortest_paths,
    graph_dijkstra_shortest_path_lengths, digraph_dijkstra_shortest_path_lengths,
    digraph_all_pairs_dijkstra_path_lengths, digraph_all_pairs_dijkstra_shortest_paths,
    graph_all_pairs_dijkstra_path_lengths, graph_all_pairs_dijkstra_shortest_paths,
    graph_floyd_warshall, digraph_floyd_warshall, graph_floyd_warshall_numpy,
    digraph_floyd_warshall_numpy, graph_floyd_warshall_successor_and_distance,
    digraph_floyd_warshall_successor_and_distance, digraph_bellman_ford_shortest_paths,
    graph_bellman_ford_shortest_paths, digraph_bellman_ford_shortest_path_lengths,
    graph_bellman_ford_shortest_path_lengths, digraph_all_pairs_bellman_ford_path_lengths,
    digraph_all_pairs_bellman_ford_shortest_paths, graph_all_pairs_bellman_ford_path_lengths,
    graph_all_pairs_bellman_ford_shortest_paths, negative_edge_cycle, find_negative_cycle,
    digraph_astar_shortest_path, graph_astar_shortest_path, graph_has_path, digraph_has_path,
    graph_all_simple_paths, digraph_all_simple_paths, graph_all_shortest_paths,
    digraph_all_shortest_paths, graph_all_pairs_all_simple_paths,
    digraph_all_pairs_all_simple_paths, graph_longest_simple_path, digraph_longest_simple_path,
    digraph_k_shortest_path_lengths, graph_k_shortest_path_lengths,

    # DAG Algorithms
    dag_longest_path, dag_longest_path_length, dag_weighted_longest_path,
    dag_weighted_longest_path_length, topological_sort, topological_generations,
    lexicographical_topological_sort, is_directed_acyclic_graph, descendants, ancestors,
    cycle_basis, simple_cycles, dominance_frontiers, immediate_dominators,

    # Centrality
    digraph_eigenvector_centrality, graph_eigenvector_centrality, digraph_betweenness_centrality,
    graph_betweenness_centrality, digraph_edge_betweenness_centrality,
    graph_edge_betweenness_centrality, digraph_closeness_centrality, graph_closeness_centrality,
    graph_newman_weighted_closeness_centrality, digraph_newman_weighted_closeness_centrality,
    graph_degree_centrality, digraph_degree_centrality, in_degree_centrality,
    out_degree_centrality, graph_katz_centrality, digraph_katz_centrality,

    # Graph matrices
    digraph_adjacency_matrix, graph_adjacency_matrix, graph_distance_matrix,
    digraph_distance_matrix,

    # Link analysis
    digraph_pagerank, graph_pagerank, digraph_hits,

    # Isomorphism
    digraph_is_isomorphic, graph_is_isomorphic, digraph_is_subgraph_isomorphic,
    graph_is_subgraph_isomorphic, digraph_vf2_mapping, graph_vf2_mapping,

    # Coloring
    graph_greedy_color, graph_two_color, digraph_two_color, graph_greedy_edge_color,
    graph_misra_gries_edge_color, graph_bipartite_edge_color, graph_is_bipartite,
    digraph_is_bipartite,
    
    # Community detection
    label_propagation_communities,

    # Other algorithms
    transitive_reduction, layers, collect_runs, collect_bicolor_runs, max_weight_matching,
    is_matching, is_maximal_matching, is_planar, is_planar_kurat, minimum_spanning_edges,
    minimum_spanning_tree, graph_line_graph, graph_transitivity, digraph_transitivity,
    graph_token_swapper, graph_core_number, digraph_core_number, graph_complement,
    digraph_complement, local_complement, digraph_maximum_bisimulation,

    # Tensor product
    graph_tensor_product, digraph_tensor_product,

    # Cartesian product
    graph_cartesian_product, digraph_cartesian_product,

    # Union
    graph_union, digraph_union,

    # Rendering
    graphviz_draw,

    # Serialization
    graph_from_dot, digraph_from_dot, graph_to_dot, digraph_to_dot, adjacency_data_to_graph,
    adjacency_data_to_digraph, graph_to_adjacency_data, digraph_to_adjacency_data,
    json_graph_schema, json_digraph_schema, graph_to_json_schema, digraph_to_json_schema,
    json_schema_to_graph, json_schema_to_digraph, graphml_to_digraph, graphml_to_graph,
    json_to_digraph, json_to_graph, pygraphviz_to_digraph, pygraphviz_to_graph,

    # Random graph generation functions
    directed_gnp_random_graph, undirected_gnp_random_graph, directed_gnm_random_graph,
    undirected_gnm_random_graph, undirected_random_bipartite_graph,
    directed_random_bipartite_graph, undirected_sbm_random_graph, directed_sbm_random_graph,
    hyperbolic_random_graph, random_geometric_graph, barabasi_albert_graph,
    directed_barabasi_albert_graph,

    # Layout
    graph_random_layout, digraph_random_layout, graph_spring_layout,
    digraph_spring_layout, graph_kamada_kawai_layout, digraph_kamada_kawai_layout,
    digraph_circular_layout, graph_circular_layout, graph_shell_layout, digraph_shell_layout,
    graph_bipartite_layout, digraph_bipartite_layout, graph_spiral_layout, digraph_spiral_layout,
    graph_multipartite_layout, digraph_multipartite_layout,

    # Constants
    __version__, ColoringStrategy,
)

_S = TypeVar("_S", default=Any)
_T = TypeVar("_T", default=Any)
_BFSVisitor = TypeVar("_BFSVisitor", bound=visit.BFSVisitor)
_DFSVisitor = TypeVar("_DFSVisitor", bound=visit.DFSVisitor)
_DijkstraVisitor = TypeVar("_DijkstraVisitor", bound=visit.DijkstraVisitor)

class PyDAG(Generic[_S, _T], PyDiGraph[_S, _T]): ...

def distance_matrix(
    graph: PyGraph | PyDiGraph,
    parallel_threshold: int = ...,
    as_undirected: bool = ...,
    null_value: float = ...,
) -> npt.NDArray[np.float64]: ...
def unweighted_average_shortest_path_length(
    graph: PyGraph | PyDiGraph,
    parallel_threshold: int = ...,
    disconnected: bool = ...,
) -> float: ...
def adjacency_matrix(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    null_value: float = ...,
) -> npt.NDArray[np.float64]: ...
def all_simple_paths(
    graph: PyGraph | PyDiGraph,
    from_: int,
    to: int,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> list[list[int]]: ...
def floyd_warshall(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    parallel_threshold: int = ...,
) -> AllPairsPathLengthMapping: ...
def floyd_warshall_numpy(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    parallel_threshold: int = ...,
) -> npt.NDArray[np.float64]: ...
def floyd_warshall_successor_and_distance(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def astar_shortest_path(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    node: int,
    goal_fn: Callable[[_S], bool],
    edge_cost_fn: Callable[[_T], float],
    estimate_cost_fn: Callable[[_S], float],
) -> NodeIndices: ...
def dijkstra_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    source: int,
    target: int | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def has_path(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], source: int, target: int, as_undirected: bool = ...
) -> bool: ...
def all_pairs_dijkstra_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathMapping: ...
def all_pairs_all_simple_paths(
    graph: PyGraph | PyDiGraph,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> AllPairsMultiplePathMapping: ...
def all_pairs_dijkstra_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathLengthMapping: ...
def dijkstra_shortest_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def k_shortest_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    start: int,
    k: int,
    edge_cost: Callable[[_T], float],
    goal: int | None = ...,
) -> PathLengthMapping: ...
def all_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    source: int,
    target: int,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> list[list[int]]: ...
def dfs_edges(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], source: int | None = ...) -> EdgeList: ...
@overload
def is_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
@overload
def is_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
@overload
def is_isomorphic_node_match(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    matcher: Callable[[_S, _S], bool],
    id_order: bool = ...,
) -> bool: ...
@overload
def is_isomorphic_node_match(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    matcher: Callable[[_S, _S], bool],
    id_order: bool = ...,
) -> bool: ...
@overload
def is_subgraph_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
@overload
def is_subgraph_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
def transitivity(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> float: ...
def core_number(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> int: ...
def complement(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
) -> PyGraph[_S, _T | None] | PyDiGraph[_S, _T | None]: ...
def random_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def spring_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    pos: dict[int, tuple[float, float]] | None = ...,
    fixed: set[int] | None = ...,
    k: float | None = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: float = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: int = ...,
    scale: int = ...,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def networkx_converter(graph: Any, keep_attributes: bool = ...) -> PyGraph | PyDiGraph: ...
def bipartite_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    first_nodes: set[int],
    horizontal: bool = ...,
    scale: int = ...,
    center: tuple[float, float] | None = ...,
    aspect_ratio=...,
) -> Pos2DMapping: ...
def circular_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    scale: int = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def shell_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    nlist: list[list[int]] | None = ...,
    rotate: float | None = ...,
    scale: int = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def spiral_layout(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    scale: int = ...,
    center: tuple[float, float] | None = ...,
    resolution: float = ...,
    equidistant: bool = ...,
) -> Pos2DMapping: ...
def num_shortest_paths_unweighted(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], source: int
) -> NodesCountMapping: ...
def betweenness_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def closeness_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T], wf_improved: bool = ...
) -> CentralityMapping: ...
def newman_weighted_closeness_centrality(
    graph: PyGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    wf_improved: bool = ...,
    default_weight: float = ...,
) -> CentralityMapping: ...
def degree_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
) -> CentralityMapping: ...
def edge_betweenness_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def eigenvector_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def katz_centrality(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    alpha: float = ...,
    beta: float = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
@overload
def vf2_mapping(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    subgraph: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> Iterator[NodeMap]: ...
@overload
def vf2_mapping(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    subgraph: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> Iterator[NodeMap]: ...
@overload
def union(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyGraph[_S, _T]: ...
@overload
def union(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyDiGraph[_S, _T]: ...
@overload
def tensor_product(
    first: PyGraph,
    second: PyGraph,
) -> tuple[PyGraph, ProductNodeMap]: ...
@overload
def tensor_product(
    first: PyDiGraph,
    second: PyDiGraph,
) -> tuple[PyDiGraph, ProductNodeMap]: ...
@overload
def cartesian_product(
    first: PyGraph,
    second: PyGraph,
) -> tuple[PyGraph, ProductNodeMap]: ...
@overload
def cartesian_product(
    first: PyDiGraph,
    second: PyDiGraph,
) -> tuple[PyDiGraph, ProductNodeMap]: ...
def bfs_search(
    graph: PyGraph | PyDiGraph,
    source: Sequence[int] | None,
    visitor: _BFSVisitor,
) -> None: ...
def dfs_search(
    graph: PyGraph | PyDiGraph,
    source: Sequence[int] | None,
    visitor: _DFSVisitor,
) -> None: ...
def dijkstra_search(
    graph: PyGraph | PyDiGraph,
    source: Sequence[int] | None,
    weight_fn: Callable[[Any], float] | None,
    visitor: _DijkstraVisitor,
) -> None: ...
def bellman_ford_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    source: int,
    target: int | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def bellman_ford_shortest_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def all_pairs_bellman_ford_path_lengths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathLengthMapping: ...
def all_pairs_bellman_ford_shortest_paths(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float] | None,
) -> AllPairsPathMapping: ...
def node_link_json(
    graph: PyGraph[_S, _T] | PyDiGraph[_S, _T],
    path: str | None = ...,
    graph_attrs: Callable[[Any], dict[str, str]] | None = ...,
    node_attrs: Callable[[_S], dict[str, str]] | None = ...,
    edge_attrs: Callable[[_T], dict[str, str]] | None = ...,
) -> str | None: ...
def longest_simple_path(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> NodeIndices | None: ...
def isolates(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> NodeIndices: ...
def two_color(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> dict[int, int]: ...
def is_bipartite(graph: PyGraph[_S, _T] | PyDiGraph[_S, _T]) -> bool: ...
