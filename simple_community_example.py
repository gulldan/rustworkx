#!/usr/bin/env python3
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
Simple example of the label propagation community detection algorithm
"""

import rustworkx as rx

def main():
    """Create a graph with two communities and detect them using label propagation"""
    
    # Create a simple graph with two clear communities
    graph = rx.PyGraph()
    
    # Add nodes to the first community
    comm1_nodes = []
    for i in range(5):
        comm1_nodes.append(graph.add_node(f"C1-{i}"))
    
    # Add nodes to the second community
    comm2_nodes = []
    for i in range(5):
        comm2_nodes.append(graph.add_node(f"C2-{i}"))
    
    # Create dense connections within communities
    for i in range(len(comm1_nodes)):
        for j in range(i+1, len(comm1_nodes)):
            graph.add_edge(comm1_nodes[i], comm1_nodes[j], None)
    
    for i in range(len(comm2_nodes)):
        for j in range(i+1, len(comm2_nodes)):
            graph.add_edge(comm2_nodes[i], comm2_nodes[j], None)
    
    # Add a single edge between communities
    graph.add_edge(comm1_nodes[0], comm2_nodes[0], None)
    
    # Detect communities using label propagation
    communities = rx.label_propagation_communities(graph)
    
    print(f"Number of communities detected: {len(communities)}")
    
    # Print the communities
    for i, community in enumerate(communities):
        print(f"Community {i+1}:")
        for node_idx in community:
            node_data = graph.nodes()[node_idx]
            print(f"  - Node {node_idx}: {node_data}")
    
    # Print summary stats about the graph
    print("\nGraph summary:")
    print(f"  - Total nodes: {len(graph)}")
    print(f"  - Total edges: {len(graph.edges())}")
    print(f"  - Number of communities: {len(communities)}")
    
    # Print connections between communities
    print("\nConnections between communities:")
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    
    inter_community_edges = 0
    for edge in graph.edges():
        u, v = edge
        if community_map[u] != community_map[v]:
            inter_community_edges += 1
            print(f"  - Edge between community {community_map[u]+1} and {community_map[v]+1}: node {u} ({graph.nodes()[u]}) -> node {v} ({graph.nodes()[v]})")
    
    print(f"\nTotal inter-community edges: {inter_community_edges}")
    print(f"Total intra-community edges: {len(graph.edges()) - inter_community_edges}")

if __name__ == "__main__":
    main() 