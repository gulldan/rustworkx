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
Example script demonstrating the label propagation community detection algorithm.
"""

import rustworkx as rx
import matplotlib.pyplot as plt
import networkx as nx

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
    
    # Visualize the graph with communities
    try:
        # Convert to NetworkX for visualization
        nx_graph = nx.Graph()
        
        # Add nodes with community attributes
        community_map = {}
        for i, community in enumerate(communities):
            for node_idx in community:
                community_map[node_idx] = i
        
        for i, data in enumerate(graph.nodes()):
            nx_graph.add_node(i, label=data, community=community_map.get(i, -1))
        
        # Add edges
        for edge in graph.edge_list():
            u, v = edge
            nx_graph.add_edge(u, v)
        
        # Create a layout
        pos = nx.spring_layout(nx_graph, seed=42)
        
        # Draw nodes colored by community
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
        
        for comm_id in set(nx.get_node_attributes(nx_graph, 'community').values()):
            comm_nodes = [n for n, d in nx_graph.nodes(data=True) 
                          if d.get('community') == comm_id]
            nx.draw_networkx_nodes(
                nx_graph, pos, 
                nodelist=comm_nodes,
                node_color=colors[comm_id % len(colors)], 
                node_size=500,
                alpha=0.8
            )
        
        # Draw edges and labels
        nx.draw_networkx_edges(nx_graph, pos, width=1.0, alpha=0.5)
        labels = {n: d.get('label') for n, d in nx_graph.nodes(data=True)}
        nx.draw_networkx_labels(nx_graph, pos, labels, font_size=10)
        
        plt.title("Label Propagation Communities")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('label_propagation_communities.png')
        print("Graph visualization saved to 'label_propagation_communities.png'")
    except ImportError:
        print("NetworkX or matplotlib not available for visualization")

if __name__ == '__main__':
    main() 