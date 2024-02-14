import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
from networkx.algorithms import tree
# You can use this function to generate a random graph with 'num_of_nodes' nodes
# and 'completeness' probability of an edge between any two nodes
# If 'directed' is True, the graph will be directed
# If 'draw' is True, the graph will be drawn
def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               directed: bool = False,
                               draw: bool = False):
    """
    Generates a random graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted (in case of undirected graphs)
    """

    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    G.add_nodes_from(range(num_of_nodes))
    
    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
                
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(-5, 20)
                
    if draw: 
        plt.figure(figsize=(10,6))
        if directed:
            # draw with edge weights
            pos = nx.arf_layout(G)
            nx.draw(G,pos, node_color='lightblue', 
                    with_labels=True,
                    node_size=500, 
                    arrowsize=20, 
                    arrows=True)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
            
        else:
            nx.draw(G, node_color='lightblue', 
                with_labels=True, 
                node_size=500)
        
    return G

G = gnp_random_connected_graph(7, 1, False, True)
def kruskal(graph):

    sets = [{node} for node in graph.nodes()]

    line_list = []
    weight = 0

    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])

    for (u, v, w) in sorted_edges:
        u_set = next(s for s in sets if u in s)
        v_set = next(s for s in sets if v in s)

        if u_set != v_set:
            line_list.append((u, v))
            weight += w['weight']

            u_set.update(v_set)
            sets.remove(v_set)
    return line_list, weight
def prim(graph):
    graph = {node: neighbors for node, neighbors in graph.adjacency()}
    result = []
    start_vertex = next(iter(graph))
    visited = {start_vertex}
    edges = [(weight_dict['weight'], start_vertex, to) for to, weight_dict in graph[start_vertex].items()]

    while len(visited) < len(graph):
        min_edge = min(edges)
        weight, frm, to = min_edge
        edges.remove(min_edge)
        if to not in visited:
            visited.add(to)
            result.append((frm, to, weight))
            for next_to, next_weight_dict in graph[to].items():
                if next_to not in visited:
                    next_weight = next_weight_dict['weight']
                    edges.append((next_weight, to, next_to))

    total_weight = sum(weight for _, _, weight in result)
    return result, total_weight