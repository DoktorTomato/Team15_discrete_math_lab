'''
This script is created to implement floyd-warshall algorythm
'''

import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby

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

G = gnp_random_connected_graph(5, 1, True, False)
inf = float('inf')

def generate_mtrx(graph:nx.DiGraph)->list:
    '''
    This function generates a matrix of weights for given weighted graph
    '''
    weighted_mtrx = []
    for node_ in graph.nodes:
        tmp_lst = []
        for neighbor in graph.nodes:
            if neighbor == node_:
                tmp_lst.append(0)
            else:
                try:
                    tmp_lst.append(graph[node_][neighbor]['weight'])
                except KeyError:
                    tmp_lst.append(inf)
        weighted_mtrx.append(tmp_lst.copy())
    return weighted_mtrx

def floyd_warshall_mtrx(mtrx:list)->list:
    '''
    This function does floy_warshall algorythm using matrix of weights given
    '''
    for node_, line in enumerate(mtrx):
        for y, x in enumerate(mtrx):
            if y == node_ or x[node_] == inf:
                continue
            tmp_lst = []
            for ind, w in enumerate(x):
                if ind == y:
                    if (min(x[node_] + line[ind], w)) < 0:
                        return 'Negative cycle detected'
                tmp_lst.append((min(x[node_] + line[ind], w)))
            mtrx[y] = tmp_lst
    return mtrx

def from_lst_dct(mtrx:list)->dict:
    '''
    This function turns our result from list of lists to a dict
    '''
    if isinstance(mtrx, str):
        return mtrx
    res = {}
    for source, dests in enumerate(mtrx):
        weights_dct = {}
        for dest, weight in enumerate(dests):
            weights_dct[dest] = weight
        res[source] = weights_dct
    return res

def floyd_warshall_alg(graph:nx.DiGraph)->dict:
    '''
    This function does floyd_warshall_alg with all the steps inside it
    '''
    return from_lst_dct(floyd_warshall_mtrx(generate_mtrx(graph)))

print(generate_mtrx(G))
print(floyd_warshall_alg(G))
