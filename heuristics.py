import networkx as nx
from typing import Union, List

# --------------------------------- ORDERING HEURISTICS --------------------------------- #	

def min_degree(interaction_graph: nx.DiGraph, variables: List[str])-> str:
    """
    Method 1: min-degree heuristic.
    Given a dictionary of nodes with their degrees, return the node which should be eliminated next.
    """
    # compute number of edges for each node
    degrees = dict(interaction_graph.degree())
    
    # only consider the variables
    new_dict = {node: degrees[node] for node in variables}
    
    # get the node with the minimum number of neighbours
    node = min(new_dict, key=degrees.get) 
    return node 

def min_fill(interaction_graph: nx.DiGraph, variables = List[str])-> str:
    """
    Method 2: fill-in heuristic.
    Given an interaction graph, return the node which should be eliminated next.
    """
    dict_points = dict() # dict to store the number of penalty points for each node
    interaction_graph = interaction_graph
    
    for node in variables:
        
        neighbours = list(interaction_graph.neighbors(node))
        other_neighbours = [n for n in neighbours]
        
        dict_points[node] = 0
        
        # connect neighbours with edges
        for neighbour in neighbours:
            
            if len(other_neighbours) > 1:

                other_neighbours.remove(neighbour)
                
                # if neighbour has no connection to one of other neighbours, add penalty point
                for n in other_neighbours:
                    
                    if not interaction_graph.has_edge(n, neighbour):
                        dict_points[node] += 1
    
    # find node with minimum penalty points
    elimination_order = min(dict_points, key=dict_points.get)
    return elimination_order
