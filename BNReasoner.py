####################################################################################################
# BNReasoner.py
# Author: Agnes Admiraal, Nikki Moolhuisen, Tim Omar
# Date: 12-12-2020
# Description: This file contains the BNReasoner class, which is used to perform inference on a
####################################################################################################

from typing import Union, List
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

class BNReasoner():
    
    def __init__(self, 
                 order_method: str = 'min', #'min' or 'fill'
                 net: Union[str, BayesNet] = 'testing/lecture_example2.BIFXML'):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        :param method: the ordering method to use for variable elimination:
            'min' for using the min-degree heuristic
            'fill' for using the fill-in heuristic 
        """ 
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
            
        else:
            self.bn = net
        
        # the ordering method to use for variable elimination
        self.order_method = order_method


# --------------------------------- NETWORK PRUNING --------------------------------- #

    def prune(self, query: List[str], evidence: pd.Series) -> bool:
            """
            Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network
            """
            keep_node_pruning = True
            
            # create a copy of the structure
            bn = deepcopy(self.bn)

            # 1: edge-prune: delete the outgoing edges of every node that is in the evidence
            for node in evidence.index:
                for child in bn.get_children(node):
                    bn.del_edge((node, child))

                    # Replace the factor by a reduced factor
                    cpt = bn.get_cpt(child)
                    reduced_cpt = bn.reduce_factor(instantiation=evidence, cpt=cpt)
                    bn.update_cpt(child, reduced_cpt)

            # 2: node-prune: delete any leaf node that doesn’t appear in query or evidence
            to_keep = query + evidence.index.tolist()
            
            # keep pruning until no nodes can be deleted
            while keep_node_pruning:
                
                for node in bn.get_all_variables():
                
                    # prune leaf node
                    if node not in to_keep and not bn.get_children(node):
                        bn.del_var(node)
                
                    # stop pruning
                    else:
                        keep_node_pruning = False

            return bn  
    
     
# --------------------------------- HELPER FUNCTIONS --------------------------------- #

    def marginalization(self, factor: pd.DataFrame,  X: str) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.
        """
        groups = factor.columns.to_list()
        groups.remove(X)
        groups.remove('p')
        
        return factor.groupby(groups, as_index=False).sum()
    

    def max_out(self, factor: pd.DataFrame, X: str) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out.
        Remember to also keep track of which instantiation of X led to the maximized value.
        """
        groups = factor.columns.to_list()
        groups.remove(X)
        groups.remove('p')
        
        return factor.groupby(groups, as_index=False).max()


    def factor_multiplication(self, factor1: pd.DataFrame, factor2: pd.DataFrame) -> pd.DataFrame:
        """
        Given two factors f and g, compute the multiplied factor h=fg.
        """
        # find the common columns of the two factors
        common_columns = list(set(factor1.columns).intersection(factor2.columns))
        common_columns.remove('p')

        # merge the two factors by multiplying the probabilities
        output = factor1.merge(factor2, on=common_columns)   
        output['p'] = output['p_x'] * output['p_y']
        output.drop(['p_x', 'p_y'], axis=1, inplace=True)
    
        return output


    def find_ancestors(self, node : str, ancestors : List[str]) -> List[str]: 
        """
        Given a node, find all ancestors of this node
        """
        ancestors = ancestors
        
        # find parents of node
        parents = list(self.bn.structure.predecessors(node))
        
        # stop when no parents are found
        if parents:
            
            # find all ancestors of parents
            for parent in parents:
                ancestors.append(parent)
                ancestors += self.find_ancestors(parent, ancestors)
                
                # remove duplicates
                ancestors = list(set(ancestors))
                
        return ancestors
    
    
    def find_order(self) -> List[str]:
        """
        Given a set of variables X in the Bayesian network, 
        compute a good ordering for the elimination of X
        """
        
        self.order_method = 'fill'                                           # TODO: for some reason does not work as class attribute if this line is left out
        order = [] # list of nodes in order of elimination
        interaction_graph = self.bn.get_interaction_graph()
        
        # elliminate node by node until all nodes are elliminated
        for i in range(len(interaction_graph.nodes())-1):
            
            # compute number of edges for each node
            degrees = interaction_graph.degree()
            
            # determine wich node to elliminate next                        # TODO: implement other methods? 
            if self.order_method == 'min':
                chosen_node, _ = self.min_degree(degrees)
            if self.order_method == 'fill':
                chosen_node, _ = self.fill_in(interaction_graph)
              
            order.append(chosen_node)
            
            # find direct neigbour nodes
            neighbours = list(interaction_graph.neighbors(chosen_node))
            other_neighbours = [n for n in neighbours]
            
            # create new neighbourhood for each neighbour
            for neighbour in neighbours:
                
                # skip when no other neighbours
                if len(other_neighbours) > 1:
    
                    # prevent neigbour from connecting to itself
                    other_neighbours.remove(str(neighbour))
                    
                    # fill in missing edges between neighbours in neighbourhood
                    for n in other_neighbours:
                
                        if not interaction_graph.has_edge(n, neighbour):
                            interaction_graph.add_edge(neighbour, n)

            # remove current node and edges
            interaction_graph.remove_node(chosen_node)     
        
        # add last node  
        order.append(list(interaction_graph.nodes())[0])
        
        return order
   
   
    # --------------------------------- BAYES PROBABILITY FUNCTIONS --------------------------------- #
    
    def bayes_probability(self, query: List[str], evidence: List[str]) -> pd.DataFrame:
        """
        Given a set of variables X in the Bayesian network, 
        compute the CPT of X given the evidence.
        """      
        for node in query:
        
            # get ancestors of the query variable
            ancestors = self.find_ancestors(node, [])

            # get the order of elimination       
            order = self.find_order()     
            order = [element for element in order if element in ancestors]
            
            # make a list of all cpts in the BN
            list_tables = list(self.bn.get_all_cpts().values())
            
            for ancestor in order:   
                
                # get cpts for the variable and its ancestors
                tables_ancestors = [table for table in list_tables if ancestor in table.columns]
                tables_copy = tables_ancestors.copy()

                # multiply all cpts of this variable
                for i in range(len(tables_copy)-1):
                    product = self.factor_multiplication(tables_ancestors[i], tables_ancestors[i+1])
                    tables_ancestors[i+1]= product
                
                # remove allready multiplied cpts from list
                order.remove(ancestor)

                # and append summed out table of ancestor to list
                product = self.marginalization(product, ancestor)
                list_tables.append(product)

        return product
    
    
    def marginal_distributions(self, query: List[str], evidence: List[str]) -> pd.DataFrame:
        """
        Given a set of variables X in the Bayesian network, 
        compute the marginal distribuition of X given the evidence.
        """
        # TODO: implement                                                                                       # kan mss ook in de bayes probability functie
        return


     # --------------------------------- DEPENDANCY DETERMINATION --------------------------------- #

    def d_separation(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z,
        determine whether X is d-separated of Y given Z (evidence) through pruning
        """
        # convert Z to evidence pd series
        Z = pd.Series({z: True for z in Z})

        bn = self.prune(query=X+Y, evidence=Z)                                                                  # TODO: eigenlijk mag dit buiten de functie en moet dit in de main? want neem aan dat we dit ook voor andere functies nodig hebben

        # convert directed graph to undirected graph
        undirected = bn.structure.to_undirected()

        # if there is a path from X to Y, then X is not d-separated from Y given Z
        for x in X:
            for y in Y:
                if nx.has_path(undirected, x, y):
                    return False

        return True
    

    def independence(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z
        """
        return self.d_separation(X, Y, Z)
       
    # --------------------------------- ORDERING HEURISTICS --------------------------------- #	

    def min_degree(self, degrees: dict)-> str:
        """
        Method 1: min-degree heuristic.
        Given a dictionary of nodes with their degrees, return the node which should be eliminated next.
        """
        # get the dnode with the minimum number of neighbours
        next_node = min(degrees, key=lambda x: x[1]) 
        return next_node 


    def fill_in(self, interaction_graph: nx.DiGraph)-> str:
        """
        Method 2: fill-in heuristic.
        Given an interaction graph, return the node which should be eliminated next.
        """
        dict_points = {} # dict to store the number of penalty points for each node
        interaction_graph = interaction_graph
        
        for node in interaction_graph.nodes():
            
            neighbours = list(interaction_graph.neighbors(node))
            other_neighbours = [n for n in neighbours]
            
            dict_points[node] = 0
            
            # connect neighbours with edges
            for neighbour in neighbours:
                
                if len(other_neighbours) > 1:

                    other_neighbours.remove(neighbour)
                    
                    # if neighbour has no connection to one of other neighbours, add edge
                    for n in other_neighbours:
                        
                        if not interaction_graph.has_edge(n, neighbour):
                            dict_points[node] += 1
        
        # find node with minimum penalty points
        elimination_order = min(dict_points.items(), key=lambda x: x[1])
     
        return elimination_order
    
    
    
