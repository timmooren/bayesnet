####################################################################################################
# BNReasoner.py
# Author: Agnes Admiraal, Nikki Moolhuisen, Tim Omar
# Date: 12-12-2020
# Description: This file contains the BNReasoner class, which is used to perform inference on a
####################################################################################################

from typing import Union, List
from BayesNet import BayesNet
from heuristics import min_degree, min_fill
import pandas as pd
import networkx as nx
from copy import deepcopy
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


# --------------------------------- INDEPENDENCE DETERMINATION --------------------------------- #

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
    
     
# --------------------------------- MAX OUT --------------------------------- #

    @staticmethod
    def max_out(factor: pd.DataFrame, X: str) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out.
        Remember to also keep track of which instantiation of X led to the maximized value.
        """
        groups = factor.columns.to_list()
        groups.remove(X)
        groups.remove('p')
        
        return factor.groupby(groups, as_index=False).max()


# --------------------------------- HELPER FUCTIONS BAYES --------------------------------- #

    @staticmethod
    def marginalization(factor: pd.DataFrame,  variable: str) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.
        """
        groups = factor.columns.to_list()
        groups.remove(variable)
        groups.remove('p')
        cpt = factor.groupby(groups, as_index=False).sum()
        #print(cpt)
        cpt.drop(variable, axis=1, inplace=True)

        return cpt
    
    @staticmethod
    def factor_multiplication(factor1: pd.DataFrame, factor2: pd.DataFrame) -> pd.DataFrame:
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


    # --------------------------------- BAYES PROBABILITY FUNCTIONS --------------------------------- #
    
    def variable_elimination(self, query: List[str], evidence: pd.Series = pd.Series()) -> pd.DataFrame:
        """
        Given a set of variables X in the Bayesian network, 
        compute the CPT of X given the evidence.
        """      
        
        # get ancestors of the query variable
        ancestors_sets = [nx.ancestors(self.bn.get_interaction_graph(), node) for node in query]
        intersection_set = set.union(*ancestors_sets)
        ancestors = list(intersection_set)
        print(f"ancestors \n {ancestors}")
        
        # remove query from collective ancestors
        for node in query:
            ancestors = [ancestor for ancestor in ancestors if not node in ancestors]
        
        # get the order of elimination       
        order = self.find_order(ancestors)     
        
        # make a list of all cpts in the BN
        cpt_list = list(self.bn.get_all_cpts().values())
        print(f"cpt_list: \n{cpt_list}")
        
        # marginalize for every ancestor
        for ancestor in order: 
            
            # get cpts for the variable and its ancestors
            for variable in ancestors:
                tables_ancestors = [table for table in cpt_list if (ancestor in table.columns and variable in table.columns)]
            tables_copy = tables_ancestors.copy()

            # multiply all cpts for every cpt
            for i in range(len(tables_copy)-1):
                product = self.factor_multiplication(tables_ancestors[i], tables_ancestors[i+1])
                print(f"multiply: \n {tables_ancestors[i]}")
                print(f"multiply: \n {tables_ancestors[i+1]}")
                print(f"product: \n{product}")
                tables_ancestors[i+1]= product
            
            # remove all instantiations that are incompatible with the evidence
            if not evidence.empty:
                self.bn.get_compatible_instantiations_table(evidence, product)  
            
            # remove already multiplied cpts from list
            cpt_list = [table for table in cpt_list if ancestor not in table.columns]

            # and append summed out table of ancestor to list
            product = self.marginalization(product, ancestor)
            print(f"marginalized:\n {product}")
            cpt_list.append(product)
            
            #print("DONE WITH LOOP")

        return product
    
    
    def marginal_distributions(self, query: List[str], evidence: pd.Series) -> pd.DataFrame:
        """ Sum out a set of variables by using variable elimination."""
        prior_marginal = self.variable_elimination(query, evidence)
        evidence_cpt = self.bn.get_cpt(evidence.index)
        print(evidence_cpt)
        prior_prob_evidence = evidence_cpt[evidence.index == evidence.values]

        return prior_marginal
         

# --------------------------------- ORDER DETERMINATION BAYES -------------------------------- #
    
    
    def find_order(self, variables = List[str]) -> List[str]:
        """
        Given a set of variables X in the Bayesian network, 
        compute a good ordering for the elimination of X
        """
        
        self.order_method = 'min'                                           # TODO: for some reason does not work as class attribute if this line is left out
        order = [] # list of nodes in order of elimination
        interaction_graph = self.bn.get_interaction_graph()     
        
        variables_copy = variables.copy()
        
        # elliminate node by node until all nodes are eliminated
        for node in variables_copy:
            
            # determine wich node to elliminate next                        # TODO: implement other methods? 
            if self.order_method == 'min':
                chosen_node = min_degree(interaction_graph, variables)
                
            if self.order_method == 'fill':
                chosen_node = min_fill(interaction_graph, variables)
            
            order.append(chosen_node)  
            variables.remove(chosen_node)         
            
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
        
        return order
   
       
