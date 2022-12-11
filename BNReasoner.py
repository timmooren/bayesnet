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


    # --------------------------------- BAYES PROBABILITY FUNCTIONS --------------------------------- #

    def variable_elimination(self, query: List[str], evidence: pd.Series = pd.Series()) -> pd.DataFrame:
        """
        Given a set of variables X in the Bayesian network,
        compute the CPT of X given the evidence.
        return: prior marginal
        """
        # get ancestors of the query variable
        interaction_graph = self.bn.get_interaction_graph()
        ancestors_sets = [nx.ancestors(interaction_graph, node) for node in query]
        intersection_set = set.union(*ancestors_sets)
        ancestors = list(intersection_set)

        # remove query from collective ancestors
        ancestors = [ancestor for ancestor in ancestors if not ancestor in query]

        # get the order of elimination
        order = self.find_order(ancestors)

        # make a list of all cpts of ancestors and query in the BN
        all_variables = order + query
        cpt_list = []

        for variable in all_variables:
            cpt = self.bn.get_cpt(variable)

            # in case of evidence, remove incompatible rows from cpt
            if not evidence.empty:
                cpt = self.bn.get_compatible_instantiations_table(evidence, cpt)

            cpt_list.append(cpt)

        # marginalize for every ancestor
        for ancestor in order:
            # get cpts for the variable and its ancestors
            tables_ancestors = [table for table in cpt_list if ancestor in table.columns]
            tables_copy = tables_ancestors.copy()

            # chain rule: multiply all cpts for every cpt
            for i in range(len(tables_copy)-1):
                product = self.factor_multiplication(tables_ancestors[i], tables_ancestors[i+1])
                tables_ancestors[i+1]= product

            # remove already multiplied cpts from list
            cpt_list = [table for table in cpt_list if ancestor not in table.columns]

            # and append summed out table of ancestor to list
            product = self.marginalization(product, ancestor)
            cpt_list.append(product)

        # in case of multiple variables in query, and a variable in query is an ancestor of other variable in query
        # there will be a factor 'outside' of the summations, this factor also needs to be multiplied
        if len(cpt_list) > 1:
            for i in range(len(cpt_list)-1):
                product = self.factor_multiplication(cpt_list[i], cpt_list[i+1])
                cpt_list[i+1] = product

        return product


    def marginal_distributions(self, query: List[str], evidence: pd.Series) -> pd.DataFrame:
        """
        Sum out a set of variables by using variable elimination.
        return: posterior marginal
        """
        # compute Pr(Q ^ E)
        prior_marginal = self.variable_elimination(query, evidence)
        posterior_marginal = prior_marginal

        for variable, value in evidence.items():
            # compute the marginal P(E)
            cpt_evidence = self.bn.get_cpt(variable)
            marginal = cpt_evidence.loc[cpt_evidence[variable]==value, 'p'].iloc[0]

            #compute the posterior marginal Pr(Q|E) = Pr(Q ^ E) / P(E)
            posterior_marginal['p'] = posterior_marginal['p'] / marginal

        return posterior_marginal


    # --------------------------------- Most Likely Instantiations --------------------------------- #


    def MAP(self, query: List[str], evidence: pd.Series) -> pd.DataFrame:
        """
        Compute the maximum a-posteriory instantiation + value of query variables Q,
        given a possibly empty evidence e
        """
        # get the marginal distribution
        posterior_marginal = self.marginal_distributions(query, evidence)

        # get the most likely instantiation
        max_out = self.max_out(posterior_marginal, query)

        return max_out


    def MPE(self, evidence: pd.Series) -> pd.Series:
        """
        Compute the most likely instantiation of the query variables given the evidence.
        """
        # get all variables in the BN
        variables = self.bn.get_variables()
        # get all query variables
        query = [variable for variable in variables if not variable in evidence.index]

        return self.MAP(query, evidence)


        # # compute the marginal distributions
        # posterior_marginal = self.marginal_distributions(query, evidence)

        # # max out
        # max_out = self.max_out(posterior_marginal, query)

        # # get the most likely instantiation
        # # most_likely_instantiation = marginal_distributions.loc[marginal_distributions['p'].idxmax(), query]

        # return max_out