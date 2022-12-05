from typing import Union, List
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def prune(self, query: List[str], evidence: pd.Series) -> bool:
        """
        Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network
        """
        # create a copy of the structure
        bn = deepcopy(self.bn)

        # 1: edge-prune: delete the outgoing edges of every node that is in the evidence
        for node in evidence.index:
            for child in bn.get_children(node):
                bn.del_edge((node, child))

                # Replace the factor by a reduced factor
                cpt = bn.get_cpt(child)
                reduced_cpt = bn.reduce_factor(
                    instantiation=evidence, cpt=cpt)
                bn.update_cpt(child, reduced_cpt)

        # 2 node-prune: delete any leaf node that doesn’t appear in query or evidence
        to_keep = query + evidence.index.tolist()

        keep_node_pruning = True
        while keep_node_pruning:
            keep_node_pruning = False
            for node in bn.get_all_variables():
                if node not in to_keep and not bn.get_children(node):
                    bn.del_var(node)
                    keep_node_pruning = True

        return bn



    def d_separation(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z,
        determine whether X is d-separated of Y given Z (evidence) through pruning
        """
        # convert Z to evidence pd series
        Z = pd.Series({z: True for z in Z})

        bn = self.prune(query=X+Y, evidence=Z)

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
        common_columns = list(set(factor1.columns).intersection(factor2.columns))
        common_columns.remove('p')

        output = factor1.merge(factor2, on=common_columns)
        output['p'] = output['p_x'] * output['p_y']

        return output


    def min_degree(self, X: List[str]) -> List[str]:
        """
        Given a set of variables X in the Bayesian network,
        compute a good ordering for the elimination of X
        based on the min-degree heuristics and the min-fill heuristics
        """
        order = []
        interaction_graph = self.bn.get_interaction_graph()
        # plot the interaction graph
        # nx.draw(interaction_graph, with_labels=True)
        # plt.show()
        # find variable with the minimum degree in the interaction graph
        degrees = interaction_graph.degree()
        min_degree, _ = min(degrees, key=lambda x: x[1])

        # Queue variable 𝑋 ∈ 𝑿 ⊆ 𝑉 with the minimum degree in the interaction graph to the ordering.
        order.append(min_degree)

        # Sum-out 𝑋 from the interaction graph.



    def min_fill(self, X: List[str]) -> List[str]:
        """
        Given a set of variables X in the Bayesian network,
        compute a good ordering for the elimination of X
        based on the min-degree heuristics and the min-fill heuristics
        """
        interaction_graph = self.bn.get_interaction_graph()