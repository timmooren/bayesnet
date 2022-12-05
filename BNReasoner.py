from typing import Union, List
from BayesNet import BayesNet
import pandas as pd
import networkx as nx


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
        successful_prune = False

        # 1: edge-prune: delete the outgoing edges of every node that is in the evidence
        for node in evidence.index:
            for child in self.bn.get_children(node):
                self.bn.del_edge((node, child))

                # Replace the factor by a reduced factor
                cpt = self.bn.get_cpt(node)
                reduced_cpt = self.bn.reduce_factor(
                    instantiation=evidence, cpt=cpt)
                self.bn.update_cpt(node, reduced_cpt)

        # 2 node-prune: delete any leaf node that doesn’t appear in query or evidence
        to_keep = query + evidence.index.tolist()

        for node in self.bn.get_all_variables():
            if node not in to_keep and self.bn.is_leaf(node):
                self.bn.del_node(node)
                successful_prune

        return successful_prune

    def d_separation(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z,
        determine whether X is d-separated of Y given Z through pruning
        """
        while self.prune(X + Y, Z):
            continue

        # if there is a path from X to Y, then X is not d-separated from Y given Z
        for x in X:
            for y in Y:
                if nx.has_path(self.bn.structure, x, y):
                    return False

        return True

    def independence(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z
        """
        return self.d_separation(X, Y, Z)

    def marginalization(self, factor: pd.DataFrame,  X: List[str]) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.
        """
        return factor.groupby(X, as_index=False).sum()

    def max_out(self, factor: pd.DataFrame, X: List[str]) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out.
        Remember to also keep track of which instantiation of X led to the maximized value.
        """
        return factor.groupby(X, as_index=False).max()

    def factor_multiplication(self, factor1: pd.DataFrame, factor2: pd.DataFrame) -> pd.DataFrame:
        """
        Given two factors f and g, compute the multiplied factor h=fg.
        """
        return factor1.merge(factor2, on=self.bn.get_all_variables())

    def ordering(self, X: List[str]) -> List[str]:
        """
        Given a set of variables X in the Bayesian network,
        compute a good ordering for the elimination of X
        based on the min-degree heuristics and the min-fill heuristics
        """
        interaction_graph = self.bn.get_interaction_graph(X)
