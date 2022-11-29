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
    def prune(self, query: List[str], evidence: List[str]) -> bool:
        """
        Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network
        """
        successful_prune = False

        # 1: edge-prune: delete the outgoing edges of every node that is in the evidence
        for node in evidence:
            for child in self.bn.get_children(node):
                self.bn.del_edge((node, child))
                successful_prune = True
                # TODO update cpt?

        # 2 delete any leaf node that doesn’t appear in query or evidence
        to_delete = query + evidence

        for node in self.bn.get_all_variables():
            if node not in to_delete and not self.bn.get_children(node):
                self.bn.del_node(node)
                successful_prune

        return successful_prune

    def d_separation(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z through pruning
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

    def marginalization(self, X: List[str]) -> pd.DataFrame:
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.
        """
