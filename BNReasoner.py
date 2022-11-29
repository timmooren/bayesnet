from typing import Union, List
from BayesNet import BayesNet
import pandas as pd

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
    def prune(self, query : List[str], evidence: pd.Series) -> None:
        """
        Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network
        """

        # 1: edge-prune: delete the outgoing edges of every node that is in the evidence
        for node in evidence:
            for child in self.bn.get_children(node):
                self.bn.del_edge((node, child))
                # TODO update cpt

        # 2 delete any leaf node that doesn’t appear in query or evidence
        to_delete = query + list(evidence.index)

        for node in self.bn.get_all_variables():
            if node not in to_delete:
                # delete if node is leaf
                if not self.bn.get_children(node):
                    self.bn.del_node(node)


