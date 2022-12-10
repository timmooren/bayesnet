from typing import Union, List
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import itertools


# TODO: shorten the code by removing repetative code

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
                bd_cpt = bn.reduce_factor(
                    instantiation=evidence, cpt=cpt)
                bn.update_cpt(child, reduced_cpt)

        # 2 node-prune: delete any leaf node that doesn’t appear in query or evidence
        to_keep = query + evidence.index.tolist()

        keep_node_pruning = True

        while keep_node_pruning:

            for node in bn.get_all_variables():
                if node not in to_keep and not bn.get_children(node):
                    bn.del_var(node)

                # stop pruning if no nodes were deleted
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
        common_columns = list(
            set(factor1.columns).intersection(factor2.columns))

        common_columns.remove('p')

        output = factor1.merge(factor2, on=common_columns)

        output['p'] = output['p_x'] * output['p_y']
        output.drop(['p_x', 'p_y'], axis=1, inplace=True)

        return output

    def fill(self, interaction_graph: nx.DiGraph) -> str:
        dict_points = {}
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

        return min(dict_points.items(), key=lambda x: x[1])

    def find_order(self, arg: str) -> List[str]:
        """
        Given a set of variables X in the Bayesian network,
        compute a good ordering for the elimination of X
        """
        order = []
        interaction_graph = self.bn.get_interaction_graph()

        # repeats n-1 times
        for i in range(len(interaction_graph.nodes()) - 1):
            degrees = interaction_graph.degree()
            if arg == 'min':
                chosen_node, _ = min(degrees, key=lambda x: x[1])
            if arg == 'fill':
                chosen_node, _ = self.fill(interaction_graph)

            # Queue variable 𝑋 ∈ 𝑿 ⊆ 𝑉 with the minimum degree in the interaction graph to the ordering.
            order.append(chosen_node)

            # find neigbours
            neighbours = list(interaction_graph.neighbors(chosen_node))
            other_neighbours = [n for n in neighbours]

            # connect neighbours with edges
            for neighbour in neighbours:

                # skip when no other neighbours
                if len(other_neighbours) > 1:

                    other_neighbours.remove(str(neighbour))

                    for n in other_neighbours:

                        if not interaction_graph.has_edge(n, neighbour):
                            interaction_graph.add_edge(neighbour, n)

            # remove node and edges
            interaction_graph.remove_node(chosen_node)

            # nx.draw(interaction_graph, with_labels=True)
            # plt.show()
        order.append(list(interaction_graph.nodes())[0])
        return order

    def variable_elimination(self, query: List[str], evidence: List[str]) -> pd.DataFrame:

        # get all variables in the network
        for node in query:
            # if node in evidence.index:
            #     raise ValueError("Query variable cannot be in evidence")

            # get all grandparents
            ancestors = self.find_ancestors(node, [])

            #order = self.find_order('min')
            order = ['A', 'B']

            for ancestor in ancestors:

                tables = [table for table in self.bn.get_all_cpts(
                ).values() if ancestor in table.columns]
                tables_copy = tables.copy()
                for i in range(len(tables_copy)-1):
                    product = self.factor_multiplication(
                        tables[i], tables[i+1])
                    print(f"multiply: \n{tables[i]}, \n{tables[i+1]}")
                    print(node, ancestor)
                    print(f"after mult: \n{product}")
                    tables[i+1] = product

                product = self.marginalization(product, ancestor)

                # remove all tables with ancestor
                # do make list of self.bn.get_all_cpts() (copy) and then remove tables from list when using in multiplication.
                for table in tables:
                    self.bn.del_var(????)

                # add new table to list
                self.bn.add_var("product", product)

                print(f"marg compeleted:\n {product}")

            # visualize results
            # print(node)
            # print(self.bn.get_cpt(node))
        return

    def find_ancestors(self, node: str, ancestors: List[str]):
        ancestors = ancestors

        parents = list(self.bn.structure.predecessors(node))

        if parents:
            for parent in parents:
                ancestors.append(parent)
                ancestors += self.find_ancestors(parent, ancestors)
                ancestors = list(set(ancestors))
        return ancestors
