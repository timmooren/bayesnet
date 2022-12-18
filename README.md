# bayesnet

## Welcome to the Bayesian Network Reasoner!

A Bayesian network is a probabilistic graphical model that represents a set of variables and their probabilistic dependencies. It is a powerful tool for representing and reasoning about complex systems, and can be used to make predictions and decisions under uncertainty. This Bayesian network reasoner allows you to create, edit, and reason with Bayesian networks.  You can use the reasoning algorithms to answer probabilistic queries about your system. For example, you can use the probability of a given variable given the values of other variables, or compute the expected value of a variable.

## BNReasoner Class
The BNReasoner class is used to perform inference on a Bayesian network. It provides a range of functions for querying the network, including methods for independence determination, variable elimination, and maximum a posteriori (MAP) and maximum probable explanation (MPE) inference.

To use the BNReasoner class, you will first need to instantiate it with a Bayesian network in BIFXML format or as a BayesNet object. You can then use the various methods provided to query the network and perform inference.  Examples of Bayesian Networks are provided in the testing folder.

Some of the key features of the BNReasoner class include:

Independence determination: the d_separation method allows you to determine whether two sets of variables are independent given a third set of variables.

Variable elimination: the variable elimination method performs variable elimination on the network to compute the probability of a given query variable given the values of a set of evidence variables.

MAP and MPE inference: the map and mpe inference methods allow you to compute the maximum a posteriori and maximum probable explanation of a given query variable given the values of a set of evidence variables.

### Good to know:
This Bayesian Network Reasoner is only build to process binary coded Bayesian Networks.

### representations:
- variable: str
- instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
- cpt: pd.Dataframe
- factor: pd.Dataframe
- edges: List[Tuple[str, str]]
