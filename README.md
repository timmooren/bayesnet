# bayesnet

## Welcome to the Bayesian Network Reasoner!

A Bayesian network is a probabilistic graphical model that represents a set of variables and their probabilistic dependencies. It is a powerful tool for representing and reasoning about complex systems, and can be used to make predictions and decisions under uncertainty. This Bayesian network reasoner allows you to create, edit, and reason with Bayesian networks.  You can use the reasoning algorithms to answer probabilistic queries about your system. For example, you can use the probability of a given variable given the values of other variables, or compute the expected value of a variable. Please note that this Bayesian Network Reasoner is only build to process binary coded Bayesian Networks.

## BNReasoner Class
The BNReasoner class is used to perform inference on a Bayesian network. It provides a range of functions for querying the network, including methods for independence determination, variable elimination, and maximum a posteriori (MAP) and maximum probable explanation (MPE) inference.

To use the BNReasoner class, you will first need to instantiate it with a Bayesian network in BIFXML format or as a BayesNet object. You can then use the various methods provided to query the network and perform inference.  Examples of Bayesian Networks are provided in the testing folder.


### Methods

The BNReasoner class contains the following methods:

- __init__: initializes the BNReasoner class with a Bayesian network in BIFXML format or as a BayesNet object, and a method and heuristic for use in variable elimination.

- load_new_bn: allows you to load an existing Bayesian network into the BNReasoner.

- prune: given a set of query variables and evidence, performs node- and edge-pruning on the Bayesian network.

- d_separation: determines whether two sets of variables are independent given a third set of variables, using the d-separation criterion.

- variable_elimination: performs variable elimination on the Bayesian network to compute the probability of a given query variable given the values of a set of evidence variables.

- MAP: computes the maximum a posteriori of a given query variable given the values of a set of evidence variables.

- MPE: computes the maximum probable explanation of a given query variable given the values of a set of evidence variables.

Additionally, the class includes several other methods and functions, such as max_out, marginalization, and factor_multiplication, which are used to support the above methods.

### Running the code
To run this code, you will need to do the following:

Make sure that you have the necessary dependencies installed. The code imports several modules, including typing, BayesNet, heuristics, helper_functions, pandas, networkx, copy, and matplotlib. You will need to have these modules installed in your environment in order to run the code.

Instantiate the BNReasoner class. To do this, you will need to specify a Bayesian network in BIFXML format or as a BayesNet object, as well as a method and heuristic for use in variable elimination. For example:

```
reasoner = BNReasoner(net='my_network.BIFXML', method='min', heuristic='min-degree')
```

Use the methods of the BNReasoner class to perform inference on the network. For example, you could use the variable elimination method to perform variable elimination, or the map_inference method to compute the maximum a posteriori of a given query variable.
Here is an example of how you might use the variable elimination method:

```
# This would compute the probability of the variable P given the values of the variables B and E. 
query_variable = 'P'
evidence_variables = {'B': 1, 'E': 0}
result = reasoner.variable_elimination(query_variable, evidence_variables)
```

### Representations:
- variable: str
- instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
- cpt: pd.Dataframe
- factor: pd.Dataframe
- edges: List[Tuple[str, str]]
