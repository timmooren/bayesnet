# %%
import BayesNet
import networkx as nx
from BNReasoner import BNReasoner
import pandas as pd

# %%
path = 'testing/lecture_example.BIFXML'

# %%
bn = BayesNet.BayesNet()
bn.load_from_bifxml(path)

# %%
br = BNReasoner(bn)

br.min_degree(['Wet Grass'])
