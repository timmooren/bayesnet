# %%
import BayesNet
import networkx as nx
from BNReasoner import BNReasoner
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%
path = 'testing/lecture_example2.BIFXML'

# %%
bn = BayesNet.BayesNet()
bn.load_from_bifxml(path)

# %%
br = BNReasoner(bn)

br.variable_elimination(["C"], [])