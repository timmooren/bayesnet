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
for x in br.bn.get_all_variables():
    cpt = br.bn.get_cpt(x)
    print(cpt)

breakpoint()

# %%
br.bn.draw_structure()

# %%
br.prune(query=['Wet Grass?'], evidence=pd.Series(
    {'Rain?': False, 'Winter?': True}))

# %%
br.bn.draw_structure()

# %%
# a
print(br.bn.get_cpt('Winter?'))

# %%
# b
print(br.bn.get_cpt('Sprinkler?'))


# %%
# c
print(br.bn.get_cpt('Rain?'))

# %%
# d
print(br.bn.get_cpt('Wet Grass?'))


