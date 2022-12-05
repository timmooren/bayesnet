# %%
import BayesNet
import networkx as nx
from BNReasoner import BNReasoner
import pandas as pd
from example_lecture import create_lecture_example

# %%
path = 'testing/lecture_example2.BIFXML'

# %%
bn = create_lecture_example()
print(bn)

# %%
br = BNReasoner(bn)

# %%
# br.bn.draw_structure()



# %%
# should be separated
print(br.d_separation(['Visit to Asia', 'Smoker'], ['Dyspnoea', 'Positive X-ray'], ['Tuberculosis or Cancer', 'Bronchitis']))
# should not be separated
print(br.d_separation(['Positive X-ray'], ['Smoker'], ['Lung Cancer', 'Dyspnoea']))

