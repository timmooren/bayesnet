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

br = BNReasoner(bn)

cpt = br.bn.get_cpt('Wet Grass?')

# examples from slide Maximising-Out – Introduction
# marg = br.marginalization(cpt, 'Wet Grass?')
# max_out = br.max_out(cpt, 'Wet Grass?')

# factor table from slide Multiplication of Factors
data = [
    [True, True, 0.448],
    [True, False, 0.192],
    [False, True, 0.112],
    [False, False, 0.248]
]


df = pd.DataFrame(data, columns=['Wet Grass?', 'nothing', 'p']) #A=winter, B=sprinkler, C=rain, D=wet grass, E =slippery road

mult = br.factor_multiplication(df, cpt)
print(mult)
