from BayesNet import BayesNet
import pandas as pd


def create_lecture_example() -> BayesNet:
    variables = [
        'Visit to Asia',
        'Tuberculosis',
        'Smoker',
        'Lung Cancer',
        'Tuberculosis or Cancer',
        'Bronchitis',
        'Positive X-ray',
        'Dyspnoea'
        ]

    edges = [
        ('Visit to Asia', 'Tuberculosis'),
        ('Tuberculosis', 'Tuberculosis or Cancer'),
        ('Tuberculosis or Cancer', 'Positive X-ray'),
        ('Tuberculosis or Cancer', 'Dyspnoea'),
        ('Smoker', 'Lung Cancer'),
        ('Smoker', 'Bronchitis'),
        ('Lung Cancer', 'Tuberculosis or Cancer'),
        ('Bronchitis', 'Dyspnoea')
        ]

    cpts = {var: pd.DataFrame() for var in variables}
    bn = BayesNet()
    bn.create_bn(variables=variables, edges=edges, cpts=cpts)

    return bn