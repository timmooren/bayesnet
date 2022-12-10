variables = ['interests', 'values', 'personality', 'attraction']
edges = [('interests', 'attraction'), ('values', 'personality'), ('values', 'attraction')]
cpts = {
    'interests': pd.DataFrame(...),  # conditional probability table for interests variable
    'values': pd.DataFrame(...),     # conditional probability table for values variable
    'personality': pd.DataFrame(...), # conditional probability table for personality variable
    'attraction': pd.DataFrame(...)   # conditional probability table for attraction variable
}

bn = Bayesian
