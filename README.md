# bayesnet

## representations:
- variable: str
- instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
- cpt: pd.Dataframe
- factor: pd.Dataframe
- edges: List[Tuple[str, str]]