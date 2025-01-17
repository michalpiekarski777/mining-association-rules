import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures.base import Measure


class AntiSupport(Measure):
    def __init__(self):
        super().__init__()

    def calculate(self, antecedent: frozenset[str], consequent: frozenset[str], df: pd.DataFrame) -> float:
        count_antecedent_not_consequent = df[
            (df[list(antecedent)].sum(axis=1) == len(antecedent)) & (df[list(consequent)].sum(axis=1) < len(consequent))
        ]

        return len(count_antecedent_not_consequent) / len(df)
