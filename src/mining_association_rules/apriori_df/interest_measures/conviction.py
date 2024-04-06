import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures import Confidence, Support
from src.mining_association_rules.apriori_df.interest_measures.base import Measure


class Conviction(Measure):
    def __init__(self):
        super().__init__()

    def calculate(
        self,
        antecedent: frozenset[str],
        consequent: frozenset[str],
        df: pd.DataFrame,
        support: Support,
        confidence: Confidence,
    ) -> float:
        consequent_support = support.calculate(consequent, df)
        rule_confidence = confidence.calculate(antecedent, consequent, df, support)

        return (1 - consequent_support) / (1 - rule_confidence)
