import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.apriori_df.interest_measures.confidence import Confidence
from src.mining_association_rules.apriori_df.interest_measures.support import Support


class Conviction(Measure):
    def __init__(self):
        self._support = Support()
        self._confidence = Confidence()
        super().__init__()

    def calculate(
        self, antecedent: frozenset[str], consequent: frozenset[str], df: pd.DataFrame
    ) -> float:
        consequent_support = self._support.calculate(consequent, df)
        rule_confidence = self._confidence.calculate(antecedent, consequent, df)

        return (1 - consequent_support) / (1 - rule_confidence)
