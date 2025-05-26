import pandas as pd

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.apriori_df.interest_measures.support_count import SupportCount


class Lift(Measure):
    def __init__(self):
        self._support_count = SupportCount()
        super().__init__()

    def calculate(self, antecedent: frozenset[str], consequent: frozenset[str], df: pd.DataFrame) -> float:
        rule_support_count = self._support_count.calculate(antecedent | consequent, df)
        antecedent_support_count = self._support_count.calculate(antecedent, df)
        consequent_support_count = self._support_count.calculate(consequent, df)

        return len(df) * rule_support_count / (antecedent_support_count * consequent_support_count)
