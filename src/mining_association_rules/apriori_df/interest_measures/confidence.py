import time

import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.apriori_df.interest_measures.support import Support


class Confidence(Measure):
    def __init__(self):
        self._support = Support()
        super().__init__()

    def calculate(self, antecedent: frozenset[str], consequent: frozenset[str], df: pd.DataFrame) -> float:
        start = time.perf_counter()
        rule_support = self._support.calculate(antecedent | consequent, df)
        antecedent_support = self._support.calculate(antecedent, df)
        confidence = rule_support / antecedent_support
        self.calculations_time += time.perf_counter() - start
        self.calculations_count += 1

        return confidence
