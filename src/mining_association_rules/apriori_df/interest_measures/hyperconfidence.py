import time

import pandas as pd
from scipy.stats import hypergeom

from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.apriori_df.interest_measures.support_count import SupportCount


class HyperConfidence(Measure):
    def __init__(self):
        self._support_count = SupportCount()
        super().__init__()

    def calculate(self, antecedent: frozenset[str], consequent: frozenset[str], df: pd.DataFrame) -> float:
        start = time.perf_counter()
        rule_support_count = self._support_count.calculate(antecedent | consequent, df)
        antecedent_support_count = self._support_count.calculate(antecedent, df)
        consequent_support_count = self._support_count.calculate(consequent, df)
        hyperconfidence = hypergeom.cdf(rule_support_count, len(df), antecedent_support_count, consequent_support_count)
        self.calculations_time += time.perf_counter() - start
        self.calculations_count += 1

        return hyperconfidence
