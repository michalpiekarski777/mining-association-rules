import time

import pandas as pd
from scipy.stats import hypergeom

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.apriori_df.interest_measures.support_count import SupportCount


class HyperLift(Measure):
    def __init__(self):
        self._support_count = SupportCount()
        super().__init__()

    def calculate(
        self,
        antecedent: frozenset[str],
        consequent: frozenset[str],
        df: pd.DataFrame,
        quantile=0.99,
    ) -> float:
        start = time.perf_counter()
        rule_support_count = self._support_count.calculate(antecedent | consequent, df)
        antecedent_support_count = self._support_count.calculate(antecedent, df)
        consequent_support_count = self._support_count.calculate(consequent, df)
        hyperlift = rule_support_count / hypergeom.ppf(
            quantile,
            len(df),
            antecedent_support_count,
            consequent_support_count,
        )
        self.calculations_time += time.perf_counter() - start
        self.calculations_count += 1

        return hyperlift
