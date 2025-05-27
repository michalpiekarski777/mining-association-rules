import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.apriori_df.interest_measures.support_count import SupportCount
from src.mar.common.utils import consts
from src.mar.common.utils.typed_dicts import RuleCandidate


class HyperConfidence(Measure):
    def __init__(self):
        self._support = SupportCount()
        super().__init__()

    def calculate(
        self,
        rule_candidates: list[RuleCandidate],
        df: pd.DataFrame,
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> np.ndarray:
        antecedent_supports = self._support.calculate(
            [c["antecedent"] for c in rule_candidates],
            df,
            minsup,
        )
        consequent_supports = self._support.calculate(
            [c["consequent"] for c in rule_candidates],
            df,
            minsup,
        )
        itemset_supports = self._support.calculate(
            [c["itemset"] for c in rule_candidates],
            df,
            minsup,
        )
        results = np.empty(len(rule_candidates), dtype=float)

        for i in range(len(rule_candidates)):
            results[i] = hypergeom.cdf(itemset_supports[i], len(df), antecedent_supports[i], consequent_supports[i])

        return results
