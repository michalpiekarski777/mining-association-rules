import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.apriori_df.interest_measures.support_count import SupportCount
from src.mar.common.utils import consts
from src.mar.common.utils.typed_dicts import RuleCandidate


class HyperLift(Measure):
    def __init__(self):
        self._support = SupportCount()
        super().__init__()

    def calculate(
        self,
        rule_candidates: list[RuleCandidate],
        df: pd.DataFrame,
        minsup: float = consts.SUPPORT_THRESHOLD,
        quantile=0.99,
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
        ppfs = np.empty(len(rule_candidates), dtype=float)

        for i in range(len(rule_candidates)):
            ppfs[i] = hypergeom.cdf(quantile, len(df), antecedent_supports[i], consequent_supports[i])

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(itemset_supports, ppfs)
