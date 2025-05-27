import numpy as np
import pandas as pd

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.apriori_df.interest_measures.support_count import SupportCount
from src.mar.common.utils import consts
from src.mar.common.utils.typed_dicts import RuleCandidate


class Lift(Measure):
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

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = len(df) * itemset_supports
            denominator = antecedent_supports * consequent_supports

            return np.divide(numerator, denominator)
