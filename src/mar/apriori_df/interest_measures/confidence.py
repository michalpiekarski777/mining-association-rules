import numpy as np
import pandas as pd

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.apriori_df.interest_measures.support import Support
from src.mar.common.utils import consts
from src.mar.common.utils.typed_dicts import RuleCandidate


class Confidence(Measure):
    def __init__(self):
        self._support = Support()
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
        itemset_supports = self._support.calculate(
            [c["itemset"] for c in rule_candidates],
            df,
            minsup,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(itemset_supports, antecedent_supports)
