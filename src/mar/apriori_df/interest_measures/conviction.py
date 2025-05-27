import numpy as np
import pandas as pd

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.apriori_df.interest_measures.confidence import Confidence
from src.mar.apriori_df.interest_measures.support import Support
from src.mar.common.utils import consts
from src.mar.common.utils.typed_dicts import RuleCandidate


class Conviction(Measure):
    def __init__(self):
        self._support = Support()
        self._confidence = Confidence()
        super().__init__()

    def calculate(
        self,
        rule_candidates: list[RuleCandidate],
        df: pd.DataFrame,
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> np.ndarray:
        consequent_supports = self._support.calculate(
            [c["consequent"] for c in rule_candidates],
            df,
            minsup,
        )
        rule_confidences = self._confidence.calculate(rule_candidates, df, minsup)

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = 1 - consequent_supports
            denominator = 1 - rule_confidences
            return np.divide(numerator, denominator)
