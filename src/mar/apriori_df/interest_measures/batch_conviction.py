import numpy as np
import pandas as pd

from src.mar.apriori_df.interest_measures import BatchConfidence
from src.mar.apriori_df.interest_measures import BatchSupport
from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.common.utils import consts
from src.mar.common.utils.typed_dicts import RuleCandidate


class BatchConviction(Measure):
    def __init__(self):
        self._support = BatchSupport()
        self._confidence = BatchConfidence()
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
