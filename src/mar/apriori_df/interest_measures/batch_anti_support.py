import numpy as np
import pandas as pd

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.common.utils.typed_dicts import RuleCandidate


class BatchAntiSupport(Measure):
    def __init__(self):
        super().__init__()

    def calculate(self, rule_candidates: RuleCandidate, df: pd.DataFrame, *args, **kwargs) -> float:
        anti_supports = np.empty(len(rule_candidates), dtype=float)
        for i, rule_candidate in enumerate(rule_candidates):
            antecedent = rule_candidate["antecedent"]
            consequent = rule_candidate["consequent"]
            anti_supports[i] = len(
                df[
                    (df[list(antecedent)].sum(axis=1) == len(antecedent))
                    & (df[list(consequent)].sum(axis=1) < len(consequent))
                ],
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(anti_supports, len(df))
