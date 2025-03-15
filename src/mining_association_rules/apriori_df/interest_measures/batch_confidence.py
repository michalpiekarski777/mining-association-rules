import numpy as np
import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.apriori_df.interest_measures.batch_support import BatchSupport
from src.mining_association_rules.common.utils import consts


class BatchConfidence(Measure):
    def __init__(self):
        self._batch_support = BatchSupport()
        super().__init__()

    def calculate(
        self,
        rule_candidates,
        df: pd.DataFrame,
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> list[frozenset[str]]:
        antecedent_supports = self._batch_support.calculate(
            [c["antecedent"] for c in rule_candidates],
            df,
            minsup,
        )
        itemset_supports = self._batch_support.calculate(
            [c["itemset"] for c in rule_candidates],
            df,
            minsup,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(itemset_supports, antecedent_supports)
