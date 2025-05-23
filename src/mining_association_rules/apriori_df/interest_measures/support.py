import time

import numpy as np
import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.common.utils import consts
from src.mining_association_rules.common.utils.exceptions import EmptyTransactionBaseError


class Support(Measure):
    def __init__(self):
        super().__init__()

    def calculate(
        self,
        itemsets: list[frozenset[str]],
        df: pd.DataFrame,
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> np.ndarray:
        supports = np.empty(len(itemsets), dtype=float)

        for i, itemset in enumerate(itemsets):
            if itemset in self.history:
                supports[i] = self.history[itemset]

            if df.empty is True:
                raise EmptyTransactionBaseError

            start = time.perf_counter()
            if len(itemset) > 1:
                sup = df[list(itemset)].eq(1).all(axis=1).value_counts().get(True, 0) / len(df)
            else:
                sup = df[next(iter(itemset))].value_counts().loc[1] / len(df)

            self.calculations_time += time.perf_counter() - start
            self.calculations_count += 1
            supports[i] = sup

            if sup >= minsup:
                self.history[itemset] = sup

        return supports
