import time

import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.common.utils import consts
from src.mining_association_rules.common.utils.exceptions import EmptyTransactionBaseException


class Support(Measure):
    def __init__(self):
        super().__init__()

    def calculate(
        self, itemset: frozenset[str], df: pd.DataFrame, minsup: float = consts.SUPPORT_THRESHOLD
    ) -> float:
        if itemset in self.history:
            return self.history[itemset]

        if df.empty is True:
            raise EmptyTransactionBaseException

        start = time.perf_counter()
        if len(itemset) > 1:
            sup = df[list(itemset)].eq(1).all(axis=1).value_counts().get(True, 0) / len(df)
        else:
            sup = df[list(itemset)[0]].value_counts().loc[1] / len(df)
        self.calculations_time += time.perf_counter() - start
        self.calculations_count += 1

        if sup >= minsup:
            self.history[itemset] = sup

        return sup
