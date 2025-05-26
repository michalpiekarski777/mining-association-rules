import pandas as pd

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.common.utils.exceptions import EmptyTransactionBaseError


class SupportCount(Measure):
    def __init__(self):
        super().__init__()

    def calculate(self, itemset: frozenset[str], df: pd.DataFrame) -> float:
        if itemset in self.history:
            return self.history[itemset]

        if df.empty is True:
            raise EmptyTransactionBaseError

        if len(itemset) > 1:
            support_count = df[list(itemset)].eq(1).all(axis=1).value_counts().get(True, 0)
        else:
            support_count = df[next(iter(itemset))].value_counts().loc[1]

        self.history[itemset] = support_count

        return support_count
