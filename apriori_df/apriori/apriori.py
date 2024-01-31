import logging
import time

import pandas as pd

from common.apriori.apriori import RuleGenerator
from common.utils import consts
from common.utils.exceptions import EmptyTransactionBaseException

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataFrameRuleGenerator(RuleGenerator):
    def __init__(self):
        super().__init__()

    def support(self, itemset: set, df: pd.DataFrame) -> float:
        if df.empty is True:
            raise EmptyTransactionBaseException

        start = time.perf_counter()
        supported_df = df[df[list(itemset)].eq(1).all(axis=1)]
        self.support_calculations_time += time.perf_counter() - start
        self.support_calculations += 1

        return len(supported_df) / len(df)

    def find_frequent_itemsets(
        self, df: pd.DataFrame, minsup: float = consts.SUPPORT_THRESHOLD
    ) -> list[set]:
        """
        Finds all subsets of elements_universe that meet support threshold
        :param elements_universe:
        :return:
        """
        frequent_itemsets = []
        itemsets = [
            {element} for element in df.columns if self.support({element}, df) >= minsup
        ]
        logger.info(f"{1} elements frequent itemsets {len(itemsets)}")
        frequent_itemsets.extend(itemsets)
        for i in range(2, len(df.columns)):
            candidates = self._apriori_gen(itemsets, df, minsup)
            itemsets = [
                candidate
                for candidate in candidates
                if self.support(candidate, df) >= minsup
            ]
            logger.info(f"{i} elements frequent itemsets {len(itemsets)}")
            frequent_itemsets.extend(itemsets)
            if not itemsets:
                break

        logger.info(
            f"""
                Calculating support to retrieve frequent itemsets took {self.support_calculations_time}
                and was done {self.support_calculations} times
            """
        )

        return frequent_itemsets
