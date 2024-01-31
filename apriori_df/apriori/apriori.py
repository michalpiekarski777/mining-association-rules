import logging
import time

import pandas as pd

from common.apriori.apriori import RuleGenerator
from common.utils import consts
from common.utils.exceptions import EmptyTransactionBaseException
from common.utils.typed_dicts import AssociationRule

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

    def generate_strong_association_rules(
        self, transactions: pd.DataFrame, elements: set | None = None
    ) -> list[AssociationRule]:
        start = time.perf_counter()
        frequent_itemsets = self.find_frequent_itemsets(
            transactions, consts.SUPPORT_THRESHOLD
        )
        rules = self._generate_association_rules(frequent_itemsets, transactions)
        self.total_duration = time.perf_counter() - start

        return rules

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

        return frequent_itemsets
