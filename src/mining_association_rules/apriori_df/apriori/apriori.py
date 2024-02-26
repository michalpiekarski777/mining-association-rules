import logging
import time

import pandas as pd

from src.mining_association_rules.common.apriori.apriori import RuleGenerator
from src.mining_association_rules.common.utils import consts
from src.mining_association_rules.common.utils.exceptions import EmptyTransactionBaseException
from src.mining_association_rules.common.utils.typed_dicts import AssociationRule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataFrameRuleGenerator(RuleGenerator):
    def __init__(self):
        super().__init__()

    def support(self, itemset: set, df: pd.DataFrame) -> float:
        if df.empty is True:
            raise EmptyTransactionBaseException

        start = time.perf_counter()
        if len(itemset) > 1:
            supported_transactions_count = len(df[df[list(itemset)].eq(1).all(axis=1)])
        else:
            supported_transactions_count = df[list(itemset)[0]].value_counts().iloc[-1]
        self.support_calculations_time += time.perf_counter() - start
        self.support_calculations += 1

        return supported_transactions_count / len(df)

    def alternative_support(self, df: pd.DataFrame, minsup: float) -> pd.DataFrame:
        if df.empty is True:
            raise EmptyTransactionBaseException
        start = time.perf_counter()
        ret = df.loc[:, (df.sum() / len(df)) > minsup]
        self.support_calculations_time += time.perf_counter() - start
        self.support_calculations += 1
        return ret

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
        :param df: transaction database
        :param minsup: minimum support threshold
        :return: list of subsets of elements universe
        """
        frequent_itemsets = []
        start = time.perf_counter()
        # itemsets = [
        #     {element} for element in df.columns if self.support({element}, df) >= minsup
        # ]
        # df = df[[list(element)[0] for element in itemsets]]
        df = self.alternative_support(df, minsup)
        itemsets = [{element} for element in df.columns]
        # df = df[[list(element)[0] for element in itemsets]]
        logger.info(
            f"Finding frequent itemsets of length 1 took {time.perf_counter() - start}"
        )
        logger.info(f"{1} elements frequent itemsets {len(itemsets)}")
        frequent_itemsets.extend(itemsets)
        for i in range(2, len(df.columns)):
            candidates = self._apriori_gen(itemsets)
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
