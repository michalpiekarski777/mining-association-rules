import logging
import time

import pandas as pd

from src.mining_association_rules.common.apriori.apriori import RuleGenerator
from src.mining_association_rules.common.utils import consts
from src.mining_association_rules.common.utils.exceptions import EmptyTransactionBaseException
from src.mining_association_rules.common.utils.typed_dicts import AssociationRule, MeasureThreshold

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataFrameRuleGenerator(RuleGenerator):
    def __init__(
        self,
        source: str,
        itemset_measures: MeasureThreshold,
        rule_measures: MeasureThreshold,
    ):
        self.supports: dict[frozenset[str], float] = {}
        self.support_counts: dict[frozenset[str], float] = {}
        super().__init__(
            runner="df",
            source=source,
            itemset_measures=itemset_measures,
            rule_measures=rule_measures,
        )

    def truncate_infrequent(self, df: pd.DataFrame, minsup: float) -> pd.DataFrame:
        if df.empty is True:
            raise EmptyTransactionBaseException

        return df.loc[:, (df.sum() / len(df)) > minsup]

    def generate_strong_association_rules(
        self, transactions: pd.DataFrame, elements: frozenset[str] | None = None
    ) -> list[AssociationRule]:
        start = time.perf_counter()
        frequent_itemsets = self.find_frequent_itemsets(transactions, consts.SUPPORT_THRESHOLD)
        rules = self._generate_association_rules(frequent_itemsets, transactions)
        self.total_duration = time.perf_counter() - start

        return rules

    def find_frequent_itemsets(
        self, df: pd.DataFrame, minsup: float = consts.SUPPORT_THRESHOLD
    ) -> list[frozenset[str]]:
        """
        Finds all subsets of elements_universe that meet support threshold
        :param df: transaction database
        :param minsup: minimum support threshold
        :return: list of subsets of elements universe
        """
        minsup = list(self.itemset_measures.values())[0]
        frequent_itemsets = []
        start = time.perf_counter()
        df = self.truncate_infrequent(df, minsup)
        itemsets = [frozenset([element]) for element in df.columns]
        self._logger.info(
            f"Finding frequent itemsets of length 1 took {time.perf_counter() - start}"
        )
        self._logger.info(f"{1} elements frequent itemsets {len(itemsets)}")
        frequent_itemsets.extend(itemsets)
        for i in range(2, len(df.columns)):
            candidates = self._apriori_gen(itemsets)
            itemsets = [
                candidate
                for candidate in candidates
                if list(self.itemset_measures.keys())[0].calculate(candidate, df, minsup) >= minsup
            ]
            self._logger.info(f"{i} elements frequent itemsets {len(itemsets)}")
            frequent_itemsets.extend(itemsets)
            if not itemsets:
                break

        return frequent_itemsets
