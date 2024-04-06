import logging
import time

import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures import Confidence, Support
from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.common.apriori.apriori import RuleGenerator
from src.mining_association_rules.common.utils import consts
from src.mining_association_rules.common.utils.exceptions import EmptyTransactionBaseException
from src.mining_association_rules.common.utils.typed_dicts import AssociationRule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataFrameRuleGenerator(RuleGenerator):
    itemset_measures = ["support"]
    rule_measures = [
        "confidence",
        "anti_support",
        "lift",
        "conviction",
        "rule_interest_function",
        "gain_function",
        "dependency_factor",
    ]

    def __init__(self, source: str, itemset_measure: type[Measure], rule_measure: type[Measure]):
        self.supports: dict[frozenset[str], float] = {}
        self.support_counts: dict[frozenset[str], float] = {}
        super().__init__(
            runner="df", source=source, itemset_measure=itemset_measure, rule_measure=rule_measure
        )
        if itemset_measure not in self.itemset_measures:
            self._logger.warning(
                f"Itemset measure {itemset_measure} not implemented, using support"
            )
            self.itemset_measure = Support()

        if rule_measure not in self.rule_measures:
            self._logger.warning(f"Rule measure {rule_measure} not implemented, using confidence")
            self.rule_measure = Confidence()

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
                if self.itemset_measure.calculate(candidate, df) >= minsup
            ]
            self._logger.info(f"{i} elements frequent itemsets {len(itemsets)}")
            frequent_itemsets.extend(itemsets)
            if not itemsets:
                break

        return frequent_itemsets
