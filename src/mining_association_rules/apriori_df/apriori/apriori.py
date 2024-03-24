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
    def __init__(
        self, source: str, itemset_measure: str = "support", rule_measure: str = "confidence"
    ):
        super().__init__(
            runner="df", source=source, itemset_measure=itemset_measure, rule_measure=rule_measure
        )

    def support(self, itemset: set, df: pd.DataFrame) -> float:
        if df.empty is True:
            raise EmptyTransactionBaseException

        start = time.perf_counter()
        if len(itemset) > 1:
            sup = df[list(itemset)].eq(1).all(axis=1).value_counts().get(True, 0) / len(df)
        else:
            sup = df[list(itemset)[0]].value_counts().loc[1] / len(df)
        self.support_calculations_time += time.perf_counter() - start
        self.support_calculations += 1

        return sup

    def anti_support(self, antecedent: set, consequent: set, df: pd.DataFrame) -> float:
        rule_support = self.support(antecedent | consequent, df)
        antecedent_support = self.support(set(antecedent), df)
        consequent_support = self.support(set(consequent), df)

        return (consequent_support - rule_support) / (1 - antecedent_support)

    def confidence(self, antecedent: set, consequent: set, df: pd.DataFrame) -> float:
        rule_support = self.support(antecedent | consequent, df)
        antecedent_support = self.support(set(antecedent), df)

        return rule_support / antecedent_support

    def lift(self, antecedent: set, consequent: set, df: pd.DataFrame) -> float:
        rule_support = self.support(antecedent | consequent, df)
        antecedent_support = self.support(set(antecedent), df)
        consequent_support = self.support(set(consequent), df)

        return rule_support / (antecedent_support * consequent_support)

    def conviction(self, antecedent: set, consequent: set, df: pd.DataFrame) -> float:
        consequent_support = self.support(consequent, df)
        rule_confidence = self.confidence(antecedent, consequent, df)

        return (1 - consequent_support) / (1 - rule_confidence)

    def truncate_infrequent(self, df: pd.DataFrame, minsup: float) -> pd.DataFrame:
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
        frequent_itemsets = self.find_frequent_itemsets(transactions, consts.SUPPORT_THRESHOLD)
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
        df = self.truncate_infrequent(df, minsup)
        itemsets = [{element} for element in df.columns]
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
                if getattr(self, self.itemset_measure, self.support)(candidate, df) >= minsup
            ]
            self._logger.info(f"{i} elements frequent itemsets {len(itemsets)}")
            frequent_itemsets.extend(itemsets)
            if not itemsets:
                break

        return frequent_itemsets
