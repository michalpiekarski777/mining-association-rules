import logging
import time
from typing import cast

import pandas as pd

from src.mar.apriori_df.interest_measures.base import Measure
from src.mar.common.apriori.apriori import RuleGenerator
from src.mar.common.utils.exceptions import EmptyTransactionBaseError
from src.mar.common.utils.typed_dicts import AssociationRule
from src.mar.common.utils.typed_dicts import MeasureTypedDict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataFrameRuleGenerator(RuleGenerator):
    def __init__(
        self,
        itemset_measures: dict[type[Measure], float],
        rule_measures: dict[type[Measure], float],
        *,
        verbose: bool = False,
    ):
        self.supports: dict[frozenset[str], float] = {}
        self.support_counts: dict[frozenset[str], float] = {}
        super().__init__(
            runner="df",
            itemset_measures=itemset_measures,
            rule_measures=rule_measures,
            verbose=verbose,
        )

    def truncate_infrequent(self, df: pd.DataFrame, minsup: float) -> pd.DataFrame:
        """According to the Apriori rule, we can remove columns that represent infrequent itemsets of length 1"""
        if df.empty is True:
            raise EmptyTransactionBaseError

        return df.loc[:, (df.sum() / len(df)) >= minsup]

    def generate_strong_association_rules(
        self,
        transactions: pd.DataFrame,
        elements: frozenset[str] | None = None,
    ) -> list[AssociationRule]:
        start = time.perf_counter()
        frequent_itemsets = self.find_frequent_itemsets(transactions)
        t2 = time.perf_counter()
        self._logger.info(
            "Finding %(n)s frequent itemsets took %(time)s",
            {"time": t2 - start, "n": len(frequent_itemsets)},
        )
        rules = self._generate_association_rules(frequent_itemsets, transactions)
        t3 = time.perf_counter()
        self._logger.info(
            "Generating %(n)s association rules took %(time)s",
            {"time": t3 - t2, "n": len(rules)},
        )
        self.total_duration = time.perf_counter() - start

        return rules

    def find_frequent_itemsets(self, df: pd.DataFrame) -> list[frozenset[str]]:
        """
        Finds all subsets of elements universe that meet itemset measures thresholds
        :param df: transaction database
        :return: list of subsets of elements universe that meet itemset measures thresholds
        """
        minsup = next(iter(self.itemset_measures.values()))
        frequent_itemsets = []
        df = self.truncate_infrequent(df, minsup)
        itemsets = [frozenset([element]) for element in df.columns]
        frequent_itemsets.extend(itemsets)
        for _ in range(2, len(df.columns)):
            candidates = self._apriori_gen(itemsets)
            supports = next(iter(self.itemset_measures)).calculate(candidates, df, minsup)
            itemsets = [c for c, s in zip(candidates, supports, strict=False) if s >= minsup]
            frequent_itemsets.extend(itemsets)
            if not itemsets:
                break

        return frequent_itemsets

    def _generate_association_rules(
        self,
        frequent_itemsets: list[frozenset[str]],
        df: pd.DataFrame,
    ) -> list[AssociationRule]:
        """
        :param frequent_itemsets: list of sets containing frequent itemsets
        :param df: transaction database
        :return: association rules with metrics data
        """
        rule_candidates = [
            {
                "antecedent": frozenset(subset),
                "consequent": itemset - frozenset(subset),
                "itemset": itemset,
            }
            for itemset in frequent_itemsets
            for subset in self._generate_subset_combinations(itemset)
        ]
        minsup = next(iter(self.itemset_measures.values()))
        supports = next(iter(self.itemset_measures)).calculate([c["itemset"] for c in rule_candidates], df, minsup)
        rule_measures_list = [measure.calculate(rule_candidates, df, minsup) for measure in self.rule_measures]
        for index, (support, candidate) in enumerate(zip(supports, rule_candidates, strict=False)):
            threshold_meeting_measures = {}
            for i, (measure, threshold) in enumerate(self.rule_measures.items()):
                value = rule_measures_list[i][index]
                if value >= threshold:
                    threshold_meeting_measures[measure] = value

            if len(threshold_meeting_measures) == len(self.rule_measures):
                itemset_measure: MeasureTypedDict = {
                    "name": type(next(iter(self.itemset_measures.keys()))).__name__,
                    "value": round(support, 3),
                }
                rule_measures = [
                    cast(MeasureTypedDict, {"name": type(name).__name__, "value": round(value, 3)})
                    for name, value in threshold_meeting_measures.items()
                ]
                self._rules.append(
                    {
                        "antecedent": candidate["antecedent"],
                        "consequent": candidate["consequent"],
                        "itemset_measure": itemset_measure,
                        "rule_measures": rule_measures,
                    },
                )

        return self._rules
