import time
from abc import ABCMeta, abstractmethod
from itertools import chain, combinations

import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures import *  # noqa: F401, F403
from src.mining_association_rules.common.utils.loggers import Logger
from src.mining_association_rules.common.utils.typed_dicts import AssociationRule, MeasureThreshold


class RuleGenerator(metaclass=ABCMeta):
    total_duration: float = 0.0

    def __init__(
        self,
        runner: str,
        itemset_measures: MeasureThreshold,
        rule_measures: MeasureThreshold,
        source: str = "",
        logger_class: type[Logger] = Logger,
    ):
        self.start = time.perf_counter()
        self.itemset_measures = {measure(): threshold for measure, threshold in itemset_measures.items()}
        self.rule_measures = {measure(): threshold for measure, threshold in rule_measures.items()}
        self._runner = runner
        self._source = source
        self._rules: list[AssociationRule] = []
        self._logger = logger_class(name="Rules")
        self._logger.info(f"Generating association rules for source {self._source}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._logger.info(f"Rules generated using {self._runner} database in {self.total_duration} seconds")
        for itemset_measure in self.itemset_measures:
            itemset_measure_name = type(itemset_measure).__name__
            itemset_measure_count = itemset_measure.calculations_count
            itemset_measure_time = itemset_measure.calculations_time
            log = f"Calculating {itemset_measure_name} was done {itemset_measure_count} and took {itemset_measure_time}"
            self._logger.info(log)

        for rule_measure in self.rule_measures:
            rule_measure_name = type(rule_measure).__name__
            rule_measure_count = rule_measure.calculations_count
            rule_measure_time = rule_measure.calculations_time
            log = f"Calculating {rule_measure_name} was done {rule_measure_count} and took {rule_measure_time}"
            self._logger.info(log)
        self._logger.info(f"Found {len(self._rules)} rules")

        for rule in self._rules:
            rule_members = f"Association rule {set(rule['antecedent'])} -> {set(rule['consequent'])},"
            itemset_measure_log = f"{rule['itemset_measure']['name']}: {rule['itemset_measure']['value']}"
            rule_measure_log = ""
            for rule_measure in rule["rule_measures"]:
                rule_measure_log += f"{rule_measure['name']}: {round(rule_measure['value'], 3)}, "

            metrics = itemset_measure_log + "," + rule_measure_log
            self._logger.info(rule_members + metrics)

    @abstractmethod
    def generate_strong_association_rules(self, *args, **kwargs) -> list[AssociationRule]:
        raise NotImplementedError

    def _apriori_gen(self, itemsets: list[frozenset[str]]) -> list[frozenset[str]]:
        """
        Returns list of k-element candidate itemsets
        :param elements:
        :return:
        """
        candidates = []
        set_length = len(itemsets[0])
        sorted_itemsets = [sorted(itemset) for itemset in itemsets]

        for index, (itemset, sorted_itemset) in enumerate(zip(itemsets, sorted_itemsets)):
            for j in range(index + 1, len(itemsets)):
                if sorted_itemset[: set_length - 1] == sorted_itemsets[j][: set_length - 1]:
                    candidates.append(itemset.union(itemsets[j]))
        return candidates

    def _generate_association_rules(
        self,
        frequent_itemsets: list[frozenset[str]],
        transactions: pd.DataFrame | list[set],
    ) -> list[AssociationRule]:
        """
        For each frequent itemsts find not empty subsets subLi, so the support of Li divided by support of subLi is
        greater than minconf, return all association rules in form of lists of dictionaries containing association rules
        in form of 2 sets (antecedent and consequent)
        :param frequent_itemsets:
        :param minconf:
        :return:
        """
        minsup = list(self.itemset_measures.values())[0]
        for itemset in frequent_itemsets:
            for subset in self._generate_subset_combinations(itemset):
                itemset_measure = list(self.itemset_measures.keys())[0].calculate(itemset, transactions, minsup)
                antecedent = frozenset(subset)
                consequent = itemset - frozenset(subset)
                threshold_meeting_measures = {}
                for measure, threshold in self.rule_measures.items():
                    value = measure.calculate(antecedent, consequent, transactions)
                    if value > threshold:
                        threshold_meeting_measures[measure] = value

                if len(threshold_meeting_measures) == len(self.rule_measures):
                    self._rules.append(
                        dict(
                            antecedent=antecedent,
                            consequent=consequent,
                            itemset_measure=dict(
                                name=type(list(self.itemset_measures.keys())[0]).__name__,
                                value=round(itemset_measure, 3),
                            ),
                            rule_measures=[
                                dict(name=type(name).__name__, value=value)
                                for name, value in threshold_meeting_measures.items()
                            ],
                        )
                    )
        return self._rules

    def _generate_subset_combinations(self, elements: frozenset[str]) -> list[tuple]:
        """
        :param elements: set of elements
        :return: list of not empty subsets of set elements excluding set of length len(elements)
        """
        return list(chain.from_iterable(combinations(elements, size) for size in range(1, len(elements))))
