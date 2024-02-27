import logging
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from itertools import chain, combinations
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.common.utils import consts
from src.mining_association_rules.common.utils.typed_dicts import AssociationRule


class RuleGenerator(metaclass=ABCMeta):
    support_calculations: int = 0
    support_calculations_time: float = 0.0
    total_duration: float = 0.0

    def _initialize_logger(self) -> logging.Logger:
        file_name = datetime.now().strftime("%Y%m%d_%H%M_%s")
        handler = logging.FileHandler(Path(ROOT_DIR) / "logs" / file_name)
        logger = logging.getLogger(__name__)
        logger.addHandler(handler)
        logger.info(f"Generating association rules for source {self._source}")
        return logger

    def __init__(self, runner: str, source: str):
        self.start = time.perf_counter()
        self._runner = runner
        self._source = source
        self._rules: list[AssociationRule] = []
        self._logger = self._initialize_logger()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._logger.info(
            f"Rules generated using {self._runner} database in {self.total_duration} seconds"
        )
        self._logger.info(
            f"Calculating support was done {self.support_calculations} and took {self.support_calculations_time}"
        )
        self._logger.info(f"Found {len(self._rules)} rules")
        for rule in self._rules:
            rule_members = f"Association rule {rule['antecedent']} -> {rule['consequent']},"
            metrics = f" support: {rule['support']}, confidence: {rule['confidence']}"
            self._logger.info(rule_members + metrics)

    @abstractmethod
    def generate_strong_association_rules(self, *args, **kwargs) -> list[AssociationRule]:
        raise NotImplementedError

    @abstractmethod
    def support(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def _apriori_gen(
        self,
        itemsets: list[set],
    ) -> list[set]:
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
        frequent_itemsets: list[set],
        transactions: pd.DataFrame | list[set],
        minconf: float = consts.CONFIDENCE_THRESHOLD,
    ) -> list[AssociationRule]:
        """
        For each frequent itemsts find not empty subsets subLi, so the support of Li divided by support of subLi is
        greater than minconf, return all association rules in form of lists of dictionaries containing association rules
        in form of 2 sets (antecedent and consequent)
        :param frequent_itemsets:
        :param minconf:
        :return:
        """
        for itemset in frequent_itemsets:
            for subset in self._generate_subset_combinations(itemset):
                rule_support = self.support(itemset, transactions)
                subset_support = self.support(set(subset), transactions)
                rule_confidence = rule_support / subset_support

                if rule_confidence >= minconf:
                    self._rules.append(
                        dict(
                            antecedent=set(subset),
                            consequent=itemset - set(subset),
                            support=round(rule_support, 3),
                            confidence=round(rule_confidence, 3),
                        )
                    )
        return self._rules

    def _generate_subset_combinations(self, elements: set) -> list[tuple]:
        """
        :param elements: set of elements
        :return: list of not empty subsets of set elements excluding set of length len(elements)
        """
        return list(
            chain.from_iterable(combinations(elements, size) for size in range(1, len(elements)))
        )
