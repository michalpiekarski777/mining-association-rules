import logging
import time
from abc import ABCMeta, abstractmethod
from itertools import chain, combinations

import pandas as pd

from common.utils import consts
from common.utils.typed_dicts import AssociationRule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RuleGenerator(metaclass=ABCMeta):
    def __init__(self):
        self.start = time.perf_counter()
        self.support_calculations = 0
        self.support_calculations_time = 0.0
        self.total_duration = 0.0

    def generate_strong_association_rules(
        self, transactions: pd.DataFrame | list[set], elements: set | None = None
    ) -> list[AssociationRule]:
        raise NotImplementedError

    @abstractmethod
    def find_frequent_itemsets(
        self,
        transactions: pd.DataFrame | list[set],
        minsup: float = consts.SUPPORT_THRESHOLD,
    ):
        raise NotImplementedError

    @abstractmethod
    def support(self, itemset: set, transactions: pd.DataFrame | list[set]) -> float:
        raise NotImplementedError

    def _apriori_gen(
        self,
        itemsets: list[set],
        transactions: pd.DataFrame | list[set],
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> list[set]:
        """
        Returns list of k-element candidate itemsets
        :param elements:
        :return:
        """
        start = time.perf_counter()
        candidates = []
        set_length = len(itemsets[0])
        sorted_itemsets = [sorted(itemset) for itemset in itemsets]

        for index, (itemset, sorted_itemset) in enumerate(
            zip(itemsets, sorted_itemsets)
        ):
            for j in range(index + 1, len(itemsets)):
                if (
                    sorted_itemset[: set_length - 1]
                    == sorted_itemsets[j][: set_length - 1]
                ):
                    candidates.append(itemset.union(itemsets[j]))
        end = time.perf_counter()
        logger.info(f"Generating candidates of length 2 took {end - start}")
        return [
            candidate
            for candidate in candidates
            if self.support(candidate, transactions) >= minsup
        ]

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
        rules: list[AssociationRule] = []
        for itemset in frequent_itemsets:
            for subset in self._generate_subset_combinations(itemset):
                if (
                    self.support(itemset, transactions)
                    / self.support(set(subset), transactions)
                    >= minconf
                ):
                    rules.append(
                        dict(
                            antecedent=set(subset),
                            consequent=itemset - set(subset),
                        )
                    )
        return rules

    def _generate_subset_combinations(self, elements: set) -> list[tuple]:
        """
        :param elements: set of elements
        :return: list of not empty subsets of set elements excluding set of length len(elements)
        """
        return list(
            chain.from_iterable(
                combinations(elements, size) for size in range(1, len(elements))
            )
        )
