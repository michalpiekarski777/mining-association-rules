import logging
import time
from abc import ABCMeta, abstractmethod

import pandas as pd

from common.utils import consts

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RuleGenerator(metaclass=ABCMeta):
    def __init__(self):
        self.start = time.perf_counter()
        self.support_calculations = 0
        self.support_calculations_time = 0.0

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
