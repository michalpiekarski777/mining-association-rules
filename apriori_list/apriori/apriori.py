import logging
import time

from common.apriori.apriori import RuleGenerator
from common.utils import consts
from common.utils.exceptions import EmptyTransactionBaseException
from common.utils.typed_dicts import AssociationRule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ListRuleGenerator(RuleGenerator):
    def __init__(self):
        super().__init__()

    def support(self, itemset: set, transactions: list[set]) -> float:
        if not transactions:
            raise EmptyTransactionBaseException
        start = time.perf_counter()
        supported = [
            itemset for transaction in transactions if itemset.issubset(transaction)
        ]
        self.support_calculations_time += time.perf_counter() - start
        self.support_calculations += 1
        return len(supported) / len(transactions)

    def generate_strong_association_rules(
        self, transactions: list[set], elements: set = None
    ) -> list[AssociationRule]:
        start = time.perf_counter()
        frequent_itemsets = self.find_frequent_itemsets(
            transactions, elements, consts.SUPPORT_THRESHOLD
        )
        rules = self._generate_association_rules(frequent_itemsets, transactions)
        self.total_duration = time.perf_counter() - start

        return rules

    def find_frequent_itemsets(
        self,
        transactions: list[set],
        elements_universe: set,
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> list[set]:
        """
        Finds all subsets of elements_universe that meet support threshold
        :param elements_universe:
        :return:
        """
        frequent_itemsets = []
        itemsets = [
            {element}
            for element in elements_universe
            if self.support({element}, transactions) >= minsup
        ]
        if not itemsets:
            return frequent_itemsets

        logger.info(f"{1} elements frequent itemsets {len(itemsets)}")
        frequent_itemsets.extend(itemsets)
        for i in range(2, len(elements_universe)):
            logger.info(f"Searching for itemsets of length {i}")
            candidates = self._apriori_gen(itemsets)
            itemsets = [
                candidate
                for candidate in candidates
                if self.support(candidate, transactions) >= minsup
            ]
            logger.info(f"{i} elements frequent itemsets {len(itemsets)}")
            frequent_itemsets.extend(itemsets)
            if not itemsets:
                break

        return frequent_itemsets
