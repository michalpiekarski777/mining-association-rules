import logging
import time

from src.mar.common.apriori.apriori import RuleGenerator
from src.mar.common.utils import consts
from src.mar.common.utils.exceptions import EmptyTransactionBaseError
from src.mar.common.utils.typed_dicts import AssociationRule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ListRuleGenerator(RuleGenerator):
    def __init__(self, source: str, *, verbose: bool = False):
        super().__init__(runner="list", source=source, verbose=verbose)

    def support(self, itemset: set, transactions: list[set]) -> float:
        if not transactions:
            raise EmptyTransactionBaseError
        start = time.perf_counter()
        supported = [itemset for transaction in transactions if itemset.issubset(transaction)]
        self.support_calculations_time += time.perf_counter() - start
        self.support_calculations += 1
        return len(supported) / len(transactions)

    def generate_strong_association_rules(self, transactions: list[set], elements: set) -> list[AssociationRule]:
        start = time.perf_counter()
        frequent_itemsets = self.find_frequent_itemsets(transactions, elements, consts.SUPPORT_THRESHOLD)
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
        frequent_itemsets: list[set] = []
        itemsets = [{element} for element in elements_universe if self.support({element}, transactions) >= minsup]
        if not itemsets:
            return frequent_itemsets

        self._logger.info("1 element frequent itemsets {len(itemsets)}")
        frequent_itemsets.extend(itemsets)
        for i in range(2, len(elements_universe)):
            candidates = self._apriori_gen(itemsets)
            itemsets = [candidate for candidate in candidates if self.support(candidate, transactions) >= minsup]
            log_info = {"itemset_len": len(itemsets), "element_len": i}
            self._logger.info("%(element_len)s elements frequent itemsets %(itemset_len)s", log_info)
            frequent_itemsets.extend(itemsets)
            if not itemsets:
                break

        return frequent_itemsets

    def _generate_association_rules(
        self,
        frequent_itemsets: list[frozenset[str]],
        transactions: list[set],
    ) -> list[AssociationRule]:
        """
        For each frequent itemsts find not empty subsets subLi, so the support of Li divided by support of subLi is
        greater than minconf, return all association rules in form of lists of dictionaries containing association rules
        in form of 2 sets (antecedent and consequent)
        :param frequent_itemsets:
        :param minconf:
        :return:
        """
        minsup = next(iter(self.itemset_measures.values()))
        for itemset in frequent_itemsets:
            for subset in self._generate_subset_combinations(itemset):
                itemset_measure = next(iter(self.rule_itemset_measures.keys())).calculate(itemset, transactions, minsup)
                antecedent = frozenset(subset)
                consequent = itemset - frozenset(subset)
                threshold_meeting_measures = {}
                for measure, threshold in self.rule_measures.items():
                    value = measure.calculate(antecedent, consequent, transactions)
                    if value > threshold:
                        threshold_meeting_measures[measure] = value

                if len(threshold_meeting_measures) == len(self.rule_measures):
                    self._rules.append(
                        {
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "itemset_measure": {
                                "name": type(next(iter(self.itemset_measures.keys()))).__name__,
                                "value": round(itemset_measure, 3),
                            },
                            "rule_measures": [
                                {"name": type(name).__name__, "value": value}
                                for name, value in threshold_meeting_measures.items()
                            ],
                        },
                    )
        return self._rules
