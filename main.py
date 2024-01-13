import logging
from itertools import chain, combinations
from typing import TypedDict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SUPPORT_THRESHOLD = 0.1
CONFIDENCE_THRESHOLD = 0.1


class EmptyTransactionBaseException(Exception):
    pass


class DifferentLengthItemsetException(Exception):
    pass


class AssociationRule(TypedDict):
    antecedent: set
    consequent: set


def apriori_gen(
    itemsets: list[set], transactions: list[set], minsup: float = 0.4
) -> list[set]:
    """
    Returns list of k-element candidate itemsets
    :param elements:
    :return:
    """
    if not itemsets or any(len(itemset) != len(itemsets[0]) for itemset in itemsets):
        raise DifferentLengthItemsetException

    candidates = []
    set_length = len(itemsets[0])
    sorted_itemsets = [sorted(itemset) for itemset in itemsets]

    for index, (itemset, sorted_itemset) in enumerate(zip(itemsets, sorted_itemsets)):
        for j in range(index + 1, len(itemsets)):
            if sorted_itemset[: set_length - 1] == sorted_itemsets[j][: set_length - 1]:
                candidates.append(itemset.union(itemsets[j]))

    return [
        candidate
        for candidate in candidates
        if support(candidate, transactions) >= minsup
    ]


def support(itemset: set, transactions: list[set]) -> float:
    if not transactions:
        raise EmptyTransactionBaseException

    supported = [
        itemset for transaction in transactions if itemset.issubset(transaction)
    ]
    return len(supported) / len(transactions)


def find_frequent_itemsets(
    elements_universe: set, transactions: list[set], minsup: float = 0.4
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
        if support({element}, transactions) >= minsup
    ]
    logger.info(f"{1} elements frequent itemsets {itemsets}")
    frequent_itemsets.extend(itemsets)
    for i in range(2, len(elements_universe)):
        candidates = apriori_gen(itemsets, transactions, minsup)
        itemsets = [
            candidate
            for candidate in candidates
            if support(candidate, transactions) >= minsup
        ]
        logger.info(f"{i} elements frequent itemsets {itemsets}")
        frequent_itemsets.extend(itemsets)
        if len(itemsets) == len(frequent_itemsets):
            break
    return frequent_itemsets


def generate_subset_combinations(elements: set) -> list[tuple]:
    """
    :param elements: set of elements
    :return: list of not empty subsets of set elements excluding set of length len(elements)
    """
    return list(
        chain.from_iterable(
            combinations(elements, size) for size in range(1, len(elements))
        )
    )


def generate_association_rules(
    frequent_itemsets: list[set], transactions: list[set], minconf: float = 0.5
) -> list[AssociationRule]:
    """
    For each frequent itemsts find not empty subsets subLi, so the support of Li divided by support of subLi is greater
    than minconf, return all association rules in form of lists of dictionaries containing association rules
    in form of 2 sets (antecedent and consequent)
    :param frequent_itemsets:
    :param minconf:
    :return:
    """
    rules: list[AssociationRule] = []
    for itemset in frequent_itemsets:
        for subset in generate_subset_combinations(itemset):
            if support(itemset, transactions) / support(set(subset), transactions) >= minconf:
                rules.append(
                    dict(
                        antecedent=set(subset),
                        consequent=itemset - set(subset),
                    )
                )
    return rules


def generate_strong_association_rules(
    elements_universe: set, transactions: list[set]
) -> list[AssociationRule]:
    frequent_itemsets = find_frequent_itemsets(elements_universe, transactions)
    strong_association_rules = generate_association_rules(frequent_itemsets, transactions)
    return strong_association_rules


def main():
    elements = {"cola", "orzeszki", "pieluszki", "piwo"}
    transactions = [
        {"cola", "orzeszki"},
        {"orzeszki", "pieluszki", "piwo"},
        {"cola"},
        {"cola", "orzeszki", "piwo"},
        {"piwo", "orzeszki", "pieluszki"},
    ]
    rules = generate_strong_association_rules(elements, transactions)
    for rule in rules:
        logger.info(f"Association rule {rule['antecedent']} -> {rule['consequent']}")


if __name__ == "__main__":
    main()
