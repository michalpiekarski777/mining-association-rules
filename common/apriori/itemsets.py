import logging

import pandas as pd

from common.apriori.attractiveness_measures import support

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def apriori_gen(
    itemsets: list[set], transactions: pd.DataFrame | list[set], minsup: float = 0.4
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

    return [
        candidate
        for candidate in candidates
        if support(candidate, transactions) >= minsup
    ]
