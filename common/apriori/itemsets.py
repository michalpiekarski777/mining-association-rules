import logging
import time

import pandas as pd

from common.apriori.attractiveness_measures import support
from common.utils import consts

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def apriori_gen(
    itemsets: list[set],
    transactions: pd.DataFrame | list[set],
    minsup: float = consts.SUPPORT_THRESHOLD,
    sp=None,
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

    for index, (itemset, sorted_itemset) in enumerate(zip(itemsets, sorted_itemsets)):
        for j in range(index + 1, len(itemsets)):
            if sorted_itemset[: set_length - 1] == sorted_itemsets[j][: set_length - 1]:
                candidates.append(itemset.union(itemsets[j]))
    end = time.perf_counter()
    logger.info(f"Generating candidates of length 2 took {end - start}")
    return [
        candidate
        for candidate in candidates
        if support(candidate, transactions, sp) >= minsup
    ]
