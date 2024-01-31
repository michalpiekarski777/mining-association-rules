import logging

from common.apriori.attractiveness_measures import support
from common.apriori.itemsets import apriori_gen
from common.utils import consts

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_frequent_itemsets(
    elements_universe: set,
    transactions: list[set],
    minsup: float = consts.SUPPORT_THRESHOLD,
    sp=None,
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
        if support({element}, transactions, sp) >= minsup
    ]
    if not itemsets:
        return frequent_itemsets

    logger.info(f"{1} elements frequent itemsets {len(itemsets)}")
    frequent_itemsets.extend(itemsets)
    for i in range(2, len(elements_universe)):
        logger.info(f"Searching for itemsets of length {i}")
        candidates = apriori_gen(itemsets, transactions, minsup, sp)
        itemsets = [
            candidate
            for candidate in candidates
            if support(candidate, transactions, sp) >= minsup
        ]
        logger.info(f"{i} elements frequent itemsets {len(itemsets)}")
        frequent_itemsets.extend(itemsets)
        if not itemsets:
            break
    logger.info(
        f"Calculating support took {sp.duration} and was done {sp.supp_calculations} times"
    )
    return frequent_itemsets
