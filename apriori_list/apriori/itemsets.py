import logging

from common.apriori.attractiveness_measures import support
from common.apriori.itemsets import apriori_gen
from common.utils import consts

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_frequent_itemsets(
    elements_universe: set, transactions: list[set], minsup: float = consts.SUPPORT_THRESHOLD
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
