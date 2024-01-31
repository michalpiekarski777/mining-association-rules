import time
from itertools import chain, combinations

import pandas as pd

from apriori_df.apriori.apriori import DataFrameRuleGenerator
from apriori_list.apriori.itemsets import find_frequent_itemsets as li_find_frequent_itemsets
from common.apriori.attractiveness_measures import support
from common.utils import consts
from common.utils.typed_dicts import AssociationRule


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
    frequent_itemsets: list[set],
    transactions: pd.DataFrame | list[set],
    minconf: float = consts.CONFIDENCE_THRESHOLD,
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
            if (
                support(itemset, transactions, None)
                / support(set(subset), transactions, None)
                >= minconf
            ):
                rules.append(
                    dict(
                        antecedent=set(subset),
                        consequent=itemset - set(subset),
                    )
                )
    return rules


def generate_strong_association_rules(
    transactions: pd.DataFrame | list[set], elements: set | None = None
) -> list[AssociationRule]:
    if isinstance(transactions, pd.DataFrame):
        rule_generator = DataFrameRuleGenerator()
        frequent_itemsets = rule_generator.find_frequent_itemsets(
            transactions, consts.SUPPORT_THRESHOLD
        )

    elif isinstance(transactions, list):
        frequent_itemsets = li_find_frequent_itemsets(
            elements, transactions, consts.SUPPORT_THRESHOLD
        )

    else:
        raise TypeError

    start = time.perf_counter()
    rules = generate_association_rules(frequent_itemsets, transactions)
    end = time.perf_counter()
    print(f"Generating rules took {end - start} seconds")
    return rules
