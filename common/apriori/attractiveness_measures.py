import time

import pandas as pd

from apriori_df.apriori.attractiveness_measures import support as df_support
from apriori_list.apriori.attractiveness_measures import support as list_support


def support(itemset: set, transactions: pd.DataFrame | list[set], sp) -> float:
    start = time.perf_counter()
    if isinstance(transactions, list):
        result = list_support(itemset, transactions)
        duration = time.perf_counter() - start
        if sp is not None:
            sp.duration += duration
            sp.supp_calculations += 1
        return result

    elif isinstance(transactions, pd.DataFrame):
        result = df_support(itemset, transactions)
        duration = time.perf_counter() - start
        if sp is not None:
            sp.duration += duration
            sp.supp_calculations += 1
        return result

    raise TypeError
