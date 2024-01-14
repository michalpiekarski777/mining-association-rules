import pandas as pd

from apriori_df.apriori.attractiveness_measures import support as df_support
from apriori_list.apriori.attractiveness_measures import support as list_support


def support(itemset: set, transactions: pd.DataFrame | list[set]) -> float:
    if isinstance(transactions, list):
        return list_support(itemset, transactions)

    elif isinstance(transactions, pd.DataFrame):
        return df_support(itemset, transactions)

    raise TypeError
