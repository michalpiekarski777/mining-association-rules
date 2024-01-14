import pandas as pd

from common.utils.exceptions import EmptyTransactionBaseException


def support(itemset: set, df: pd.DataFrame) -> float:
    if df.empty is True:
        raise EmptyTransactionBaseException

    supported_df = df[df[list(itemset)].eq(1).all(axis=1)]
    return len(supported_df) / len(df)
