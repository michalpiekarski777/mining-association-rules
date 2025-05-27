import numpy as np
import pandas as pd

from src.mar.apriori_df.interest_measures.batch_support_count import BatchSupportCount
from src.mar.common.utils import consts


class BatchSupport(BatchSupportCount):
    def __init__(self):
        super().__init__()

    def calculate(
        self,
        itemsets: list[frozenset[str]],
        df: pd.DataFrame,
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> np.ndarray:
        return super().calculate(itemsets, df, minsup) / len(df)
