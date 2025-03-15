import time

import numpy as np
import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.common.utils import consts


class BatchSupport(Measure):
    def __init__(self):
        super().__init__()

    def calculate(
        self,
        itemsets: list[frozenset[str]],
        df: pd.DataFrame,
        minsup: float = consts.SUPPORT_THRESHOLD,
    ) -> np.ndarray:
        start = time.perf_counter()
        item_to_idx = {item: idx for idx, item in enumerate(df.columns)}
        candidate_indices = [np.array([item_to_idx[item] for item in c]) for c in itemsets]
        data = df.to_numpy().astype(bool)
        supports = np.array([data[:, cols].all(axis=1).sum() for cols in candidate_indices])
        self.calculations_time += time.perf_counter() - start
        self.calculations_count += 1

        return supports
