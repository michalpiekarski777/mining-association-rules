import logging
import time

import pandas as pd

from common.utils.exceptions import EmptyTransactionBaseException

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def support(itemset: set, df: pd.DataFrame) -> float:
    logger.info("Calculating support")
    if df.empty is True:
        raise EmptyTransactionBaseException
    start = time.perf_counter()
    supported_df = df[df[list(itemset)].eq(1).all(axis=1)]
    end = time.perf_counter()
    logger.info(f"Creating supporting df took {end - start}")
    return len(supported_df) / len(df)
