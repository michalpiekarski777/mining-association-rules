import logging
import time

import pandas as pd

from apriori_list.sources.data import elements, transactions
from common.apriori.rules import generate_strong_association_rules

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    df = pd.read_csv("apriori_df/sources/1.csv", index_col=False, dtype="uint8")
    start = time.perf_counter()
    rules = generate_strong_association_rules(df)
    end = time.perf_counter()
    logger.info(f"Rules generated using DataFrame transaction database in {end - start} seconds")
    for rule in rules:
        logger.info(f"Association rule {rule['antecedent']} -> {rule['consequent']}")

    start = time.perf_counter()
    rules = generate_strong_association_rules(transactions, elements)
    end = time.perf_counter()
    logger.info(f"Rules generated using Python lists transaction database in {end - start} seconds")
    for rule in rules:
        logger.info(f"Association rule {rule['antecedent']} -> {rule['consequent']}")


if __name__ == "__main__":
    main()
