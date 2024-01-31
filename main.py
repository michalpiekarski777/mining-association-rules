import logging
import sys
import time
from pathlib import Path

import pandas as pd

from apriori_list.sources.read_csv import read_transactions
from common.apriori.rules import generate_strong_association_rules
from config import ROOT_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    runner = sys.argv[1]

    start = time.perf_counter()

    if runner == "df":
        df = pd.read_csv(
            Path(ROOT_DIR) / "apriori_df" / "sources" / "groceries.csv", index_col=False
        )
        rules = generate_strong_association_rules(df)
    else:
        path = Path(ROOT_DIR) / "apriori_list" / "sources" / "groceries.csv"
        elements, transactions = read_transactions(path)
        rules = generate_strong_association_rules(transactions, elements)

    end = time.perf_counter()
    logger.info(f"Rules generated using {runner} database in {end - start} seconds")
    logger.info(f"Found {len(rules)} rules")
    for rule in rules:
        logger.info(f"Association rule {rule['antecedent']} -> {rule['consequent']}")


if __name__ == "__main__":
    main()
