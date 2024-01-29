import logging
import time

import pandas as pd

import database_generator
from apriori_list.sources.read_csv import read_transactions
# from apriori_list.sources.data import elements, transactions
from common.apriori.rules import generate_strong_association_rules

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Support:
    def __init__(self):
        self.duration = 0.0
        self.supp_calculations = 0


def main():
    # elements, transactions, to_df_transactions = database_generator.generate_elements_and_transactions()
    # df = pd.DataFrame(to_df_transactions)
    # df.fillna(0, inplace=True)
    df = pd.read_csv("dataset.csv", index_col=False)
    df = df.astype(pd.SparseDtype("uint8", 0))
    start = time.perf_counter()
    sp = Support()
    rules = generate_strong_association_rules(df, elements=None, sp=sp)
    end = time.perf_counter()
    logger.info(f"Rules generated using DataFrame transaction database in {end - start} seconds")
    logger.info(f"Found {len(rules)} rules")
    # for rule in rules:
    #     logger.info(f"Association rule {rule['antecedent']} -> {rule['consequent']}")

    # elements, transactions = read_transactions("apriori_list/sources/Groceries_dataset.csv")
    # start = time.perf_counter()
    # support = Support()
    # rules = generate_strong_association_rules(transactions, elements, support)
    # end = time.perf_counter()
    # logger.info(f"Rules generated using Python lists transaction database in {end - start} seconds")
    # logger.info(f"Found {len(rules)} rules")
    # for rule in rules:
    #     logger.info(f"Association rule {rule['antecedent']} -> {rule['consequent']}")


if __name__ == "__main__":
    main()
