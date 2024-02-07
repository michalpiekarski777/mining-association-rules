import logging
import sys
import time
from pathlib import Path

import pandas as pd

from apriori_df.apriori.apriori import DataFrameRuleGenerator
from apriori_list.apriori.apriori import ListRuleGenerator
from common.utils.read_csv import read_transactions_shop
from config import ROOT_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    runner = sys.argv[1] if len(sys.argv) > 1 else "default"
    if runner == "default":
        start = time.perf_counter()
        df = pd.read_parquet(Path(ROOT_DIR) / "sources" / "shop.parquet")
        logger.info(f"Reading dataframe took {time.perf_counter() - start}")
        rule_generator = DataFrameRuleGenerator()
        rules = rule_generator.generate_strong_association_rules(df)
    else:
        path = Path(ROOT_DIR) / "sources" / "shop.csv"
        elements, transactions = read_transactions_shop(path)
        rule_generator = ListRuleGenerator()
        rules = rule_generator.generate_strong_association_rules(transactions, elements)

    logger.info(
        f"Rules generated using {runner} database in {rule_generator.total_duration} seconds"
    )
    logger.info(
        f"""
            Calculating support to generate association rules took {rule_generator.support_calculations_time}
            and was done {rule_generator.support_calculations} times
        """
    )
    logger.info(f"Found {len(rules)} rules")
    for rule in rules:
        logger.info(f"Association rule {rule['antecedent']} -> {rule['consequent']}")


if __name__ == "__main__":
    main()
