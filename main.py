import logging
import sys
from pathlib import Path

import pandas as pd

from apriori_df.apriori.apriori import DataFrameRuleGenerator
from apriori_list.sources.read_csv import read_transactions
from config import ROOT_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    runner = sys.argv[1]

    if runner == "df":
        df = pd.read_csv(
            Path(ROOT_DIR) / "apriori_df" / "sources" / "groceries.csv", index_col=False
        )
        rule_generator = DataFrameRuleGenerator()
        rules = rule_generator.generate_strong_association_rules(df)
    else:
        path = Path(ROOT_DIR) / "apriori_list" / "sources" / "groceries.csv"
        elements, transactions = read_transactions(path)
        # rules = generate_strong_association_rules(transactions, elements)

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
