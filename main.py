import logging
import sys
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator
from src.mining_association_rules.apriori_list.apriori.apriori import ListRuleGenerator
from src.mining_association_rules.common.utils.read_csv import read_transactions_shop

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    runner = sys.argv[1] if len(sys.argv) > 1 else "default"
    if runner == "default":
        source = "survey.parquet"
        df = pd.read_parquet(Path(ROOT_DIR) / "sources" / source)
        generator_class = DataFrameRuleGenerator
        kwargs = dict(transactions=df)

    else:
        source = "shop.csv"
        path = Path(ROOT_DIR) / "sources" / source
        elements, transactions = read_transactions_shop(path)
        generator_class = ListRuleGenerator
        kwargs = dict(transactions=transactions, elements=elements)

    with generator_class(source=source) as rule_gen:
        rule_gen.generate_strong_association_rules(**kwargs)


if __name__ == "__main__":
    main()
