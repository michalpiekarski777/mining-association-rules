import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator
from src.mining_association_rules.apriori_list.apriori.apriori import ListRuleGenerator
from src.mining_association_rules.common.utils.enums import RunnerType
from src.mining_association_rules.common.utils.read_csv import read_transactions_shop
from src.mining_association_rules.common.utils.runners import run

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Mining association rules",
        description="Generator of association rules from your dataset",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        type=str,
        help="Name of CSV or Parquet file placed in sources directory",
    )
    parser.add_argument(
        "-i",
        "--itemset_measures",
        default=["Support"],
        type=str,
        nargs="*",
        help="List of interest measures used to generate frequent itemsets",
        choices=["Support"],
    )
    parser.add_argument(
        "-r",
        "--rule_measures",
        default=["Confidence"],
        type=str,
        nargs="*",
        help="List of interest measured used to generate strong association rules",
        choices=[
            "AntiSupport",
            "Confidence",
            "Conviction",
            "DependencyFactor",
            "GainFunction",
            "Lift",
            "RuleInterestFunction",
        ],
    )
    parser.add_argument("-b", "--backend", default="df", choices=["df", "list"])
    return parser.parse_args()


def prepare_df_gen(
    source: str, itemset_measures: list[str], rule_measures: list[str]
) -> tuple[DataFrameRuleGenerator, dict]:
    path = Path(ROOT_DIR) / "sources" / source
    df = pd.read_csv(path) if source.endswith(".csv") else pd.read_parquet(path)
    rule_gen = DataFrameRuleGenerator(
        source=source, itemset_measures=itemset_measures, rule_measures=rule_measures
    )
    kwargs = dict(transactions=df)
    return rule_gen, kwargs


def prepare_list_gen(source: str) -> tuple[ListRuleGenerator, dict]:
    path = Path(ROOT_DIR) / "sources" / source
    elements, transactions = read_transactions_shop(path)
    rule_gen = ListRuleGenerator(source=source)
    kwargs = dict(transactions=transactions, elements=elements)
    return rule_gen, kwargs


def main():
    args = parse_args()

    if args.backend == RunnerType.DATAFRAME:
        rule_gen, kwargs = prepare_df_gen(args.file, args.itemset_measures, args.rule_measures)
    elif args.backend == RunnerType.LIST:
        rule_gen, kwargs = prepare_list_gen(args.file)
    else:
        sys.exit("Invalid runner")

    run(rule_gen, kwargs)


if __name__ == "__main__":
    main()
