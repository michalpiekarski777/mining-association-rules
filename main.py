import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator
from src.mining_association_rules.apriori_df.interest_measures.anti_support import AntiSupport
from src.mining_association_rules.apriori_df.interest_measures.confidence import Confidence
from src.mining_association_rules.apriori_df.interest_measures.conviction import Conviction
from src.mining_association_rules.apriori_df.interest_measures.dependency_factor import (
    DependencyFactor,
)
from src.mining_association_rules.apriori_df.interest_measures.gain_function import GainFunction
from src.mining_association_rules.apriori_df.interest_measures.lift import Lift
from src.mining_association_rules.apriori_df.interest_measures.rule_interest_function import (
    RuleInterestFunction,
)
from src.mining_association_rules.apriori_df.interest_measures.support import Support
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
        help="Name of Parquet file placed in sources directory",
    )
    parser.add_argument("-r", "--runner", default="df", choices=["df", "list"])
    return parser.parse_args()


def prepare_df_gen(source: str) -> tuple[DataFrameRuleGenerator, dict]:
    df = pd.read_parquet(Path(ROOT_DIR) / "sources" / source)
    rule_gen = DataFrameRuleGenerator(
        source=source,
        itemset_measure=Support,
        rule_measures=[
            AntiSupport,
            Confidence,
            Conviction,
            DependencyFactor,
            GainFunction,
            Lift,
            RuleInterestFunction,
        ],
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

    if args.runner == RunnerType.DATAFRAME.value:
        rule_gen, kwargs = prepare_df_gen(args.file)
    elif args.runner == RunnerType.LIST.value:
        rule_gen, kwargs = prepare_list_gen(args.file)
    else:
        sys.exit("Invalid runner")

    run(rule_gen, kwargs)


if __name__ == "__main__":
    main()
