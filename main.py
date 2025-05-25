import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator
from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.apriori_list.apriori.apriori import ListRuleGenerator
from src.mining_association_rules.common.utils.enums import RunnerType
from src.mining_association_rules.common.utils.measures import interest_measures_classes
from src.mining_association_rules.common.utils.measures import rule_measures_classes
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
        "-c",
        "--config",
        type=str,
        help="Path to json config file containing file, itemset_measures and rule_measures fields",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Name of CSV or Parquet file placed in sources directory",
    )
    parser.add_argument(
        "-i",
        "--itemset_measures",
        default=["batch_support=0.1"],
        type=str,
        nargs="*",
        help="List of interest measures used to generate frequent itemsets",
    )
    parser.add_argument(
        "-r",
        "--rule_measures",
        default=["batch_confidence=0.1"],
        type=str,
        nargs="*",
        help="List of interest measures used to generate strong association rules",
    )
    parser.add_argument("-b", "--backend", default="df", choices=["df", "list"])
    args = parser.parse_args()

    if not args.config and not args.file:
        msg = "Either config or file argument required"
        raise argparse.ArgumentTypeError(msg)

    if args.config:
        with Path.open(args.config) as f:
            config = json.load(f)
            args.file = config["file"]
            args.itemset_measures = {
                interest_measures_classes[measure]: threshold
                for measure, threshold in config["itemset_measures"].items()
            }
            args.rule_measures = {
                rule_measures_classes[measure]: threshold for measure, threshold in config["rule_measures"].items()
            }

    else:
        args.itemset_measures = {
            parse_measures_threshold(measure, interest_measures_classes)[0]: parse_measures_threshold(
                measure,
                interest_measures_classes,
            )[1]
            for measure in args.itemset_measures
        }
        args.rule_measures = {
            parse_measures_threshold(measure, rule_measures_classes)[0]: parse_measures_threshold(
                measure,
                rule_measures_classes,
            )[1]
            for measure in args.rule_measures
        }
    return args


def parse_measures_threshold(value: str, measure_classes: dict) -> tuple[Measure, float]:
    try:
        measure, threshold = value.split("=")
        return measure_classes[measure], float(threshold)
    except ValueError:
        msg = f"Invalid format for metric and threshold: '{value}'. Expected format is metric=threshold."
        raise argparse.ArgumentTypeError(msg) from None
    except KeyError:
        msg = f"Metric value {value.split('=')[0]} is not available in {list(measure_classes.keys())}"
        raise argparse.ArgumentTypeError(msg) from None


def prepare_df_gen(
    source: str,
    itemset_measures: dict[type[Measure], float],
    rule_measures: dict[type[Measure], float],
) -> tuple[DataFrameRuleGenerator, dict]:
    path = Path(ROOT_DIR) / "sources" / source
    df = pd.read_csv(path) if source.endswith(".csv") else pd.read_parquet(path)
    rule_gen = DataFrameRuleGenerator(itemset_measures=itemset_measures, rule_measures=rule_measures)
    kwargs = {"transactions": df}
    return rule_gen, kwargs


def prepare_list_gen(source: str) -> tuple[ListRuleGenerator, dict]:
    path = Path(ROOT_DIR) / "sources" / source
    elements, transactions = read_transactions_shop(path)
    rule_gen = ListRuleGenerator(source=source)
    kwargs = {"elements": elements, "transactions": transactions}
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
