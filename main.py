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
from src.mining_association_rules.common.utils.read_csv import read_transactions_shop
from src.mining_association_rules.common.utils.runners import run

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    runner = sys.argv[1] if len(sys.argv) > 1 else "default"
    if runner == "default":
        source = "survey.parquet"
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

    else:
        source = "shop.csv"
        path = Path(ROOT_DIR) / "sources" / source
        elements, transactions = read_transactions_shop(path)
        rule_gen = ListRuleGenerator(source=source)
        kwargs = dict(transactions=transactions, elements=elements)

    run(rule_gen, kwargs)


if __name__ == "__main__":
    main()
