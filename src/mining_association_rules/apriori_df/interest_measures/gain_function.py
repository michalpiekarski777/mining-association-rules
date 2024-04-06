import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures import SupportCount
from src.mining_association_rules.apriori_df.interest_measures.base import Measure


class GainFunction(Measure):
    def __init__(self):
        super().__init__()

    def gain_function(
        self,
        antecedent: frozenset[str],
        consequent: frozenset[str],
        df: pd.DataFrame,
        support_count: SupportCount,
        gain: float = 0.8,
    ) -> float:
        rule_support_count = support_count.calculate(antecedent | consequent, df)
        antecedent_support_count = support_count.calculate(antecedent, df)

        return rule_support_count - gain * antecedent_support_count
