import pandas as pd

from src.mining_association_rules.apriori_df.interest_measures import SupportCount
from src.mining_association_rules.apriori_df.interest_measures.base import Measure


class DependencyFactor(Measure):
    def __init__(self):
        super().__init__()

    def calculate(
        self,
        antecedent: frozenset[str],
        consequent: frozenset[str],
        df: pd.DataFrame,
        support_count: SupportCount,
    ) -> float:
        rule_support_count = support_count.calculate(antecedent | consequent, df)
        antecedent_support_count = support_count.calculate(antecedent, df)
        consequent_support_count = support_count.calculate(consequent, df)

        numerator = (rule_support_count / antecedent_support_count) - (
            consequent_support_count / len(df)
        )
        denominator = (rule_support_count / antecedent_support_count) + (
            consequent_support_count / len(df)
        )

        return numerator / denominator
