from typing import TypedDict

from src.mining_association_rules.apriori_df.interest_measures.base import Measure


class MeasureTypedDict(TypedDict):
    name: str
    value: float


class AssociationRule(TypedDict):
    antecedent: frozenset[str]
    consequent: frozenset[str]
    itemset_measure: Measure
    rule_measures: list[MeasureTypedDict]
