from typing import TypedDict


class Measure(TypedDict):
    name: str
    value: float


class AssociationRule(TypedDict):
    antecedent: set
    consequent: set
    itemset_measure: Measure
    rule_measure: Measure
