from typing import TypedDict


class Measure(TypedDict):
    name: str
    value: float


class AssociationRule(TypedDict):
    antecedent: frozenset[str]
    consequent: frozenset[str]
    itemset_measure: Measure
    rule_measure: Measure
