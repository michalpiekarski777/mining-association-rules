from typing import TypedDict


class MeasureThreshold(TypedDict):
    measure: str
    threshold: float


class Measure(TypedDict):
    name: str
    value: float


class AssociationRule(TypedDict):
    antecedent: frozenset[str]
    consequent: frozenset[str]
    itemset_measure: Measure
    rule_measures: list[Measure]
