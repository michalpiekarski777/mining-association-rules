from typing import TypedDict


class MeasureTypedDict(TypedDict):
    name: str
    value: float


class AssociationRule(TypedDict):
    antecedent: frozenset[str]
    consequent: frozenset[str]
    itemset_measure: MeasureTypedDict
    rule_measures: list[MeasureTypedDict]
