from typing import TypedDict


class AssociationRule(TypedDict):
    antecedent: set
    consequent: set
    support: float
    confidence: float
