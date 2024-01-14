from typing import TypedDict


class AssociationRule(TypedDict):
    antecedent: set
    consequent: set
