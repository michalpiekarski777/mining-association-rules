from contextlib import AbstractContextManager
from typing import Protocol

from src.mining_association_rules.common.utils.typed_dicts import AssociationRule


class RuleGeneratorProtocol(Protocol, AbstractContextManager):
    support_calculations: int
    support_calculations_time: float
    total_duration: float
    _runner: str
    _source: str
    _rules: list[AssociationRule]

    def generate_strong_association_rules(self, *args, **kwargs) -> list[AssociationRule]: ...

    def support(self, *args, **kwargs) -> float: ...
