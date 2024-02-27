from typing import Any

from src.mining_association_rules.common.utils.protocols import RuleGeneratorProtocol


def run(rules_gen: RuleGeneratorProtocol, kwargs: dict[Any, Any]) -> None:
    with rules_gen:
        rules_gen.generate_strong_association_rules(**kwargs)

    return None
