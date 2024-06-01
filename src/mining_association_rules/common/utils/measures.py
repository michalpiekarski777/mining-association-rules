from src.mining_association_rules.apriori_df.interest_measures import (
    AntiSupport,
    Confidence,
    Conviction,
    DependencyFactor,
    GainFunction,
    Lift,
    RuleInterestFunction,
    Support,
)

interest_measures_classes = {"support": Support}

rule_measures_classes = {
    "anti_support": AntiSupport,
    "confidence": Confidence,
    "conviction": Conviction,
    "dependency_factor": DependencyFactor,
    "gain_function": GainFunction,
    "lift": Lift,
    "rule_interest_function": RuleInterestFunction,
}
