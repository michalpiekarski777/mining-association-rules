from src.mar.apriori_df.interest_measures import AntiSupport
from src.mar.apriori_df.interest_measures import Confidence
from src.mar.apriori_df.interest_measures import Conviction
from src.mar.apriori_df.interest_measures import DependencyFactor
from src.mar.apriori_df.interest_measures import GainFunction
from src.mar.apriori_df.interest_measures import HyperConfidence
from src.mar.apriori_df.interest_measures import HyperLift
from src.mar.apriori_df.interest_measures import Lift
from src.mar.apriori_df.interest_measures import RuleInterestFunction
from src.mar.apriori_df.interest_measures import Support

interest_measures_classes = {"support": Support}

rule_measures_classes = {
    "anti_support": AntiSupport,
    "confidence": Confidence,
    "conviction": Conviction,
    "dependency_factor": DependencyFactor,
    "gain_function": GainFunction,
    "lift": Lift,
    "rule_interest_function": RuleInterestFunction,
    "hyperconfidence": HyperConfidence,
    "hyperlift": HyperLift,
}
