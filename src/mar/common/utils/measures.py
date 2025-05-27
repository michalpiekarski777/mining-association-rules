from src.mar.apriori_df.interest_measures import AntiSupport
from src.mar.apriori_df.interest_measures import BatchConfidence
from src.mar.apriori_df.interest_measures import BatchConviction
from src.mar.apriori_df.interest_measures import BatchDependencyFactor
from src.mar.apriori_df.interest_measures import BatchSupport
from src.mar.apriori_df.interest_measures import Confidence
from src.mar.apriori_df.interest_measures import Conviction
from src.mar.apriori_df.interest_measures import DependencyFactor
from src.mar.apriori_df.interest_measures import GainFunction
from src.mar.apriori_df.interest_measures import HyperConfidence
from src.mar.apriori_df.interest_measures import HyperLift
from src.mar.apriori_df.interest_measures import Lift
from src.mar.apriori_df.interest_measures import RuleInterestFunction
from src.mar.apriori_df.interest_measures import Support

interest_measures_classes = {"batch_support": BatchSupport, "support": Support}

rule_measures_classes = {
    "anti_support": AntiSupport,
    "batch_confidence": BatchConfidence,
    "batch_conviction": BatchConviction,
    "batch_dependency_factor": BatchDependencyFactor,
    "confidence": Confidence,
    "conviction": Conviction,
    "dependency_factor": DependencyFactor,
    "gain_function": GainFunction,
    "lift": Lift,
    "rule_interest_function": RuleInterestFunction,
    "hyperconfidence": HyperConfidence,
    "hyperlift": HyperLift,
}
