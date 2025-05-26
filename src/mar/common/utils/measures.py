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
from src.mar.apriori_df.interest_measures.batch_confidence import BatchConfidence
from src.mar.apriori_df.interest_measures.batch_support import BatchSupport

interest_measures_classes = {"batch_support": BatchSupport, "support": Support}

rule_measures_classes = {
    "anti_support": AntiSupport,
    "batch_confidence": BatchConfidence,
    "confidence": Confidence,
    "conviction": Conviction,
    "dependency_factor": DependencyFactor,
    "gain_function": GainFunction,
    "lift": Lift,
    "rule_interest_function": RuleInterestFunction,
    "hyperconfidence": HyperConfidence,
    "hyperlift": HyperLift,
}
