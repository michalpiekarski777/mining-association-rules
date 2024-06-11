from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator
from src.mining_association_rules.apriori_df.interest_measures import (
    Confidence,
    HyperConfidence,
    Support,
)

# convert your dataset to required format
elements = ["cola", "peanuts", "diapers", "beer"]
transactions = [
    ["cola", "peanuts"],
    ["peanuts", "diapers", "beer"],
    ["cola"],
    ["cola", "peanuts", "beer"],
    ["peanuts", "diapers", "beer"],
]
path = Path(ROOT_DIR) / "sources" / "survey.parquet"
df = pd.read_parquet(path)
# specify measures used during the processing with its thresholds
itemset_measures = {Support: 0.4}
rule_measures = {Confidence: 0.2, HyperConfidence: 0.1}
# run algorithm
rule_gen = DataFrameRuleGenerator(itemset_measures=itemset_measures, rule_measures=rule_measures)
rules = rule_gen.generate_strong_association_rules(transactions=df)
print(rules[0])
