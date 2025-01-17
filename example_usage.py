from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator
from src.mining_association_rules.apriori_df.interest_measures import Confidence, HyperConfidence, Support

# read dataset from file put in sources catalog
path = Path(ROOT_DIR) / "sources" / "survey.parquet"
df = pd.read_parquet(path)
# specify measures used during the processing with its thresholds
itemset_measures = {Support: 0.7}
rule_measures = {Confidence: 0.7, HyperConfidence: 0.9}
# run algorithm
rule_gen = DataFrameRuleGenerator(itemset_measures=itemset_measures, rule_measures=rule_measures)
rules = rule_gen.generate_strong_association_rules(transactions=df)
print(rules[0])
