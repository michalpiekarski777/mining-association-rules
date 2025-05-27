from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mar.apriori_df.apriori import DataFrameRuleGenerator
from src.mar.apriori_df.interest_measures import Confidence
from src.mar.apriori_df.interest_measures import Support

# read dataset from file put in sources catalog
path = Path(ROOT_DIR) / "sources" / "survey.parquet"
df = pd.read_parquet(path)
# specify measures used during the processing with its thresholds
itemset_measures = {Support: 0.6}
rule_measures = {Confidence: 0.6}
# run algorithm
with DataFrameRuleGenerator(itemset_measures=itemset_measures, rule_measures=rule_measures) as rule_gen:
    rule_gen.generate_strong_association_rules(transactions=df)
