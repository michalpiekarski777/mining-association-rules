from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator
from src.mining_association_rules.apriori_df.interest_measures.batch_confidence import BatchConfidence
from src.mining_association_rules.apriori_df.interest_measures.batch_support import BatchSupport

# read dataset from file put in sources catalog
path = Path(ROOT_DIR) / "sources" / "survey.parquet"
df = pd.read_parquet(path)
# specify measures used during the processing with its thresholds
itemset_measures = {BatchSupport: 0.6}
rule_measures = {BatchConfidence: 0.6}
# run algorithm
with DataFrameRuleGenerator(itemset_measures=itemset_measures, rule_measures=rule_measures) as rule_gen:
    rule_gen.generate_strong_association_rules(transactions=df)
