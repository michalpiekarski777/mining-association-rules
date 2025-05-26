from src.mar.apriori_df.apriori import DataFrameRuleGenerator
from src.mar.apriori_df.interest_measures import BatchConfidence
from src.mar.apriori_df.interest_measures import BatchSupport
from src.mar.common.utils.csv_to_df import convert_dataset_to_dataframe

# convert your dataset to required format
elements = ["cola", "peanuts", "diapers", "beer"]
transactions = [
    ["cola", "peanuts"],
    ["peanuts", "diapers", "beer"],
    ["cola"],
    ["cola", "peanuts", "beer"],
    ["peanuts", "diapers", "beer"],
]
df = convert_dataset_to_dataframe(elements, transactions)
# specify measures used during the processing with its thresholds
itemset_measures = {BatchSupport: 0.4}
rule_measures = {BatchConfidence: 0.2}
# run algorithm
rule_gen = DataFrameRuleGenerator(itemset_measures=itemset_measures, rule_measures=rule_measures)
rules = rule_gen.generate_strong_association_rules(transactions=df)
