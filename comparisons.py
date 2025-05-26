from pathlib import Path

from config import ROOT_DIR

from src.mar.apriori_df.interest_measures.batch_confidence import BatchConfidence
from src.mar.apriori_df.interest_measures.batch_support import BatchSupport

import apyori
import efficient_apriori
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

path = Path(ROOT_DIR) / "sources" / "survey.parquet"
df = pd.read_parquet(path)
MIN_SUPPORT = 0.6
MIN_CONFIDENCE = 0.6

transactions = [row[row == 1].index.tolist() for _, row in df.iterrows()]

# Efficient-Apriori
itemsets, rules = efficient_apriori.apriori(transactions, min_support=0.6,  min_confidence=0.6)
print(len(rules))

# Apyori
apyori_rules = list(apyori.apriori(transactions, min_support=0.6,  min_confidence=0.6))
print(len(apyori_rules))

# Mlxtend
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
print(len(rules_df))
