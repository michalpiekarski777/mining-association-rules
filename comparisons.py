from pathlib import Path

import apyori
import efficient_apriori
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

from config import ROOT_DIR

path = Path(ROOT_DIR) / "sources" / "survey.parquet"
df = pd.read_parquet(path)
MIN_SUPPORT = 0.6
MIN_CONFIDENCE = 0.6

transactions = [row[row == 1].index.tolist() for _, row in df.iterrows()]

# Efficient-Apriori
itemsets, rules = efficient_apriori.apriori(transactions, min_support=0.6, min_confidence=0.6)

# Apyori
apyori_rules = list(apyori.apriori(transactions, min_support=0.6, min_confidence=0.6))

# Mlxtend
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
