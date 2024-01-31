import pandas as pd

from apriori_list.sources.read_csv import read_transactions

elements, transactions = read_transactions("../../apriori_list/sources/groceries.csv")

df = pd.DataFrame(0, columns=list(elements), index=range(len(transactions)))

for i, transaction in enumerate(transactions):
    df.loc[i, list(transaction)] = 1

df.to_csv("dataset.csv", index=False)
