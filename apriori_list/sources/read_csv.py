import pandas as pd


def read_transactions(path):
    df = pd.read_csv(path)
    elements = set(df["itemDescription"].unique())
    groupby_agg = df.groupby(["Member_number", "Date"])[["itemDescription"]].agg(set)
    transactions = groupby_agg["itemDescription"].to_list()
    return elements, transactions


def read_transactions_df(path):
    df = pd.read_csv(path)
    elements = set(df["itemDescription"].unique())
    groupby_agg = df.groupby(["Member_number", "Date"])[["itemDescription"]].agg(set)
    transactions = groupby_agg["itemDescription"].to_list()
    return elements, transactions
