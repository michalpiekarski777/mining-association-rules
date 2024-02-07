import pandas as pd


def read_transactions_groceries(path):
    df = pd.read_csv(path)
    elements = set(df["itemDescription"].unique())
    groupby_agg = df.groupby(["Member_number", "Date"])[["itemDescription"]].agg(set)
    transactions = groupby_agg["itemDescription"].to_list()
    return elements, transactions


def read_transactions_shop(path):
    df = pd.read_csv(path, sep=";")
    elements = set(df["Itemname"].unique())
    groupby_agg = df.groupby(["BillNo"])[["Itemname"]].agg(set)
    transactions = groupby_agg["Itemname"].to_list()
    return elements, transactions


def read_mobile_survey(path):
    pass
