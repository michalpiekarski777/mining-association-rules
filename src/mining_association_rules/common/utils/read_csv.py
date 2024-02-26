import re

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
    def cat_strings(row):
        row.dropna(inplace=True)
        values = set()
        for col in row:
            for value in re.split(r"[;,]", col):
                values.add(value.strip())
        return values

    df = pd.read_csv(path)
    df = df.iloc[
        :, 8:21
    ]  # select columns with answers to questions about what apps users use
    apps = df.apply(cat_strings, axis=1)
    elements = set(apps.explode().unique())
    transactions = apps.tolist()
    return elements, transactions
