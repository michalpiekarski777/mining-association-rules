import re
from collections import Counter
from contextlib import suppress

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
        return {value.strip() for col in row.dropna() for value in re.split(r"[;,]", col)}

    def get_app_names(series):
        all_apps = []
        for row in series:
            for app in row:
                with suppress(Exception):
                    all_apps.append(app)
        return all_apps

    def filter_app_names(app_names, threshold=1):
        app_counts = Counter(app_names)
        return [app for app, count in app_counts.items() if count >= threshold and len(app) > 1]

    df = pd.read_csv(path)
    df = df.iloc[:, 8:21]  # select columns with answers to questions about what apps users use
    app_series = df.apply(cat_strings, axis=1)
    app_names = get_app_names(app_series)
    elements = filter_app_names(app_names, threshold=int(len(df) * 0.05))
    transactions = [{element for element in transaction if element in elements} for transaction in app_series.tolist()]

    return elements, transactions
