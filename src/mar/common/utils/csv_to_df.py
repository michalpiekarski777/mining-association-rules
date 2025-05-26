from pathlib import Path

import pandas as pd

from config import ROOT_DIR
from src.mar.common.utils.enums import FileFormat
from src.mar.common.utils.read_csv import read_mobile_survey
from src.mar.common.utils.read_csv import read_transactions_groceries


def convert_dataset_to_dataframe(elements: list, transactions: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(0, columns=list(elements), index=range(len(transactions)))

    for i, transaction in enumerate(transactions):
        df.loc[i, list(transaction)] = 1

    return df


def dump_dataframe(df, path, file_format):
    if file_format == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)


def convert(elements, transactions, path, file_format):
    df = convert_dataset_to_dataframe(elements, transactions)
    dump_dataframe(df, path, file_format)


def transform_groceries_dataset():
    input_path = Path(ROOT_DIR) / "apriori_df" / "sources" / "raw" / "groceries.csv"
    elements, transactions = read_transactions_groceries(input_path)
    output_path = Path(ROOT_DIR) / "apriori_df" / "sources" / "groceries.csv"
    convert(elements, transactions, output_path, file_format=FileFormat.CSV)


def transform_mobile_survey_dataset():
    input_path = Path(ROOT_DIR) / "sources" / "raw" / "raw-data.csv"
    elements, transactions = read_mobile_survey(input_path)
    output_path = Path(ROOT_DIR) / "sources" / "survey.parquet"
    convert(elements, transactions, output_path, file_format=FileFormat.PARQUET)
