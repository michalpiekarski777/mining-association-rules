from enum import StrEnum


class FileFormat(StrEnum):
    PARQUET = "parquet"
    CSV = "csv"


class RunnerType(StrEnum):
    DATAFRAME = "df"
    LIST = "list"
