from pathlib import Path

import pandas as pd
import pytest

from config import ROOT_DIR


@pytest.fixture
def example_dataset():
    return pd.read_csv(Path(ROOT_DIR) / "sources" / "book_example.csv")
