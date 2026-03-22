from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml


TARGET_COLUMN = "survived"


def load_dataset() -> pd.DataFrame:
    dataset = fetch_openml(name="titanic", version=1, as_frame=True)
    df = dataset.frame.copy()
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y