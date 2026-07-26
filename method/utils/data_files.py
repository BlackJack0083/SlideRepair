from __future__ import annotations

from pathlib import Path

import pandas as pd

DATAFRAME_INDEX_COLUMN = "__index__"


def read_dataframe_csv(path: str | Path) -> pd.DataFrame:
    """Load a rendered dataframe CSV with its preserved index."""
    path = Path(path)
    df = pd.read_csv(path)
    if DATAFRAME_INDEX_COLUMN not in df.columns:
        raise ValueError(f"CSV is missing required index column: {path}")
    result = df.set_index(DATAFRAME_INDEX_COLUMN)
    result.index.name = None
    return result
