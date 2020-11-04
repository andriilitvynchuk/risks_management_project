from typing import List

import numpy as np
import pandas as pd


def create_event_happened_day(variable: pd.Series, percent: float = 0.03) -> pd.Series:
    target: List[int] = []
    for index in range(len(variable) - 1):
        target.append(int((variable[index] - variable[index + 1]) / variable[index] > percent))
    target.append(np.nan)
    return pd.Series(target, index=variable.index)
