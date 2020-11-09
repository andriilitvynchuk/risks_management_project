from typing import List

import numpy as np
import pandas as pd


def create_days_to_fall_relative_to_current_day(variable: pd.Series, percent: float = 0.02) -> pd.Series:
    target: List[int] = []
    for index in range(len(variable) - 1):
        # target.append(int((variable[index] - variable[index + 1]) / variable[index] > percent))
        fall_happened = False
        for add_index in range(index + 1, len(variable)):
            if (variable[index] - variable[add_index]) > percent:
                target.append(add_index - index)
                fall_happened = True
                break
        if not fall_happened:
            target.append(np.nan)
    target.append(np.nan)
    return pd.Series(target, index=variable.index)
