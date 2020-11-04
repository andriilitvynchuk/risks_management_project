import sys
from typing import Iterable, NoReturn, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


sys.path.append(".")
from probability_prediction.target import create_event_happened_day
from utils.dataset import create_arma_table


def print_scores(
    y_true: Iterable, y_predict: Iterable, prefix: str = "", threshold: Optional[float] = None
) -> NoReturn:
    if threshold is None:
        best_acc = 0
        best_f1 = 0
        best_threshold = None
        for tmp_threshold in np.arange(0, 1, 0.001):
            f1 = f1_score((y_true > tmp_threshold), (y_predict > tmp_threshold))
            acc = accuracy_score((y_true > tmp_threshold), (y_predict > tmp_threshold))
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_threshold = tmp_threshold
        print(f"{prefix} Threshold: {best_threshold} Best F1-score: {best_f1:.4f}, Best Accuracy: {best_acc:.4f}")
    else:
        f1 = f1_score((y_true > threshold), (y_predict > threshold))
        acc = accuracy_score((y_true > threshold), (y_predict > threshold))
        print(f"{prefix} F1-score: {f1:.4f}, Accuracy: {acc:.4f}")


def main() -> NoReturn:
    table = pd.read_csv("./data/tesla_stock.csv", index_col=0)
    table.index = pd.to_datetime(table.index)
    variable = table["Close"]
    percent = 0.02

    x = create_arma_table(variable=variable, p=50, q=50, ma_window=10)
    x["event_happened_day"] = create_event_happened_day(variable=variable, percent=percent)
    x = x.dropna()

    x_train, x_test = x.loc[:"2019-12"], x.loc["2020-01"]
    x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1]
    x_test, y_test = x_test.iloc[:, :-1], x_test.iloc[:, -1]

    models = [
        LogisticRegression(fit_intercept=False, n_jobs=-1, C=100, max_iter=5000),
        SVC(C=100, probability=True),
        RandomForestClassifier(n_estimators=100),
        KNeighborsClassifier(5),
    ]

    fig, axs = plt.subplots(2, 1, figsize=(16, 16))

    for model in models:
        model.fit(x_train, y_train)
        y_predict = model.predict_proba(x_test)[:, 1]

        class_name = model.__class__.__name__
        print_scores(y_test, y_predict, f"{class_name} day event prediction")
        print("")

        axs[0].plot(y_test.index, y_predict, label=class_name)

    axs[0].plot(y_test, label="True")
    axs[0].legend()
    axs[0].set_title(f"Probability that price will fall by {percent}% relative to previous day")

    # axs[1].plot(true_month_losses, label="True")
    # axs[1].legend()
    # axs[1].set_title("Losses relative to previous month average price")
    plt.show()


if __name__ == "__main__":
    main()