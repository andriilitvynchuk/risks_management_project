import sys
from typing import Iterable, NoReturn

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor


sys.path.append(".")
from day_prediction.target import create_days_to_fall_relative_to_previous_month
from utils.dataset import create_arma_table


def print_scores(y_true: Iterable, y_predict: Iterable, prefix: str = "") -> NoReturn:
    mae = mean_absolute_error(y_true, y_predict)
    mse = mean_squared_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    print(f"{prefix} MAE: {mae:4f} MSE: {mse:.4f}, R2-score: {r2:.4f}4")


def main() -> NoReturn:
    table = pd.read_csv("./data/tesla_stock.csv", index_col=0)
    table.index = pd.to_datetime(table.index)
    variable = table["Close"]
    percent = 0.02

    x = create_arma_table(variable=variable, p=50, q=50, ma_window=10)
    x["days_to_fall_relative_to_previous_month"] = create_days_to_fall_relative_to_previous_month(
        variable=variable, percent=percent
    )
    x = x.dropna()
    x = x.loc["2019":]

    x_train, x_test = x.loc[:"2019-11"], x.loc["2019-12"]
    x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1]
    x_test, y_test = x_test.iloc[:, :-1], x_test.iloc[:, -1]

    models = [
        LinearRegression(fit_intercept=False, n_jobs=-1),
        Ridge(fit_intercept=False, alpha=1),
        Lasso(fit_intercept=False, alpha=1),
        SVR(),
        RandomForestRegressor(n_estimators=50, max_depth=5),
        XGBRegressor(n_estimators=50, reg_lambda=0.005, max_depth=5, n_jobs=-1),
    ]

    for model in models:
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        print(y_predict, y_test.values)

        class_name = model.__class__.__name__
        print_scores(y_train, model.predict(x_train), f"{class_name} train month event prediction")
        print_scores(y_test, y_predict, f"{class_name} test month event prediction")
        print("")

        plt.plot(y_test.index, y_predict, label=class_name)

    plt.plot(y_test, label="True")
    plt.legend()
    plt.title(f"Days to fall by {percent}% relative to previous month")
    plt.show()


if __name__ == "__main__":
    main()
