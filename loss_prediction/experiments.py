import sys
from typing import NoReturn

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sys.path.append(".")
from loss_prediction.target import create_next_day_price
from utils.dataset import create_arma_table


def main() -> NoReturn:
    table = pd.read_csv("./data/tesla_stock.csv", index_col=0)
    table.index = pd.to_datetime(table.index)
    variable = table["Close"]

    x = create_arma_table(variable=variable, p=10, q=5, ma_window=10)
    x["next_day_price"] = create_next_day_price(variable=variable)
    x = x.dropna()

    x_train, x_test = x.loc[:"2019-12"], x.loc["2020-01"]
    x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1]
    x_test, y_test = x_test.iloc[:, :-1], x_test.iloc[:, -1]

    model = LinearRegression(fit_intercept=False, n_jobs=-1)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    print(f"MAE: {mae:4f} MSE: {mse:.4f}, R2-score: {r2:.4f}")

    plt.plot(y_test.index, y_predict, label="predict")
    plt.plot(y_test, label="true")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
