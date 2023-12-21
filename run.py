import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv('temperature.csv')
x = data['일시'].values.reshape(-1, 1)
y = data['평균기온(°C)'].values
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()


def sklearn_linearReg():
    model=LinearRegression()
    model.fit(x,y)

    print(f'intercept: {model.intercept_}, coef: {model.coef_}')
    pred = model.predict(x)

    plt.scatter(x, y)
    plt.plot(x, pred, color='green')
    plt.xlabel('year')
    plt.ylabel('mean temperature(°C)')
    plt.show()

def custom_linearReg():
    def plot_prediction(pred, y):
        X = scaler_x.inverse_transform(x.reshape(-1, 1))
        Y = scaler_y.inverse_transform(y.reshape(-1, 1))
        plt.figure(figsize=(16, 6))
        plt.plot(X, Y)
        plt.plot(X, scaler_y.inverse_transform(pred.reshape(-1, 1)))
        plt.xlabel('year')
        plt.ylabel('mean temperature(°C)')
        plt.xticks(np.arange(min(X), max(X)+1, 5))
        plt.show()

    W = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)

    learning_rate = 0.001
    epoch = 10000

    for _ in range(epoch):
        Y_Pred = W * x + b
        error = np.sqrt((Y_Pred - y) ** 2).mean()

        if error < 0.001:
            print(f'error: {error}')
            break

        w_grad = learning_rate * ((Y_Pred - y) * x).mean()
        b_grad = learning_rate * (Y_Pred - y).mean()

        W = W - w_grad
        b = b - b_grad

    y_hat = W * x + b
    SSE = ((y_hat - y) ** 2).sum()
    SSR = ((y_hat - y.mean()) ** 2).sum()
    SST = SSR + SSE
    R2 = SSR / SST
    MSE = mean_squared_error(y, y_hat)

    print(f'SSE: {SSE}, R2: {R2}')
    print("결정계수: ", r2_score(y, y_hat))
    print("MSE: ", MSE)
    print(f'W: {W}, b: {b}')
    # plot_prediction(y_hat, y)
    return W, b, SSE, SSR, R2, MSE


def linear_regression_gradient_descent():
    X = np.random.rand(100)
    noise = np.random.normal(0, 0.02, 100)  # 약간의 무작위 노이즈 추가
    Y = 0.2 * X + 0.5 + noise

    def plot_prediction(pred, y):
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y)
        plt.scatter(X, pred)
        plt.show()

    W = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)

    learning_rate = 0.9
    epoch = 10000

    for i in range(epoch):
        Y_Pred = W * X + b
        error = np.sqrt((Y_Pred - Y) ** 2).mean()

        if error < 0.001:
            print(f'error: {error}')
            break

        w_grad = learning_rate * ((Y_Pred - Y) * X).mean()
        b_grad = learning_rate * (Y_Pred - Y).mean()

        W = W - w_grad
        b = b - b_grad

    y_hat = W * X + b
    SSE = ((y_hat - Y) ** 2).sum()
    SSR = ((y_hat - Y.mean()) ** 2).sum()
    SST = SSR + SSE
    R2 = SSR / SST
    MSE = mean_squared_error(Y, y_hat)

    print(f'SSE: {SSE}, R2: {R2}')
    print("결정계수: ", r2_score(Y, y_hat))
    print("MSE: ", MSE)
    print(f'W: {W}, b: {b}')
    #plot_prediction(y_hat, Y)
    return W, b, SSE, SSR, R2, MSE

def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


if __name__ == '__main__':
    a = custom_linearReg()
    b = linear_regression_gradient_descent()

    df = pd.DataFrame(np.array(a, b), index=['custom', 'sklearn'], column=['W', 'b', 'SSE', 'SSR', 'R2', 'MSE'])
    print(df)
