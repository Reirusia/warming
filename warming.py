import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# get data
data = pd.read_csv('temperature.csv')
x = data['일시'].values
y = data['평균기온(°C)'].values

# normalize
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()


def sklearn_linearReg():
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    pred = model.predict(x.reshape(-1, 1))
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, pred)

    print(f'Sklearn Linear Regression - Intercept: {model.intercept_}, Coef: {model.coef_}')

    return model.coef_[0], model.intercept_, r2, rmse, pred


def custom_linearReg():
    W = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)

    learning_rate = 0.7
    max_epoch = 10000

    for epoch in range(max_epoch):
        y_hat = W * x + b
        # error: Root Mean Squared Error (RMSE)
        error = np.sqrt(((y_hat - y) ** 2).mean())

        if error < 0.001:
            print(f'Custom Linear Regression - Converged at Epoch {epoch + 1} - Error: {error}')
            break

        w_grad = learning_rate * ((y_hat - y) * x).mean()
        b_grad = learning_rate * (y_hat - y).mean()

        W = W - w_grad
        b = b - b_grad

    # result
    SSE = ((y_hat - y) ** 2).sum()
    SSR = ((y_hat - y.mean()) ** 2).sum()
    SST = SSR + SSE
    R2 = SSR / SST
    RMSE = error

    print(f'Custom Linear Regression - W: {W}, b: {b}, R2: {R2}, RMSE: {RMSE}')
    return W, b, R2, RMSE, y_hat


def plot_prediction(y, pred1, pred2):
    X = scaler_x.inverse_transform(x.reshape(-1, 1))
    Y = scaler_y.inverse_transform(y.reshape(-1, 1))
    Pred1 = scaler_y.inverse_transform(pred1.reshape(-1, 1))
    Pred2 = scaler_y.inverse_transform(pred2.reshape(-1, 1))

    plt.figure(figsize=(16, 6))
    plt.plot(X, Y, label='Actual', color='blue')
    plt.plot(X, Pred1, label='Custom Prediction', color='red')
    plt.plot(X, Pred2, label='Sklearn Prediction', color='green')
    plt.xlabel('Year')
    plt.ylabel('Mean Temperature (°C)')
    plt.xticks(np.arange(min(X)[0], max(X)[0]+1, 5))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    custom_result = custom_linearReg()
    sklearn_result = sklearn_linearReg()

    plot_prediction(y, custom_result[-1], sklearn_result[-1])

    df = pd.DataFrame([custom_result[:-1], sklearn_result[:-1]],
                      index=['Custom Linear Regression', 'Sklearn Linear Regression'],
                      columns=['W', 'b', 'R2', 'RMSE'])
    print(df)
