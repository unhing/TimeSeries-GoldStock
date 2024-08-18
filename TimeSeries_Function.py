import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose


def adf_test(data):
    indices = ['ADF: Test statistics','p-value', '# of Lags', '# of observations']
    test = adfuller(data, autolag='AIC')
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical Value ({key})"] = value

    if results[1] <= 0.05:
        print("Reject the null hypothesis H0.\n The data is stationary")
    else:
        print("Fail to reject the null hypothesis reject H0.\n The data is non-stationary")

    return results


def kpss_test(data):
    indices = ['KPSS: Test statistics', 'p value', '# of lags']
    kpss_test = kpss(data)
    results = pd.Series(kpss_test[:3], index=indices)
    for key, value in kpss_test[3].items():
        results[f"Critical Value ({key})"] = value

    if results[1] <= 0.05:
        print("Reject the null hypothesis H0.\n The data is non-stationary")
    else:
        print("Fail to reject the null hypothesis reject H0.\n The data is stationary")

    return results


def rolling_data(data):
    roll_mean = data.rolling(window=30).mean()
    roll_std = data.rolling(window=30).std()
    plt.plot(data, 'blue', label='Giá đóng cửa')
    plt.plot(roll_mean, 'red', label='Trung bình trượt')
    plt.plot(roll_std, 'green', label='Độ lệch chuẩn trượt')
    plt.title("Trung bình trượt & Độ lệch chuẩn trượt (trong 30 ngày)")
    plt.legend()
    plt.show()


def decompose_data(data):
    decompose_result = seasonal_decompose(data, model='multiplicative', period=30)
    decompose_result.plot()
    plt.show()


def correlation_plot(data):
    pd.plotting.lag_plot(data)
    plt.title("Kiểm định tự tương quan của giá đóng cửa")
    plt.show()
