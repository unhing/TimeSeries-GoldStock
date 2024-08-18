#%% Import Libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from TimeSeries.TimeSeries_Function import *
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#%% Some configs
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16

#%% Load Dataset
df = pd.read_csv('./TimeSeries/data/goldstock.csv', index_col="Date", parse_dates=True)
df = df.drop(columns='Unnamed: 0')

#%% Data Checking
print(df.info())
print("\n")
print(df.describe())

#%% Plot Close Price Stock Chart
plt.plot(df['Close'])
plt.xlabel("Năm")
plt.ylabel("Giá đóng cửa")
plt.title("Giá đóng cửa")
plt.show()

#%% Handle big data
# Giá dao động lớn nên cần chuyển về logarit
df_close = np.log(df['Close'])

#%% - Visualize data after log
plt.plot(df_close)
plt.xlabel("Năm")
plt.ylabel("Giá đóng cửa")
plt.title("Giá đóng cửa sau khi biến đổi logarit")
plt.show()

#%% Roll the data
rolling_data(df_close)

#%% Biểu đồ phân ra chuỗi thời gian
decompose_data(df_close)

#%% Kiểm định tính dừng của dữ liệu
print(adf_test(df_close))
print("---"*5)
print(kpss_test(df_close))

#%% - Kiểm tra tính mùa vụ
plot_acf(df_close)
plt.show()

#%% Kiểm định tự tương quan
correlation_plot(df_close)


#%% Split data
train_data, test_data = df_close[:int(len(df_close)*0.8)], df_close[int(len(df_close)*0.8):]
plt.plot(train_data, 'blue', label='Tập huấn luyện')
plt.plot(test_data, 'red', label='Tập kiểm tra')
plt.xlabel("Năm")
plt.ylabel("Giá đóng cửa")
plt.title("Giá đóng cửa sau khi chia tập dữ liệu")
plt.legend()
plt.show()

# %% - ARIMA
# Xác định tham số p, d, q cho mô hình ARIMA (chạy tự động)
stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
order = stepwise_fit.order
stepwise_fit.plot_diagnostics(figsize=(15, 8))
plt.show()

# %% - Fit model
model = ARIMA(train_data, order=order, trend="t")  # order là chỉ số p d q ở trên
fitted = model.fit()
print(fitted.summary())

#%%
fc = fitted.get_forecast(len(test_data))
fc_series = fc.predicted_mean
fc_series.index = test_data.index
conf = fc.conf_int(alpha=0.05)
lower_series = conf['lower Close']
lower_series.index = test_data.index
upper_series = conf['upper Close']
upper_series.index = test_data.index

# %%
mse = mean_squared_error(test_data, fc_series)
print('ARIMA Model Test MSE: ' + str(mse))
rmse = np.sqrt(mse)
print('ARIMA Model Test RMSE: ' + str(rmse))
mae = mean_absolute_error(test_data, fc_series)
print('ARIMA Model Test MAE: ' + str(mae))

# %% - Plot actual vs predicted values
plt.plot(train_data, label="Tập huấn luyện")
plt.plot(test_data, color="orange", label="Giá đóng cửa thực tế")
plt.plot(fc_series, color="red", label="Giá đóng cửa dự đoán")
plt.fill_between(lower_series.index, lower_series, upper_series, color="blue", alpha=.2)
plt.title("Dự đoán giá đóng cửa (ARIMA)")
plt.xlabel("Thời gian")
plt.ylabel("Giá")
plt.legend(loc='upper left', fontsize=16)
plt.show()


# %% - HOLT-WINTERS
# Fit Holt-Winters model
model_hw = ExponentialSmoothing(train_data, trend='add')
fitted_hw = model_hw.fit()
print(fitted_hw.summary())

# %% - Make predictions using Holt-Winters
hw_forecast = fitted_hw.forecast(len(test_data))
hw_forecast_series = pd.Series(hw_forecast.values, index=test_data.index)

# %% - Evaluate performance for Holt-Winters
hw_mse = mean_squared_error(test_data, hw_forecast)
print('Holt-Winters Model Test MSE: ' + str(hw_mse))
hw_rmse = np.sqrt(hw_mse)
print('Holt-Winters Model Test RMSE: ' + str(hw_rmse))
hw_mae = mean_absolute_error(test_data, hw_forecast)
print('Holt-Winters Model Test MAE: ' + str(hw_mae))


# %% - Plot actual vs Holt-Winters predicted values
plt.plot(train_data, label="Tập huấn luyện")
plt.plot(test_data, color="orange", label="Giá đóng cửa thực tế")
plt.plot(hw_forecast_series, color="red", label="Giá đóng cửa dự đoán")
plt.title("Dự đoán giá đóng cửa (Holt-Winters)")
plt.xlabel("Thời gian")
plt.ylabel("Giá")
plt.legend(loc='upper left', fontsize=16)
plt.show()

# %% - Save model
with open('./TimeSeries/models/TTS_ARIMA', 'wb') as file:
    pickle.dump(fitted, file)

with open('./TimeSeries/models/TTS_HW', 'wb') as file:
    pickle.dump(fitted_hw, file)

