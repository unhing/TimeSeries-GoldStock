#%% Import Libraries
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from TimeSeries.TimeSeries_Function import *
import pickle
import numpy as np
import time
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

#%% Roll data
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


#%% Define variables
window_size = 500
test_size = 100
count_loop = 0


# %% - ARIMA
# Lists to store evaluation metrics for each fold
mse_list = []
rmse_list = []
mae_list = []

# %% - Run ARIMA

for i in range(window_size, len(df_close), (window_size + test_size)):

    # Define rolling window indices
    train_start = i - window_size
    train_end = i
    test_start = i + 1
    test_end = test_start + test_size

    # Extract training and test data
    train_data = df_close.iloc[train_start:train_end]
    test_data = df_close.iloc[test_start:test_end]

    print(f"Fold {i+1}:")
    print("TRAIN: ", train_data)
    print("TEST: ", test_data)

    # Xác định tham số p, d, q cho mô hình ARIMA (chạy tự động)
    model_arima = auto_arima(train_data, trace=True, suppress_warnings=True)
    order = model_arima.order
    seasonal_order = model_arima.seasonal_order
    model_arima.plot_diagnostics(figsize=(15, 8))
    plt.show()

    time.sleep(2)

    model = ARIMA(train_data, order=order, seasonal_order=seasonal_order, trend='t')  # order là chỉ số p d q ở trên
    fitted = model.fit()
    print(fitted.summary())

    fc = fitted.get_forecast(len(test_data))
    fc_series = fc.predicted_mean
    fc_series.index = test_data.index
    conf = fc.conf_int(alpha=0.05)
    lower_series = conf['lower Close']
    lower_series.index = test_data.index
    upper_series = conf['upper Close']
    upper_series.index = test_data.index

    count_loop += 1

    # Visualization
    plt.plot(df_close, color="black")
    plt.plot(train_data, color="blue", label="Tập huấn luyện")
    plt.plot(test_data, color="orange", label="Giá đóng cửa thực tế")
    plt.plot(test_data.index, fc_series, color="red", label="Giá đóng cửa dự đoán")
    plt.fill_between(test_data.index, lower_series, upper_series, color="blue", alpha=0.3)
    plt.title("ARIMA Rolling Window " + str(count_loop))
    plt.xlabel("Thời gian")
    plt.ylabel("Giá")
    plt.legend(loc="upper left")
    plt.show()

    # Compute evaluation metrics
    mse = mean_squared_error(test_data, fc_series)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, fc_series)

    # Append to lists
    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)

    # Save model
    with open(f'./TimeSeries/models/RW_ARIMA_W{count_loop}', 'wb') as file:
        pickle.dump(fitted, file)

# %% - HOLT-WINTERS
# Lists to store evaluation metrics for Holt-Winters
hw_count_loop = 0

hw_mse_list = []
hw_rmse_list = []
hw_mae_list = []

# %% - Run Holt-Winters

for i in range(window_size, len(df_close), (window_size + test_size)):

    # Define rolling window indices
    train_start = i - window_size
    train_end = i
    test_start = i + 1
    test_end = test_start + test_size

    # Extract training and test data
    train_data = df_close.iloc[train_start:train_end]
    test_data = df_close.iloc[test_start:test_end]

    print(f"Fold {i+1}:")
    print("TRAIN: ", train_data)
    print("TEST: ", test_data)

    # Fit Holt-Winters model
    model_hw = ExponentialSmoothing(train_data, trend='add')
    fitted_hw = model_hw.fit()
    print(fitted_hw.summary())

    # Make predictions using Holt-Winters
    hw_forecast = fitted_hw.forecast(len(test_data))
    hw_forecast_series = pd.Series(hw_forecast.values, index=test_data.index)

    hw_count_loop += 1

    plt.plot(df_close, color="black")
    plt.plot(train_data, label="Tập huấn luyện")
    plt.plot(test_data, color="orange", label="Giá đóng cửa thực tế")
    plt.plot(hw_forecast_series, color="red", label="Giá đóng cửa dự đoán")
    plt.title("Holt-Winters Rolling Window " + str(hw_count_loop))
    plt.xlabel("Thời gian")
    plt.ylabel("Giá")
    plt.legend(loc='upper left')
    plt.show()

    # Compute evaluation metrics
    hw_mse = mean_squared_error(test_data, hw_forecast)
    hw_rmse = np.sqrt(hw_mse)
    hw_mae = mean_absolute_error(test_data, hw_forecast)

    # Append to lists
    hw_mse_list.append(hw_mse)
    hw_rmse_list.append(hw_rmse)
    hw_mae_list.append(hw_mae)

    # Save model
    with open(f'./TimeSeries/models/RW_HW_W{hw_count_loop}', 'wb') as file:
        pickle.dump(fitted_hw, file)

#%% Print evaluation metrics for each fold (ARIMA)
print("\nEvaluation Metrics for ARIMA:")
for i, (mse, rmse, mae) in enumerate(zip(mse_list, rmse_list, mae_list)):
    print(f"Window {i+1}:")
    print(f"  MSE: {mse}")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")

print("\nMean of Evaluation Metrics (ARIMA):")
print(f"  Mean MSE: {np.mean(mse_list)}")
print(f"  Mean RMSE: {np.mean(rmse_list)}")
print(f"  Mean MAE: {np.mean(mae_list)}")


#%% Print evaluation metrics for each fold (Holt-Winters)
print("\nEvaluation Metrics for Holt-Winters:")
for i, (hw_mse, hw_rmse, hw_mae) in enumerate(zip(hw_mse_list, hw_rmse_list, hw_mae_list)):
    print(f"Window {i+1}:")
    print(f"  MSE: {hw_mse}")
    print(f"  RMSE: {hw_rmse}")
    print(f"  MAE: {hw_mae}")

print("\nMean of Evaluation Metrics (Holt-Winters):")
print(f"  Mean MSE: {np.mean(hw_mse_list)}")
print(f"  Mean RMSE: {np.mean(hw_rmse_list)}")
print(f"  Mean MAE: {np.mean(hw_mae_list)}")
