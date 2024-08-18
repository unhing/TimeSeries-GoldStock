# Full project report

This is the Time Series section that I have been in charge of for the **Data Analysis with R-Python - Final Project**.

You can view the full project report (in Vietnamese) [HERE](https://drive.google.com/file/d/1NZYLd4mMDrIICOmgaYhamUZ9TrqRUiZv/view?usp=drive_link).

# About the dataset
The goldstock dataset includes daily recorded values of gold prices from 22/01/2014, to 19/01/2024, providing detailed information about the fluctuations in gold prices over that period.

The dataset contains 2,511 rows and 5 columns with no null values.
- **Close**: The closing price of gold on the corresponding date.
- **Volume**: Information about the trading volume of gold.
- **Open**: The opening price of gold on the corresponding date.
- **High**: The highest price of gold on the corresponding date.
- **Low**: The lowest price of gold on the corresponding date.

# Time Series Analysis Approaches
Three different approaches were used to analyze and forecast the gold prices. The performance of the models was evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## 1. Train-Test Split (8:2)
   
**Description:**

- The dataset is split into a training set (80%) and a testing set (20%).
- Two forecasting models are applied: ARIMA and Holt-Winters.
- Both models are trained on the training set and evaluated on the testing set.
  
**Evaluation:** MSE, MAE, and RMSE are calculated on the testing set for both ARIMA and Holt-Winters models.

## 2. Sliding Window Approach

**Description:**

- A sliding window approach is used where a fixed-size window moves across the time series data.
- Both ARIMA and Holt-Winters models are applied within each window.
- The window size is predefined, and the model is trained and evaluated on each window.

**Evaluation:**

- Approach 1: Average of Metrics Across Windows
  
  - MSE, MAE, and RMSE are calculated for each window.
  - The metrics are averaged across all windows to get overall performance.
  
- Approach 2: Best Window Evaluation
  - The dataset is split into a training set (80%) and a testing set (20%).
  - The sliding window approach is applied on the training set.
  - The best window is identified based on performance metrics (MSE, MAE, RMSE) on the training set.
  - The performance of the chosen window is evaluated on the testing set.
  
## 3. Expanding Window Approach

**Description:**

- In the expanding window approach, the window starts small and grows as more data is included in the training set over time.
- Both ARIMA and Holt-Winters models are applied as the window expands.

**Evaluation:**

- Approach 1: Average of Metrics Across Windows
  
  - MSE, MAE, and RMSE are calculated for each expanding window.
  - The metrics are averaged across all expanding windows to get overall performance.
  
- Approach 2: Best Window Evaluation
  
  - The dataset is split into a training set (80%) and a testing set (20%).
  - The expanding window approach is applied on the training set.
  - The best expanding window is identified based on performance metrics (MSE, MAE, RMSE) on the training set.
  - The performance of the chosen expanding window is evaluated on the testing set.
 
# Summary of Metrics
After training and validating the models, the team calculated the predictive performance using MSE, RMSE, and MAE metrics. The lower these metrics, the more accurate the model.

- MSE (Mean Squared Error): Measures the average squared difference between observed and predicted values.
- MAE (Mean Absolute Error): Measures the average absolute difference between observed and predicted values.
- RMSE (Root Mean Squared Error): Measures the square root of the average squared difference between observed and predicted values, providing error magnitude.

The table below is our final result:

| Approach | Model | MSE     | RMSE    | MAE     |
|------------|--------|---------|---------|---------|
| Train-Test Split (8:2) | ARIMA   | 0.00264 | 0.05141 | 0.03811 |
| Train-Test Split (8:2) | Holt-Winters | 0.00268 | 0.05173 | 0.03832 |
| Sliding Window (1) | ARIMA   | 0.00964 | 0.08669 | 0.07832 |
| Sliding Window (1) | Holt-Winters | 0.00955 | 0.08620 | 0.07784 |
| Sliding Window (2) | ARIMA   | 0.00671 | 0.08189 | 0.06564 |
| Sliding Window (2) | Holt-Winters | 0.00726 | 0.08521 | 0.06865 |
| Expanding Window (1) | ARIMA   | 0.01794 | 0.11611 | 0.10004 |
| Expanding Window (1) | Holt-Winters | 0.01802 | 0.11656 | 0.10042 |
| Expanding Window (2) | ARIMA   | 0.04944 | 0.22236 | 0.19490 |
| Expanding Window (2) | Holt-Winters | 0.04952 | 0.22253 | 0.19506 |

# Conclusion

Based on the obtained results, the following conclusions can be drawn:

- **Train-Test Split Method**: The ARIMA model demonstrated the highest accuracy with the lowest MSE, RMSE, and MAE values. This indicates that ARIMA is the most suitable model for this dataset when using a traditional train-test split approach.

- **Sliding Window Approach:**
  
  - **Method 2** provided better results compared to Method 1. For Method 1, the Holt-Winters model performed slightly better, but the differences in metrics compared to the ARIMA model were not significant.
  - However, since the dataset lacks seasonality, using the Holt-Winters model might not be as optimal as the ARIMA model.

- **Expanding Window Approach:**
  
  - **Method 2** did not perform as well as Method 1. This could be attributed to the presence of a spike in the 20% test set compared to the best window test set in the training phase. This result suggests that when the data exhibits unusual fluctuations, the expanding window approach may not accurately reflect the model’s expected performance.

- **Overall Comparison:**

  - While the sliding window and expanding window methods can provide the most optimal results, the spike in the 20% test set led to these methods not performing as well as the traditional train-test split method.
  - Despite this, using the sliding and expanding window approaches still offers value in assessing the stability and adaptability of the model in different scenarios.
