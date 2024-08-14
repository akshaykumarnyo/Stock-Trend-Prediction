README for Stock Trend Prediction Application
Stock Trend Prediction
This application predicts stock price trends using Long Short-Term Memory (LSTM) neural networks. Built with Streamlit, this app allows users to visualize historical stock prices and forecasts future trends.

Features
Stock Data Retrieval: Fetch historical stock data from Yahoo Finance.
Data Visualization:
Display closing prices over time.
Show moving averages (100-day and 200-day) along with the closing price.
LSTM Model Training: Utilize an LSTM model to predict future stock prices based on historical data.
Forecast Visualization: Compare actual closing prices with predicted values.
Requirements
Python 3.6+
Required Python libraries:
numpy: For numerical operations.
pandas: For data manipulation.
matplotlib: For plotting charts.
yfinance: For fetching stock data.
keras: For building and training the LSTM model.
sklearn: For data scaling.
streamlit: For building the web interface.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/username/stock-trend-prediction.git
cd stock-trend-prediction
Install the required libraries:

bash
Copy code
pip install numpy pandas matplotlib yfinance keras scikit-learn streamlit
Usage
Run the Streamlit Application:

bash
Copy code
streamlit run app.py
Enter Stock Ticker:

On the web interface, input the stock ticker symbol (e.g., PYPL) to fetch and predict stock prices.
View Visualizations:

Data Summary: Displays basic statistics of the stock data.
Price Charts: Shows closing prices over time, along with 100-day and 200-day moving averages.
Prediction: Compares actual vs. predicted closing prices using the LSTM model.
Customization
Model Parameters:

Modify the LSTM network architecture (e.g., number of layers, units) in the script to experiment with different model configurations.
Adjust the number of epochs and batch size in the model training phase.
Time Steps:

Change the time_step parameter to adjust the look-back period for the LSTM model.
Example

License
This project is licensed under the MIT License. See the LICENSE file for more details.
