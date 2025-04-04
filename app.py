from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


model = load_model('stock_prediction_lstm_model.h5')

app = Flask(__name__)

# Function to fetch and preprocess the stock data
def get_stock_data(symbol):
    stock_data = yf.download(symbol, period='5d', interval='1d')  # Get last 5 days of data
    return stock_data

# Function to preprocess the data 
def preprocess_data(input_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(input_data)
    return scaled_data, scaler  # Return both scaled data and the scaler object

# Route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the stock symbol from the form
        symbol = request.form['symbol']
        
        # Fetch the stock data
        stock_data = get_stock_data(symbol)
        
        # Get today's price (close price of the most recent day)
        todays_price_display = float(stock_data['Close'].iloc[-1])  # Ensure it's a float
        todays_price = f"${symbol} {todays_price_display:.2f}" 

        # Preprocess the data (scale it)
        data = stock_data[['Close']].values  # We are using closing prices for prediction
        processed_data, scaler = preprocess_data(data)  # Get the scaler too
        
        # Make the prediction (using the last data point for prediction)
        prediction = model.predict(processed_data[-1].reshape(1, -1, 1))
        
        # Unscale the predicted price
        predicted_price = scaler.inverse_transform([[prediction[0][0]]])[0][0]  # Unscale the prediction
        predicted_price = round(predicted_price, 2)  # Round predicted price to 2 decimal places
        
        # Today's date
        today_date = stock_data.index[-1].strftime('%Y-%m-%d')
        
        # Render the template with results
        return render_template('index.html', 
                               symbol=symbol, 
                               todays_price=todays_price, 
                               predicted_price=predicted_price,
                               today_date=today_date)

if __name__ == '__main__':
    app.run(debug=True)
