# stock-price-prediction
Stock Market Price Prediction with LSTM

Overview

This project predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock data retrieved from Alpha Vantage.

Features

Fetches stock data using Alpha Vantage API.

Preprocesses and scales data for LSTM modeling.

Trains an LSTM model for future stock price predictions.

Deploys as a Flask API for real-time predictions.

Installation

1. Clone the Repository

git clone https://github.com/Mwangimark/cnnPrediction.git
cd cnnPrediction

2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate    # On Windows

3. Install Dependencies

pip install -r requirements.txt

4. Set Up API Key

Create a .env file in the root directory and add:

ALPHA_VANTAGE_API_KEY=your_actual_api_key

Usage

1. Run the Jupyter Notebook

Execute stock_prediction.ipynb to train and evaluate the model.

2. Run the Flask App (For Deployment)

python app.py

Access the app at: http://127.0.0.1:5000/

Deployment

Render Deployment Steps

Push the latest code to GitHub.

Log in to Render and create a new web service.

Connect your GitHub repo and set up environment variables (ALPHA_VANTAGE_API_KEY).

Deploy the Flask app.

Contributing

Feel free to submit pull requests or raise issues for improvements!

License

This project is licensed under the MIT License.

Author: Mark Mwangimark

