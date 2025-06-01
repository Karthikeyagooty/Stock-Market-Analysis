from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('stock_model.pkl')
scaler = joblib.load('scaler.pkl')

# List your features in the order used for training
features = [
    'Prev_Close', 'Prev_Open', 'Prev_High', 'Prev_Low',
    'Open', 'High', 'Low', 'Adj Close', 'Volume',
    'Daily_Return', 'MA7', 'MA14', 'Price_Range', 'Volume_Change'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_scaled = scaler.transform(input_df[features])
    prediction = model.predict(input_scaled)[0]
    return jsonify({'predicted_close': prediction})

if __name__ == "__main__":
    app.run(debug=True)
