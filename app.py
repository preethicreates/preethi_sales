from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return "Sales Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    mrp = data['Item_MRP']

    scaled = scaler.transform([[mrp]])
    prediction = model.predict(scaled)

    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run()