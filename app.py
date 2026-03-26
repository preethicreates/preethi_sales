from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Home route (HTML page)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction from HTML form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        mrp = float(request.form['Item_MRP'])
        scaled = scaler.transform([[mrp]])
        prediction = model.predict(scaled)

        return render_template('index.html', prediction=round(prediction[0], 2))
    
    except Exception as e:
        return f"Error: {str(e)}"

# Optional API (JSON)
@app.route('/api_predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        mrp = float(data['Item_MRP'])
        scaled = scaler.transform([[mrp]])
        prediction = model.predict(scaled)

        return jsonify({'prediction': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run()
