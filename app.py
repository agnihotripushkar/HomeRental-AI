from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'price_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received prediction request:", data)
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model not trained yet"}), 503

        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Prepare input dataframe
        # Expecting: bedroom, bathroom, latitude, longitude
        input_data = pd.DataFrame([{
            'bedroom': data.get('bedroom'),
            'bathroom': data.get('bathroom'),
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude')
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            "estimated_price": round(prediction, 2),
            "currency": "USD" # Assuming USD for now
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
