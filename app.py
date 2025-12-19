from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
from recommendation_engine import RecommendationEngine

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'price_model.pkl')

# Initialize Recommendation Engine
recommendation_engine = RecommendationEngine()

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

@app.route('/recommend/similar/<property_id>', methods=['GET'])
def recommend_similar(property_id):
    try:
        similar_ids = recommendation_engine.get_similar_properties(property_id)
        return jsonify({"similar_properties": similar_ids})
    except Exception as e:
        print(f"Error recommending similar properties: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend/nearby', methods=['GET'])
def recommend_nearby():
    try:
        lat = request.args.get('lat')
        lng = request.args.get('lng')
        
        if not lat or not lng:
            return jsonify({"error": "lat and lng parameters are required"}), 400

        nearby_ids = recommendation_engine.get_nearby_properties(lat, lng)
        return jsonify({"nearby_properties": nearby_ids})
    except Exception as e:
        print(f"Error recommending nearby properties: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
