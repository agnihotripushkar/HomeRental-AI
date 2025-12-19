import os
import pandas as pd
from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

class RecommendationEngine:
    def __init__(self):
        self.df = None
        self.model_similar = None
        self.model_nearby = None
        self.scaler = StandardScaler()
        self._load_data_and_train()

    def _load_data_and_train(self):
        mongo_url = os.getenv("DATABASE_URL")
        if not mongo_url:
            print("Warning: DATABASE_URL not set. Recommendation engine will be empty.")
            return

        try:
            client = MongoClient(mongo_url)
            db = client.get_database()
            posts_collection = db['Post']

            # Fetch relevant fields
            cursor = posts_collection.find({}, {
                'price': 1,
                'bedroom': 1,
                'bathroom': 1,
                'latitude': 1,
                'longitude': 1,
                '_id': 1 # Keep ID to map back
            })

            data = list(cursor)
            if not data:
                print("Warning: No data found for recommendations.")
                return

            self.df = pd.DataFrame(data)
            self.df['_id'] = self.df['_id'].astype(str) # Ensure IDs are strings
            
            # Drop rows with missing values for critical features
            self.df = self.df.dropna(subset=['price', 'bedroom', 'bathroom', 'latitude', 'longitude'])

            if self.df.empty:
                print("Warning: No valid data after dropping NaNs.")
                return

            # --- Model 1: Similarity (Content-Based) ---
            # Features: bedroom, bathroom, price, latitude, longitude
            # We standardize because price and coords have very different scales
            features_similar = self.df[['bedroom', 'bathroom', 'price', 'latitude', 'longitude']]
            self.features_scaled = self.scaler.fit_transform(features_similar)
            
            self.model_similar = NearestNeighbors(n_neighbors=6, algorithm='auto') # k=6 because query item itself is usually the 1st match
            self.model_similar.fit(self.features_scaled)

            # --- Model 2: Nearby (Geospatial) ---
            # Features: latitude, longitude
            # Simple Euclidean on lat/lon is "okay" for small distances, but haversine is better. 
            # However, for simplicity and common usage (and since scikit-learn haversine requires radians),
            # we will use Euclidean for now or switch to simple filtering if needed. 
            # Let's stick to NearestNeighbors with lat/lon.
            features_nearby = self.df[['latitude', 'longitude']]
            self.model_nearby = NearestNeighbors(n_neighbors=10, algorithm='auto')
            self.model_nearby.fit(features_nearby)
            
            print(f"Recommendation Engine initialized with {len(self.df)} properties.")

        except Exception as e:
            print(f"Error initializing Recommendation Engine: {e}")

    def get_similar_properties(self, property_id, k=5):
        if self.df is None or self.model_similar is None:
            return []

        # Find index of the property
        idx = self.df.index[self.df['_id'] == property_id].tolist()
        if not idx:
            return []
        
        query_idx = idx[0]
        # Query the model
        distances, indices = self.model_similar.kneighbors([self.features_scaled[query_idx]])
        
        # indices[0] contains the indices of neighbors. 
        # The first one is the item itself, so we skip it (unless distance is not 0, which shouldn't happen for the item itself)
        neighbor_indices = indices[0][1:k+1] 
        
        similar_ids = self.df.iloc[neighbor_indices]['_id'].tolist()
        return similar_ids

    def get_nearby_properties(self, lat, lon, k=5):
        if self.df is None or self.model_nearby is None:
            return []

        try:
            lat = float(lat)
            lon = float(lon)
        except ValueError:
            return []

        # Query the model
        distances, indices = self.model_nearby.kneighbors([[lat, lon]])
        
        neighbor_indices = indices[0][:k] # Here we don't necessarily skip the first one if the query point isn't exactly an existing property
        
        nearby_ids = self.df.iloc[neighbor_indices]['_id'].tolist()
        return nearby_ids
