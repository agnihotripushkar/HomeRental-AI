import os
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

def train_model():
    mongo_url = os.getenv("DATABASE_URL")
    if not mongo_url:
        print("Error: DATABASE_URL not found in environment")
        return

    # Connect to MongoDB
    # Note: Prisma connection strings might not work directly with PyMongo if they have special params
    # But usually for MongoDB Atlas it is fine.
    try:
        client = MongoClient(mongo_url)
        db = client.get_database() # Gets the default database from URL
        posts_collection = db['Post']
        
        # Fetch data
        print("Fetching data from MongoDB...")
        cursor = posts_collection.find({}, {
            'price': 1, 
            'bedroom': 1, 
            'bathroom': 1, 
            'latitude': 1, 
            'longitude': 1, 
            '_id': 0
        })
        
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            print("No data found in Post collection to train on.")
            return

        print(f"Found {len(df)} records.")
        
        # Clean data
        df = df.dropna()
        
        # Features and Target
        X = df[['bedroom', 'bathroom', 'latitude', 'longitude']]
        y = df['price']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # Initialize and Train Model
        print("Training Random Forest Regressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Model Trained. MAE: {mae}")
        
        # Save Model
        output_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, 'price_model.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Model saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_model()
