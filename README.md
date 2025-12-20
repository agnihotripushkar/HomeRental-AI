# HomeRental-AI Service

This repository contains the Python-based AI microservice for the HomeRental application. It provides intelligent features to enhance the user experience by leveraging machine learning algorithms.

## Key Features

### 1. AI Price Suggestion
Predicts the estimated rental price of a property based on its features.
- **Model**: Trained Regressor (Random Forest/Linear Regression)
- **Input Features**: `bedroom`, `bathroom`, `latitude`, `longitude`
- **Output**: Estimated price in USD
- **Endpoint**: `POST /predict`

### 2. Recommendation Engine
Provides personalized property recommendations using Nearest Neighbors algorithms.

#### a. Similar Properties
Suggests properties that are similar to a specific listing.
- **Logic**: Content-based filtering using `bedroom`, `bathroom`, `price`, `latitude`, and `longitude`.
- **Endpoint**: `GET /recommend/similar/<property_id>`

#### b. Nearby Properties
Finds properties located geographically close to a given location.
- **Logic**: Spatial constraints using Euclidean distance on `latitude` and `longitude`.
- **Endpoint**: `GET /recommend/nearby?lat=<lat>&lng=<lng>`

## Tech Stack

The microservice is built using the following technologies:
- **Programming Language**: Python
- **Web Framework**: Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Database Integration**: PyMongo (MongoDB)
- **Server**: Gunicorn (for production)
- **Deployment**: Heroku

## Deploying to Heroku

The AI service is a Python Flask app. It must be deployed as a separate Heroku app.

### 1. Create Heroku App
Create a separate app on Heroku (e.g., `home-rental-ml`).

### 2. Add Remote
Add the Heroku git remote (give it a unique name like `heroku-ml`):
```bash
git remote add heroku-ml https://git.heroku.com/YOUR_ML_APP_NAME.git
```

### 3. Deploy
Run this command from the **root** of your repository:
```bash
git subtree push --prefix ml_service heroku-ml master
```

### 4. Verification
Your service should be accessible at: `https://YOUR_ML_APP_NAME.herokuapp.com/health` (returns `{"status":"healthy"}`).
