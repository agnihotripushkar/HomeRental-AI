# HomeRental-AI Service

This repository contains the Python-based AI microservice for the HomeRental application.

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
