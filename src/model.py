import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def train_model(X_train, y_train):
    """Train a Random Forest model"""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

def save_model(model, filename="outputs/best_model.pkl"):
    """Save trained model"""
    joblib.dump(model, filename)

def load_model(filename="outputs/best_model.pkl"):
    """Load saved model"""
    return joblib.load(filename)
