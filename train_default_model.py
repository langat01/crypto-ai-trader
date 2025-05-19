import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Create a simple dummy model for demonstration
def train_dummy_model():
    print("Training default fallback model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Train on dummy data (in a real app, you'd use real data)
    X_dummy = np.random.rand(100, 7)  # 7 features matching our feature set
    y_dummy = np.random.randint(0, 2, 100)  # Binary classification
    model.fit(X_dummy, y_dummy)
    return model

def ensure_models_directory():
    """Create models directory if needed"""
    Path("models").mkdir(exist_ok=True)

def save_default_model():
    ensure_models_directory()
    model = train_dummy_model()
    joblib.dump(model, "models/default_model.pkl")
    print("Default model saved to models/default_model.pkl")

if __name__ == "__main__":
    save_default_model()
