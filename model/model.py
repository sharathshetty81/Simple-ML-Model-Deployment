import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_save_model():
    """Train a simple model on the Iris dataset and save it"""
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature and target information
    model_info = {
        "feature_names": feature_names,
        "target_names": target_names
    }
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_info.joblib")
    joblib.dump(model_info, info_path)
    print(f"Model info saved to {info_path}")
    
    return model, model_info

if __name__ == "__main__":
    train_and_save_model()

