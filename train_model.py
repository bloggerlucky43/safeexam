import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_liveness_model():
    os.makedirs('models', exist_ok=True)

    try:
        df = pd.read_csv('data/biometric_features.csv')
    except FileNotFoundError:
        print("Dataset not found! Please run generate_dataset.py first.")
        return

    # 1. Feature Extraction
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Classifier Training
    print("Training Random Forest Classifier on Multi-Modal Features...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Validation
    y_pred = clf.predict(X_test)
    print("\n--- Random Forest Liveness Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Spoof", "Live"]))

    # 4. Export
    joblib.dump(clf, 'models/liveness_ai.joblib')
    print("Model dynamically trained and saved to models/liveness_ai.joblib")

if __name__ == "__main__":
    train_liveness_model()
