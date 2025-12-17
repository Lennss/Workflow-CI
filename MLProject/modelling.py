import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set MLflow
if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow.set_experiment("Pima-Diabetes-Basic")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Pima-Diabetes-Basic")

# Load dataset hasil preprocessing
X = pd.read_csv("diabetes_preprocessing/X_preprocessed.csv")
y = pd.read_csv("diabetes_preprocessing/y_preprocessed.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.2, random_state=42
)

# Training + autolog
with mlflow.start_run():
    mlflow.autolog()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
