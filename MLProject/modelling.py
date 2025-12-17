import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Set MLFlow
if os.getenv("GITHUB_ACTIONS") != "true":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Pima-Diabetes-Basic")

# Load data
X = pd.read_csv("diabetes_preprocessing/X_preprocessed.csv")
y = pd.read_csv("diabetes_preprocessing/y_preprocessed.csv")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Manual Log
mlflow.log_param("model_type", "RandomForestClassifier")
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", accuracy)

# Log model sebagai artifact
mlflow.sklearn.log_model(model, "model")

print("Training selesai. Accuracy:", accuracy)
