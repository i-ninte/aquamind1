import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Assuming you have the code to load and preprocess the data as well as train the model
def load_data(filename):
    data = pd.read_csv(filename)
    return data

def clean_data(data):
    if 'Unnamed: 6' in data.columns:
        data = data.drop(columns='Unnamed: 6')
    data = data.fillna(data.median())
    data.columns = data.columns.str.lower().str.replace('.', '')
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def build_rf_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

# Load the dataset
data = load_data('aqua_attributes.csv')

# Clean and preprocess the dataset
X_train, X_test, y_train, y_test, scaler = clean_data(data)

# Build the Random Forest model
model = build_rf_model()

# Train the model
model = train_model(model, X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Save the model and scaler using the current version of scikit-learn
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
