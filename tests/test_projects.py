import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pytest

df = pd.read_csv("final_report_data/cleaned_weather_data.csv")
def test_data_integrity():
    # Check that the dataset is not empty
    assert df.shape[0] > 0, "Dataset is empty"
    # Check that the required target column is present
    assert "mean_temp" in df.columns, "Target column 'mean_temp' is missing"
    # Check that there are no null values
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
  
def test_data_splitting():
    X = df.drop(columns=['mean_temp'])
    y = df['mean_temp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ensure no overlap between train and test sets
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, "Train and test sets overlap"
    
    # Check the proportions with a tolerance of ±1 row
    expected_test_size = int(0.2 * len(X))
    assert abs(X_test.shape[0] - expected_test_size) <= 1, f"Test set size is incorrect. Expected ~{expected_test_size}, got {X_test.shape[0]}"
  
def test_rf_model_training():
    X = df.drop(columns=['mean_temp'])
    y = df['mean_temp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Ensure predictions match the test set size
    assert len(predictions) == X_test.shape[0], "Predictions size does not match test set"
    # Check that predictions are numeric
    assert all(isinstance(pred, (int, float)) for pred in predictions), "Predictions are not numeric"
