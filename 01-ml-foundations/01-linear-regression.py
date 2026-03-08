import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_lesson():
    # 1. Load the dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # 2. Split the data (80% Train, 20% Test)
    # In statistics, we often use all data to fit a model. 
    # In ML, we must test on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Make Predictions
    y_pred = model.predict(X_test)
    
    # 5. Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # 6. Interpret Coefficients (Bridge from stats)
    # Which features matter most?
    coeffs = pd.DataFrame({'Feature': data.feature_names, 'Coefficient': model.coef_})
    coeffs = coeffs.sort_values(by='Coefficient', ascending=False)
    print("\n--- Model Coefficients ---")
    print(coeffs)

if __name__ == "__main__":
    run_lesson()
