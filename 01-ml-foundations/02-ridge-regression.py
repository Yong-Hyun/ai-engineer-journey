import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def compare_models():
    # 1. Load the dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Standard Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # 4. Train Ridge Regression
    # 'alpha' is the regularization strength. Higher alpha = stronger penalty.
    ridge = Ridge(alpha=1.0) 
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    # 5. Compare Results
    print("\n--- Model Comparison ---")
    print(f"{'Metric':<25} | {'Linear Regression':<20} | {'Ridge Regression':<20}")
    print("-" * 75)
    
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    print(f"{'Mean Squared Error (MSE)':<25} | {mse_lr:<20.4f} | {mse_ridge:<20.4f}")
    
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"{'R-squared Score':<25} | {r2_lr:<20.4f} | {r2_ridge:<20.4f}")

    # 6. Compare Coefficients
    print("\n--- Coefficient Comparison (First 5 Features) ---")
    print(f"{'Feature':<15} | {'Linear Coeff':<15} | {'Ridge Coeff':<15}")
    print("-" * 50)
    for i in range(5):
        print(f"{data.feature_names[i]:<15} | {lr.coef_[i]:<15.4f} | {ridge.coef_[i]:<15.4f}")

if __name__ == "__main__":
    compare_models()
