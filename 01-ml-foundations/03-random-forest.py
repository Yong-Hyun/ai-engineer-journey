import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def run_random_forest():
    # 1. Load the dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Random Forest Model
    # n_estimators: 나무의 개수 (많을수록 일반적으로 좋지만 느려짐)
    # max_depth: 각 나무의 최대 깊이 (과적합 방지)
    print("Training Random Forest (this might take a few seconds)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 4. Make Predictions
    y_pred = rf.predict(X_test)
    
    # 5. Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Random Forest Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print("(참고: Linear Regression의 R-squared는 약 0.57이었습니다.)")
    
    # 6. Feature Importance (AI 엔지니어링의 핵심!)
    # 어떤 특징이 가장 중요한지 시각화 정보 출력
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': data.feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Feature Importances ---")
    print(feature_importance_df)

if __name__ == "__main__":
    run_random_forest()
