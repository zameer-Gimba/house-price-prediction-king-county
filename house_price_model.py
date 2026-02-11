# house_price_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from given CSV path.
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Clean and prepare dataset for training.
    - Drop rows with missing values
    - Separate features and target
    """

    df = df.dropna()

    if "price" not in df.columns:
        raise ValueError("Target column 'price' not found in dataset.")

    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y


def build_pipeline():
    """
    Create ML pipeline with scaling and linear regression.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    return pipeline


def train_model(X, y):
    """
    Train model and return trained pipeline + metrics.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return pipeline, r2, mse


def main():
    """
    Main execution block.
    """

    data_path = "../data/Kc_house_data_NaN.csv"

    df = load_data(data_path)
    X, y = preprocess_data(df)

    model, r2, mse = train_model(X, y)

    print("Model Training Completed")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")


if __name__ == "__main__":
    main()
