import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import models, layers



def main(csv_path='Exam_Score_Prediction.csv'):
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Load dataset
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Inspect data
    print('\n=== Data head ===')
    print(df.head())

    print('\n=== Data info ===')
    df.info()

    print('\n=== Missing values per column ===')
    print(df.isnull().sum())

    print('\n=== Duplicate rows ===')
    print(f"Duplicates: {df.duplicated().sum()}")

    # 3. Clean data: drop duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print('Dropped duplicate rows.')

    # If any missing values remain, drop them (dataset typically has none)
    if df.isnull().any().any():
        print('Missing values found — dropping rows with missing values.')
        df = df.dropna().reset_index(drop=True)

    # 4. Encode categorical features using one-hot encoding
    # Drop student_id as it's just an identifier
    df = df.drop(columns=['student_id'])
    
    categorical_cols = [
        'gender',
        'course',
        'internet_access',
        'sleep_quality',
        'study_method',
        'facility_rating',
        'exam_difficulty',
    ]

    # Ensure columns exist (defensive)
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # 5. Define features and target
    if 'exam_score' not in df.columns:
        raise KeyError("Target column 'exam_score' not found in CSV. Check the file.")

    X = df.drop(columns=['exam_score'])
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    y = df['exam_score']

    print(f"Feature matrix shape after encoding: {X.shape}")

    # 6 & 7. Normalize features and split data into train/val/test
    # Split: train 70%, validation 15%, test 15%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 8. Build the Keras model
    input_dim = X_train_scaled.shape[1]
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear'),
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # 9. Train the model for 100 epochs with validation data
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        validation_data=(X_val_scaled, y_val),
        verbose=2,
    )

    # 10. Evaluate on test set
    y_pred = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print('\n=== Test set evaluation ===')
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # 11a. Plot training vs validation loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.tight_layout()
    loss_plot_path = 'loss_curve.png'
    plt.savefig(loss_plot_path)
    print(f"Saved loss curve to {loss_plot_path}")

    # 11b. Scatter plot of true vs predicted exam_scores with y=x line
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    plt.xlabel('True Exam Score')
    plt.ylabel('Predicted Exam Score')
    plt.title('True vs Predicted Exam Scores')
    plt.tight_layout()
    scatter_plot_path = 'true_vs_pred.png'
    plt.savefig(scatter_plot_path)
    print(f"Saved true vs predicted plot to {scatter_plot_path}")


if __name__ == '__main__':
    # Use default CSV in the repo root; change path if necessary
    csv_file = 'Exam_Score_Prediction.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at {csv_file}. Ensure the file is in the repo root.")
    main(csv_file)
