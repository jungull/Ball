import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# --- Configuration ---
PLAYER_FEATURE_FILE = "nba_games_player_features_v2.csv"
MODEL_OUTPUT_FILE = "nba_model_v5_player_powered.joblib"
TEST_SIZE = 0.2

# --- Main Script ---
print("--- Training NBA Model V5 (Player-Powered) ---")
try:
    df = pd.read_csv(PLAYER_FEATURE_FILE)
    
    target = 'point_differential'
    features = [col for col in df.columns if '_diff' in col]
    
    X = df[features]
    y = df[target]

    split_index = int(len(df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training on {len(features)} player-based features...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print("\n--- Model V5 Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print("\nBenchmark to beat (our team-based EWMA model): 12.11")

    joblib.dump(model, MODEL_OUTPUT_FILE)
    print(f"Model saved to '{MODEL_OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"ERROR: File not found. Please run feature engineering first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")