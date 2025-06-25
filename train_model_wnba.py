import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# --- Configuration ---
EWMA_FEATURE_FILE = "wnba_games_ewma_features.csv"
MODEL_OUTPUT_FILE = "wnba_model_final.joblib" # Our final, champion model
TEST_SIZE = 0.2

# --- Main Script ---
print("--- Training Final Model on EWMA Features ---")
print(f"Loading feature data from '{EWMA_FEATURE_FILE}'...")
try:
    df = pd.read_csv(EWMA_FEATURE_FILE)
    df['GAME_DATE_home'] = pd.to_datetime(df['GAME_DATE_home'])
    df = df.sort_values('GAME_DATE_home') # Sort games chronologically

    # 1. DEFINE FEATURES (X) and TARGET (y)
    # The target is what we want to predict.
    target = 'point_differential'
    
    # Corrected Feature Selection: Only use columns that END with '_diff'
    features = [col for col in df.columns if col.endswith('_diff')]
    
    X = df[features]
    y = df[target]

    print(f"\nFeatures being used for the model ({len(features)} total):")
    print(features)
    
    # 2. SPLIT DATA INTO TRAINING AND TESTING SETS
    split_index = int(len(df) * (1 - TEST_SIZE))
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\nSplitting data: {len(X_train)} games for training, {len(X_test)} games for testing.")

    # 3. INITIALIZE AND TRAIN THE MODEL
    print("Training the final RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. EVALUATE THE MODEL'S PERFORMANCE
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print("\n--- Final Model Evaluation ---")
    print(f"Mean Absolute Error (MAE) on the test set: {mae:.2f}")
    print(f"This means, on average, our model's prediction is off by about {mae:.2f} points.")
    print("\nBenchmark to beat (our previous best V1 Model): 12.27")

    # 5. SAVE THE TRAINED FINAL MODEL
    print(f"\nSaving the final trained model to '{MODEL_OUTPUT_FILE}'...")
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print("Model saved successfully.")

except FileNotFoundError:
    print(f"ERROR: The file '{EWMA_FEATURE_FILE}' was not found.")
    print("Please run 'feature_engineering_final.py' script first.")
except Exception as e:
    print(f"An error occurred: {e}")