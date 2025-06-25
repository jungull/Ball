import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# --- Configuration ---
MASTER_FEATURE_FILE = "nba_games_master_features.csv"
MODEL_OUTPUT_FILE = "nba_model_v4.joblib" # Saving as V4
TEST_SIZE = 0.2

# --- Main Script ---
print("--- Training Model V4 (Pure) ---")
print("Loading the master feature dataset...")
try:
    df = pd.read_csv(MASTER_FEATURE_FILE)
    date_col = 'GAME_DATE_home' if 'GAME_DATE_home' in df.columns else 'GAME_DATE_home_home'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 1. DEFINE FEATURES (X) and TARGET (y)
    target = 'point_differential'
    
    # V4 "PURE" APPROACH: Use ONLY the Four Factors advantage features.
    features = [
        'eFG_PCT_advantage',
        'FT_RATE_advantage',
        'ORB_PCT_advantage',
        'TOV_PCT_advantage',
    ]
    
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
    print("Training the V4 RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. MAKE PREDICTIONS ON THE TEST SET
    predictions = model.predict(X_test)

    # 5. EVALUATE THE MODEL'S PERFORMANCE
    mae = mean_absolute_error(y_test, predictions)
    print("\n--- Model V4 Evaluation ---")
    print(f"Mean Absolute Error (MAE) on the test set: {mae:.2f}")
    print(f"This means, on average, our model's prediction for the point differential is off by about {mae:.2f} points.")
    print("\nBenchmark to beat (V1 Model): 12.27")

    # 6. SHOW FEATURE IMPORTANCES
    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("The V4 model found these features most important:")
    print(feature_importance_df)

    # 7. SAVE THE TRAINED V4 MODEL
    print(f"\nSaving the trained V4 model to '{MODEL_OUTPUT_FILE}'...")
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print("Model saved successfully.")

except FileNotFoundError:
    print(f"ERROR: The file '{MASTER_FEATURE_FILE}' was not found.")
    print("Please run 'feature_engineering_v2.py' script first.")
except Exception as e:
    print(f"An error occurred: {e}")