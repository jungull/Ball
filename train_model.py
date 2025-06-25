import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib # Used to save our trained model

# --- Configuration ---
PROCESSED_DATA_FILE = "nba_games_processed.csv"
MODEL_OUTPUT_FILE = "nba_model.joblib"
# What percentage of data to use for testing. 0.2 means 20%.
TEST_SIZE = 0.2

# --- Main Script ---
print("Loading processed data...")
try:
    df = pd.read_csv(PROCESSED_DATA_FILE)
    # Convert date column to datetime to sort on it properly
    df['GAME_DATE_home'] = pd.to_datetime(df['GAME_DATE_home'])
    df = df.sort_values('GAME_DATE_home') # Sort games chronologically

    # 1. DEFINE FEATURES (X) and TARGET (y)
    # The target is what we want to predict.
    target = 'point_differential'
    # The features are the inputs the model will use to make the prediction.
    features = [col for col in df.columns if 'roll_10' in col]

    X = df[features] # Our input data
    y = df[target]   # What we want to predict

    print(f"Features being used for the model: {features}")
    
    # 2. SPLIT DATA INTO TRAINING AND TESTING SETS
    # This is the most important step in machine learning.
    # We train the model on older data and test it on newer, unseen data.
    split_index = int(len(df) * (1 - TEST_SIZE))
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"\nSplitting data: {len(X_train)} games for training, {len(X_test)} games for testing.")

    # 3. INITIALIZE AND TRAIN THE MODEL
    print("Training the RandomForestRegressor model...")
    # n_estimators is the number of "trees" in the forest. More is often better but slower.
    # random_state ensures we get the same result every time we run the script.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. MAKE PREDICTIONS ON THE TEST SET
    predictions = model.predict(X_test)

    # 5. EVALUATE THE MODEL'S PERFORMANCE
    mae = mean_absolute_error(y_test, predictions)
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE) on the test set: {mae:.2f}")
    print(f"This means, on average, our model's prediction for the point differential is off by about {mae:.2f} points.")
    
    # 6. SHOW SOME EXAMPLE PREDICTIONS
    results_df = pd.DataFrame({'Actual_Point_Diff': y_test, 'Predicted_Point_Diff': predictions})
    print("\n--- Example Predictions vs Actual ---")
    print(results_df.head(10).round(1))
    
    # 7. SAVE THE TRAINED MODEL
    print(f"\nSaving the trained model to '{MODEL_OUTPUT_FILE}'...")
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print("Model saved successfully.")

except FileNotFoundError:
    print(f"ERROR: The file '{PROCESSED_DATA_FILE}' was not found.")
    print("Please run the 'feature_engineering.py' script first.")
except Exception as e:
    print(f"An error occurred: {e}")