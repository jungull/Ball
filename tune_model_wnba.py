import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# --- Configuration ---
WNBA_EWMA_FEATURE_FILE = "wnba_games_ewma_features.csv"
TUNED_MODEL_OUTPUT_FILE = "wnba_model_tuned.joblib" # Our new, even better champion model

# --- Main Script ---
print("--- Hyperparameter Tuning for WNBA Model ---")
print(f"Loading feature data from '{WNBA_EWMA_FEATURE_FILE}'...")
try:
    df = pd.read_csv(WNBA_EWMA_FEATURE_FILE)
    df['GAME_DATE_home'] = pd.to_datetime(df['GAME_DATE_home'])
    df = df.sort_values('GAME_DATE_home')

    # For tuning, we use the entire dataset to find the best general parameters
    target = 'point_differential'
    features = [col for col in df.columns if col.endswith('_diff')]
    
    X = df[features]
    y = df[target]

    print(f"\nTuning model on {len(X)} games...")

    # 1. DEFINE THE "GRID" OF PARAMETERS TO TEST
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # 2. SET UP AND RUN THE GRID SEARCH
    print("Starting GridSearchCV... This may take several minutes.")
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='neg_mean_absolute_error', cv=5, verbose=2)
    
    grid_search.fit(X, y)

    # 3. REPORT THE BEST SETTINGS
    print("\n--- Tuning Complete ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    best_mae = -grid_search.best_score_
    print(f"Best Cross-Validated MAE from tuning: {best_mae:.2f}")

    # 4. SAVE THE BEST MODEL FOUND BY THE GRID SEARCH
    best_model = grid_search.best_estimator_
    print(f"\nSaving the best tuned WNBA model to '{TUNED_MODEL_OUTPUT_FILE}'...")
    joblib.dump(best_model, TUNED_MODEL_OUTPUT_FILE)
    print("Tuned model saved successfully.")

except FileNotFoundError:
    print(f"ERROR: The file '{WNBA_EWMA_FEATURE_FILE}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")