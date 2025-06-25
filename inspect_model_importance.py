import pandas as pd
import joblib

# --- Configuration ---
# Point this to our best, tuned WNBA model
TUNED_MODEL_FILE = "wnba_model_tuned.joblib"
# We need the data file to get the feature names in the correct order
WNBA_EWMA_FEATURE_FILE = "wnba_games_ewma_features.csv"

# --- Main Script ---
print(f"--- Inspecting Feature Importances for Model: {TUNED_MODEL_FILE} ---")

try:
    # 1. LOAD THE TUNED MODEL AND THE DATA
    model = joblib.load(TUNED_MODEL_FILE)
    df = pd.read_csv(WNBA_EWMA_FEATURE_FILE)

    # 2. GET THE LIST OF FEATURES (must be identical to the training script)
    features = [col for col in df.columns if col.endswith('_diff')]

    # 3. EXTRACT THE IMPORTANCE SCORES
    # The model stores these after being trained.
    importances = model.feature_importances_

    # 4. CREATE A DATAFRAME FOR EASY VIEWING
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # 5. SORT BY IMPORTANCE AND DISPLAY
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Add a percentage column for easier interpretation
    importance_df['Importance (%)'] = (importance_df['Importance'] * 100).map('{:.2f}%'.format)

    print("\nFeature importance represents the 'weight' or 'value' the model assigns to each feature.")
    print("A higher value means the model found that feature more predictive.\n")
    
    # Use to_string() to ensure all rows are printed
    print(importance_df[['Feature', 'Importance (%)']].to_string(index=False))

except FileNotFoundError as e:
    print(f"\nERROR: Could not find a required file: {e.filename}")
    print("Please make sure you have run the training and tuning scripts first.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")