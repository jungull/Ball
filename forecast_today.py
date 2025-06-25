import pandas as pd
import joblib
from nba_api.stats.endpoints import scoreboardv2
import sys
import os

# --- Universal Helper Function ---
def calculate_latest_ewma(historical_df, team_abbr, alpha=0.1):
    """Calculates the most recent EWMA stats for a single team."""
    team_df = historical_df[historical_df['TEAM_ABBREVIATION'] == team_abbr].copy()
    if team_df.empty:
        # FINAL FIX: Return only None, not a tuple
        return None
    
    team_df.sort_values(by='GAME_DATE', inplace=True)
    stats_to_average = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
                        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    
    for stat in stats_to_average:
        team_df[f'{stat}_ewma'] = team_df[stat].ewm(alpha=alpha, adjust=False).mean()
        
    # FINAL FIX: Return only the data Series, not a tuple
    return team_df.iloc[-1][[f'{stat}_ewma' for stat in stats_to_average]]

# --- Main Script ---
print("--- Unified Game Forecaster (Production Version) ---")

# 1. CHOOSE THE LEAGUE
league_choice = input("Which league would you like to predict? (NBA/WNBA): ").strip().upper()

if league_choice == 'NBA':
    MODEL_FILE = "nba_model_tuned.joblib"
    HISTORICAL_RAW_DATA = "nba_games_raw.csv"
    LEAGUE_ID = '00'
elif league_choice == 'WNBA':
    MODEL_FILE = "wnba_model_tuned.joblib"
    HISTORICAL_RAW_DATA = "wnba_games_raw.csv"
    LEAGUE_ID = '10'
else:
    print("Invalid choice. Please enter 'NBA' or 'WNBA'.")
    sys.exit()

PREDICTION_LOG_FILE = 'prediction_log.csv'

try:
    # 2. LOAD MODEL AND DATA
    print(f"\nLoading {league_choice} tuned model from '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    historical_df = pd.read_csv(HISTORICAL_RAW_DATA)
    historical_df['GAME_DATE'] = pd.to_datetime(historical_df['DATE'])

    # 3. BUILD THE TEAM ID -> ABBREVIATION TRANSLATOR
    team_id_map = historical_df[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates().set_index('TEAM_ID')['TEAM_ABBREVIATION'].to_dict()

    # 4. GET TODAY'S GAMES
    print(f"Fetching today's {league_choice} schedule...")
    games = scoreboardv2.ScoreboardV2(league_id=LEAGUE_ID).get_data_frames()[0]
    
    if games.empty:
        print(f"No {league_choice} games scheduled for today.")
        sys.exit()

    # 5. PROCESS GAMES AND GENERATE PREDICTIONS
    predictions_today = []
    for index, game in games.iterrows():
        home_team_id = game['HOME_TEAM_ID']
        away_team_id = game['VISITOR_TEAM_ID']
        home_team_abbr = team_id_map.get(home_team_id)
        away_team_abbr = team_id_map.get(away_team_id)

        if not home_team_abbr or not away_team_abbr:
            print(f"\nSkipping game with ID {game['GAME_ID']}. Reason: Unknown Team ID. Home: {home_team_id}, Away: {away_team_id}")
            continue
            
        print(f"\nProcessing game: {away_team_abbr} at {home_team_abbr}")

        home_stats = calculate_latest_ewma(historical_df, home_team_abbr)
        away_stats = calculate_latest_ewma(historical_df, away_team_abbr)
        
        if home_stats is None or away_stats is None:
            print(f"Skipping game due to missing historical data.")
            continue

        # Subtraction now works because home_stats and away_stats are Series, not tuples
        diff_stats = home_stats - away_stats
        
        # Create a DataFrame with the correct column names for a robust prediction
        features_for_model = pd.DataFrame([diff_stats.values], columns=model.feature_names_in_)
        
        predicted_diff = model.predict(features_for_model)[0]

        try:
            vegas_spread_str = input(f"Enter Vegas Spread for {home_team_abbr} (e.g., -5.5, or 'skip'): ")
            if vegas_spread_str.lower() == 'skip': continue
            vegas_spread = float(vegas_spread_str)
        except ValueError:
            print("Invalid input. Skipping game.")
            continue

        edge = predicted_diff - vegas_spread
        recommendation = "No Bet"
        if edge > 3.0: recommendation = f"Bet on {home_team_abbr} (Spread: {vegas_spread})"
        if edge < -3.0: recommendation = f"Bet on {away_team_abbr} (Spread: {'+' if -vegas_spread > 0 else ''}{-vegas_spread})"

        game_prediction = {
            "Date": pd.Timestamp.today().strftime('%Y-%m-%d'), "League": league_choice,
            "Home Team": home_team_abbr, "Away Team": away_team_abbr,
            "Model Prediction": f"{home_team_abbr} by {predicted_diff:.1f}",
            "Vegas Spread": f"{home_team_abbr} by {vegas_spread:.1f}",
            "Edge": f"{edge:.1f}", "Recommendation": recommendation,
            "Actual Result": "Pending"
        }
        predictions_today.append(game_prediction)

    # 6. DISPLAY AND SAVE RESULTS
    if predictions_today:
        results_df = pd.DataFrame(predictions_today)
        print(f"\n--- Today's {league_choice} Forecasts ---")
        print(results_df.to_string())

        if os.path.exists(PREDICTION_LOG_FILE):
            log_df = pd.read_csv(PREDICTION_LOG_FILE)
            log_df = pd.concat([log_df, results_df], ignore_index=True)
        else:
            log_df = results_df
        log_df.to_csv(PREDICTION_LOG_FILE, index=False)
        print(f"\nPredictions have been saved to '{PREDICTION_LOG_FILE}'")
    else:
        print("\nNo predictions were generated.")

except FileNotFoundError as e:
    print(f"ERROR: Could not find required file: {e.filename}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")