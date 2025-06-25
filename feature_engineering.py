import pandas as pd

# --- Configuration ---
INPUT_FILE = "nba_games_raw.csv"
OUTPUT_FILE = "nba_games_processed.csv"
ROLLING_WINDOW = 10 # The number of past games to average over

# --- Main Script ---
print(f"Loading data from '{INPUT_FILE}'...")
try:
    df = pd.read_csv(INPUT_FILE)
    print("Data loaded successfully. Starting feature engineering...")

    # 1. DATA CLEANING & PREPARATION
    # Convert GAME_DATE to a datetime object so we can sort by it
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    # Rename the confusing 'PLUS_MINUS' to our target variable name
    df.rename(columns={'PLUS_MINUS': 'POINT_DIFFERENTIAL'}, inplace=True)
    # Sort the data frame chronologically for each team. This is CRITICAL for rolling averages.
    df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'], inplace=True)

    # 2. CALCULATE ROLLING AVERAGES
    # These will be our model's features (predictors)
    stats_to_roll = [
        'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV', 'PTS'
    ]

    # Group by team, then calculate the rolling average for our chosen stats.
    # .shift(1) is crucial: it uses the stats from the *previous* 10 games, not including the current game.
    # This prevents data leakage (i.e., using the result of the current game to predict itself).
    for stat in stats_to_roll:
        df[f'{stat}_roll_{ROLLING_WINDOW}'] = df.groupby('TEAM_ABBREVIATION')[stat].transform(
            lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean()
        )

    # 3. CREATE A DATASET WITH ONE ROW PER GAME
    # Drop any rows that have nulls (the first few games of the season for each team)
    df.dropna(inplace=True)

    # Separate into home and away games. The '@' symbol indicates an away game.
    away_games = df[df['MATCHUP'].str.contains('@')].copy()
    home_games = df[~df['MATCHUP'].str.contains('@')].copy()

    # Rename columns to distinguish between home and away teams
    away_games = away_games.add_suffix('_away')
    home_games = home_games.add_suffix('_home')

    # Merge home and away games on the game ID and date
    merged_df = pd.merge(home_games, away_games,
                         left_on='GAME_ID_home',
                         right_on='GAME_ID_away')

    # 4. FINAL CLEANUP & FEATURE CREATION
    # Define the target variable: 1 if the home team won, 0 if they lost
    merged_df['home_team_won'] = (merged_df['POINT_DIFFERENTIAL_home'] > 0).astype(int)
    
    # We also want to predict the point spread.
    # Let's keep the home team's point differential as a primary target.
    merged_df['point_differential'] = merged_df['POINT_DIFFERENTIAL_home']

    # Select only the columns we actually need for the model
    features_to_keep = [
        'GAME_ID_home', 'GAME_DATE_home', 'TEAM_ABBREVIATION_home', 'TEAM_ABBREVIATION_away',
        'FG_PCT_roll_10_home', 'FT_PCT_roll_10_home', 'FG3_PCT_roll_10_home',
        'AST_roll_10_home', 'REB_roll_10_home', 'TOV_roll_10_home', 'PTS_roll_10_home',
        'FG_PCT_roll_10_away', 'FT_PCT_roll_10_away', 'FG3_PCT_roll_10_away',
        'AST_roll_10_away', 'REB_roll_10_away', 'TOV_roll_10_away', 'PTS_roll_10_away',
        'home_team_won', # Target 1 (Classification)
        'point_differential' # Target 2 (Regression)
    ]
    
    final_df = merged_df[features_to_keep]

    # 5. SAVE THE PROCESSED DATA
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProcessing complete!")
    print(f"The final processed data has {final_df.shape[0]} games and {final_df.shape[1]} columns.")
    print(f"Data saved to '{OUTPUT_FILE}'")
    print("\n--- First 5 rows of processed data: ---")
    print(final_df.head())

except FileNotFoundError:
    print(f"ERROR: The file '{INPUT_FILE}' was not found.")
    print("Please run the 'data_collection.py' script first.")
except Exception as e:
    print(f"An error occurred: {e}")