import pandas as pd

# --- Configuration ---
INPUT_FILE = "wnba_games_raw.csv"
OUTPUT_FILE = "wnba_games_ewma_features.csv" # New output file
# Alpha is the smoothing factor for EWMA. A smaller alpha gives more weight to past games.
# Alpha = 0.1 is a common choice and what was used in the article.
ALPHA = 0.1

# --- Main Script ---
print(f"--- Final Feature Engineering with EWMA (alpha={ALPHA}) ---")
try:
    df = pd.read_csv(INPUT_FILE)
    print("Data loaded successfully. Starting feature engineering...")

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df.rename(columns={'PLUS_MINUS': 'POINT_DIFFERENTIAL'}, inplace=True)
    df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'], inplace=True)

    # These are the raw stats we'll apply the EWMA to.
    stats_to_average = [
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
    ]

    # Calculate the EWMA for each stat
    for stat in stats_to_average:
        df[f'{stat}_ewma'] = df.groupby('TEAM_ABBREVIATION')[stat].transform(
            # Use .shift(1) to prevent data leakage from the current game
            lambda x: x.shift(1).ewm(alpha=ALPHA, adjust=False).mean()
        )

    # Drop rows that have nulls (the first game of the season for each team)
    df.dropna(inplace=True)

    # Create the final one-row-per-game dataset
    away_games = df[df['MATCHUP'].str.contains('@')].copy()
    home_games = df[~df['MATCHUP'].str.contains('@')].copy()

    away_games = away_games.add_suffix('_away')
    home_games = home_games.add_suffix('_home')

    merged_df = pd.merge(home_games, away_games,
                         left_on='GAME_ID_home',
                         right_on='GAME_ID_away')

    # Create the "Difference" or "Mismatch" features, as described in the article
    for stat in stats_to_average:
        merged_df[f'{stat}_diff'] = merged_df[f'{stat}_ewma_home'] - merged_df[f'{stat}_ewma_away']

    # Select only the columns we actually need for the model
    features_to_keep = ['GAME_ID_home', 'GAME_DATE_home', 'TEAM_ABBREVIATION_home', 'TEAM_ABBREVIATION_away']
    features_to_keep.extend([f'{stat}_diff' for stat in stats_to_average])
    features_to_keep.append('POINT_DIFFERENTIAL_home') # This is our target
    
    final_df = merged_df[features_to_keep].copy()
    final_df.rename(columns={'POINT_DIFFERENTIAL_home': 'point_differential'}, inplace=True)

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProcessing complete!")
    print(f"The final EWMA feature data has been saved to '{OUTPUT_FILE}'")
    print("\n--- First 5 rows of new feature data: ---")
    print(final_df.head())

except FileNotFoundError:
    print(f"ERROR: The file '{INPUT_FILE}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")