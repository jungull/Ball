import pandas as pd

# --- Configuration ---
RAW_DATA_FILE = "nba_games_raw.csv"
PLAYER_DATA_FILE = "nba_games_with_players.csv" # The output from the script that is running now
OUTPUT_FILE = "nba_games_master_features.csv"
ROLLING_WINDOW = 10

# --- Main Script ---
print("--- V2 Feature Engineering ---")

try:
    # 1. LOAD RAW DATA
    print(f"Loading raw data from '{RAW_DATA_FILE}'...")
    df_raw = pd.read_csv(RAW_DATA_FILE)
    df_raw['GAME_DATE'] = pd.to_datetime(df_raw['GAME_DATE'])

    # 2. CALCULATE THE FOUR FACTORS
    print("Calculating the Four Factors for each game...")
    # To calculate ORB%, we need the opponent's DREB. Let's find it.
    # Create a temporary key for merging
    df_raw['temp_key'] = df_raw.apply(lambda row: tuple(sorted((row['TEAM_NAME'], row['MATCHUP'].split(' vs. ')[-1].split(' @ ')[-1]))), axis=1)
    
    # Self-merge to find opponent stats for each game row
    opponent_stats = df_raw.copy()
    merged = pd.merge(df_raw, opponent_stats, on=['GAME_ID', 'GAME_DATE'], suffixes=('', '_opp'))
    df = merged[merged['TEAM_ID'] != merged['TEAM_ID_opp']].copy()

    # Now we can calculate the factors
    df['eFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
    df['ORB_PCT'] = df['OREB'] / (df['OREB'] + df['DREB_opp'])
    df['FT_RATE'] = df['FTM'] / df['FGA']
    
    # Select the columns we need
    four_factors_df = df[['TEAM_ABBREVIATION', 'GAME_DATE', 'GAME_ID', 'eFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FT_RATE']]
    
    # 3. CALCULATE ROLLING AVERAGES FOR THE FOUR FACTORS
    print(f"Calculating {ROLLING_WINDOW}-game rolling averages for the Four Factors...")
    four_factors_df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'], inplace=True)
    
    factors_to_roll = ['eFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FT_RATE']
    for factor in factors_to_roll:
        four_factors_df[f'{factor}_roll_{ROLLING_WINDOW}'] = four_factors_df.groupby('TEAM_ABBREVIATION')[factor].transform(
            lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean()
        )
    
    # 4. CREATE THE FINAL "ONE ROW PER GAME" DATASET
    print("Creating the final dataset with mismatch features...")
    
    # Drop games where we couldn't calculate rolling averages
    four_factors_df.dropna(inplace=True)
    
    # Split into home and away and merge
    away = four_factors_df[four_factors_df['MATCHUP'].str.contains('@')].copy()
    home = four_factors_df[~four_factors_df['MATCHUP'].str.contains('@')].copy()
    
    away = away.add_suffix('_away')
    home = home.add_suffix('_home')
    
    final_df = pd.merge(home, away, left_on='GAME_ID_home', right_on='GAME_ID_away')

    # 5. CREATE THE "MISMATCH / ADVANTAGE" FEATURES
    # This is the golden nugget from the article
    for factor in factors_to_roll:
        final_df[f'{factor}_advantage'] = final_df[f'{factor}_roll_{ROLLING_WINDOW}_home'] - final_df[f'{factor}_roll_{ROLLING_WINDOW}_away']
        
    # 6. MERGE WITH PLAYER DATA (if available)
    print(f"Loading player data from '{PLAYER_DATA_FILE}'...")
    try:
        player_df = pd.read_csv(PLAYER_DATA_FILE)
        
        # Select only the aggregated player stats and the game ID to merge on
        player_features = [col for col in player_df.columns if 'player_' in col]
        player_df_to_merge = player_df[['GAME_ID_home'] + player_features + ['point_differential']]
        
        # Merge our new Four Factor features with the player features
        final_df = pd.merge(final_df, player_df_to_merge, on='GAME_ID_home')
        print("Successfully merged player data.")

    except FileNotFoundError:
        print("Player data file not found. Skipping merge. Run add_player_stats.py to include it.")

    # 7. SAVE THE MASTER FEATURE SET
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS! Master feature set saved to '{OUTPUT_FILE}'")
    print(f"Dataset has {final_df.shape[0]} games and {final_df.shape[1]} columns.")
    print("\n--- Example Columns ---")
    print(final_df[['GAME_ID_home', 'eFG_PCT_advantage', 'ORB_PCT_advantage', 'point_differential']].head())

except FileNotFoundError as e:
    print(f"\nERROR: Could not find a required file. Make sure '{e.filename}' exists.")
    print("You may need to run previous scripts.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")