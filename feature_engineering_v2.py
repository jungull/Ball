import pandas as pd

# --- Configuration ---
RAW_DATA_FILE = "nba_games_raw.csv"
PLAYER_DATA_FILE = "nba_games_with_players.csv" # The output from the script that just finished
OUTPUT_FILE = "nba_games_master_features.csv"
ROLLING_WINDOW = 10

# --- Main Script ---
print("--- V2 Feature Engineering (Corrected Version) ---")

try:
    # 1. LOAD RAW DATA
    print(f"Loading raw data from '{RAW_DATA_FILE}'...")
    df_raw = pd.read_csv(RAW_DATA_FILE)
    df_raw['GAME_DATE'] = pd.to_datetime(df_raw['GAME_DATE'])

    # 2. CALCULATE THE FOUR FACTORS
    print("Calculating the Four Factors for each game...")
    
    # Self-merge to find opponent stats for each game row
    opponent_stats = df_raw.copy()
    merged = pd.merge(df_raw, opponent_stats, on=['GAME_ID', 'GAME_DATE'], suffixes=('', '_opp'))
    df = merged[merged['TEAM_ID'] != merged['TEAM_ID_opp']].copy()

    # Now we can calculate the factors. Added small epsilon to avoid division by zero.
    epsilon = 1e-6 
    df['eFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / (df['FGA'] + epsilon)
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'] + epsilon)
    df['ORB_PCT'] = df['OREB'] / (df['OREB'] + df['DREB_opp'] + epsilon)
    df['FT_RATE'] = df['FTM'] / (df['FGA'] + epsilon)
    
    # Select the columns we need.
    # FIX: We must include 'MATCHUP' so we can split by home/away later.
    columns_to_keep = ['TEAM_ABBREVIATION', 'GAME_DATE', 'GAME_ID', 'MATCHUP', 'eFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FT_RATE']
    four_factors_df = df[columns_to_keep].copy() # Use .copy() to prevent SettingWithCopyWarning
    
    # 3. CALCULATE ROLLING AVERAGES FOR THE FOUR FACTORS
    print(f"Calculating {ROLLING_WINDOW}-game rolling averages...")
    four_factors_df = four_factors_df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    factors_to_roll = ['eFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FT_RATE']
    for factor in factors_to_roll:
        four_factors_df[f'{factor}_roll_{ROLLING_WINDOW}'] = four_factors_df.groupby('TEAM_ABBREVIATION')[factor].transform(
            lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean()
        )
    
    # 4. CREATE THE FINAL "ONE ROW PER GAME" DATASET
    print("Creating the final dataset with mismatch features...")
    
    # Drop games where we couldn't calculate rolling averages
    four_factors_df = four_factors_df.dropna() # This is a cleaner way to handle dropna
    
    # Split into home and away and merge
    away = four_factors_df[four_factors_df['MATCHUP'].str.contains('@')].copy()
    home = four_factors_df[~four_factors_df['MATCHUP'].str.contains('@')].copy()
    
    away = away.add_suffix('_away')
    home = home.add_suffix('_home')
    
    final_df = pd.merge(home, away, left_on='GAME_ID_home', right_on='GAME_ID_away')

    # 5. CREATE THE "MISMATCH / ADVANTAGE" FEATURES
    print("Creating advantage features...")
    for factor in factors_to_roll:
        final_df[f'{factor}_advantage'] = final_df[f'{factor}_roll_{ROLLING_WINDOW}_home'] - final_df[f'{factor}_roll_{ROLLING_WINDOW}_away']
            
    # 6. MERGE WITH PLAYER DATA
    print(f"Loading player data from '{PLAYER_DATA_FILE}'...")
    player_df = pd.read_csv(PLAYER_DATA_FILE)
    
    # Select only the aggregated player stats and the game ID to merge on
    player_features = [col for col in player_df.columns if 'player_' in col]
    # We also need to bring the target variable (point_differential) from this file
    player_df_to_merge = player_df[['GAME_ID_home'] + player_features + ['point_differential']]
    
    # Merge our new Four Factor features with the player features
    final_df = pd.merge(final_df, player_df_to_merge, on='GAME_ID_home')
    print("Successfully merged player data.")

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