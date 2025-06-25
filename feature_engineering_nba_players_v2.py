import pandas as pd

# --- Configuration ---
RAW_PLAYER_LOGS = "player_stats_nba_raw.csv" # Assuming you have a raw file for NBA players
RAW_GAMES_FILE = "nba_games_raw.csv"
OUTPUT_FILE = "nba_games_player_features_v2.csv"
ROLLING_WINDOW = 10

# --- Main Script ---
print("--- V2 Player Feature Engineering: Star Power vs. Bench Power ---")

try:
    # 1. LOAD RAW PLAYER AND GAME DATA
    print("Loading raw player and game data...")
    player_logs = pd.read_csv(RAW_PLAYER_LOGS)
    games_df = pd.read_csv(RAW_GAMES_FILE)

    player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    
    # 2. CALCULATE PLAYER ROLLING STATS (EWMA is better, but rolling is simpler to start)
    print("Calculating rolling stats for each player...")
    stats_to_roll = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    player_logs.sort_values(by=['Player_ID', 'GAME_DATE'], inplace=True)
    for stat in stats_to_roll:
        player_logs[f'player_{stat}_roll'] = player_logs.groupby('Player_ID')[stat].transform(
            lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean()
        )
    player_logs.dropna(subset=[f'player_{stat}_roll' for stat in stats_to_roll], inplace=True)

    # 3. AGGREGATE STATS FOR EACH GAME
    print("Aggregating player stats for each game (this may take a while)...")
    game_features = []
    
    # We only need a few columns for the main games loop
    games_loop_df = games_df[['GAME_ID', 'GAME_DATE', 'MATCHUP']].drop_duplicates()

    for index, game in games_loop_df.iterrows():
        game_date = game['GAME_DATE']
        matchup = game['MATCHUP']
        home_team_abbr = matchup.split(' vs. ')[0]
        away_team_abbr = matchup.split(' vs. ')[1]
        
        # Find all players who played for each team BEFORE this game
        # We find their most recent stat line before this specific game
        relevant_player_stats = player_logs[player_logs['GAME_DATE'] < game_date]
        latest_player_stats = relevant_player_stats.groupby('Player_ID').last()
        
        home_players = latest_player_stats[latest_player_stats['TEAM_ABBREVIATION'] == home_team_abbr].copy()
        away_players = latest_player_stats[latest_player_stats['TEAM_ABBREVIATION'] == away_team_abbr].copy()
        
        # Sort players by their rolling PTS to identify best players
        home_players.sort_values(by='player_PTS_roll', ascending=False, inplace=True)
        away_players.sort_values(by='player_PTS_roll', ascending=False, inplace=True)
        
        # 4. CREATE THE "STAR POWER" and "BENCH POWER" FEATURES
        game_dict = {'GAME_ID': game['GAME_ID']}
        for stat in stats_to_roll:
            # Top player matchup
            game_dict[f'top1_{stat}_diff'] = home_players.head(1)[f'player_{stat}_roll'].sum() - away_players.head(1)[f'player_{stat}_roll'].sum()
            # "Starting 5" matchup
            game_dict[f'top5_{stat}_diff'] = home_players.head(5)[f'player_{stat}_roll'].sum() - away_players.head(5)[f'player_{stat}_roll'].sum()
            # "Bench" matchup (players 6-10)
            game_dict[f'bench_{stat}_diff'] = home_players.iloc[5:10][f'player_{stat}_roll'].sum() - away_players.iloc[5:10][f'player_{stat}_roll'].sum()
        
        game_features.append(game_dict)
        
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(games_loop_df)} games...")

    # 5. MERGE NEW FEATURES WITH ORIGINAL DATA
    player_features_df = pd.DataFrame(game_features)
    
    # Get the point differential from the original games file
    point_diff_df = games_df[['GAME_ID', 'PLUS_MINUS']].rename(columns={'PLUS_MINUS': 'point_differential'})
    # We only need one entry per game (the home team's perspective)
    home_point_diff_df = games_df[~games_df['MATCHUP'].str.contains('@')][['GAME_ID', 'PLUS_MINUS']].rename(columns={'PLUS_MINUS': 'point_differential'})
    
    final_df = pd.merge(player_features_df, home_point_diff_df, on='GAME_ID')
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully created new player features at '{OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"ERROR: Could not find required raw data files.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")