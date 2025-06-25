import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time
import os
import joblib # Using joblib for efficient saving/loading of checkpoints

# --- Configuration ---
SEASON = '2023-24'
PROCESSED_GAMES_FILE = "nba_games_processed.csv"
OUTPUT_FILE = "nba_games_with_players.csv"
ROLLING_WINDOW = 10

# --- NEW: Checkpoint Configuration ---
CHECKPOINT_DF_FILE = "player_stats_checkpoint.joblib"
CHECKPOINT_SET_FILE = "processed_players_checkpoint.joblib"

# --- Main Script ---
def get_all_player_logs_robust(season):
    """
    Fetches game logs for all players for a given season.
    NEW: Includes timeout and checkpointing to be more robust.
    """
    all_players = players.get_players()
    all_player_ids = [player['id'] for player in all_players]

    # --- Checkpoint Loading ---
    if os.path.exists(CHECKPOINT_DF_FILE) and os.path.exists(CHECKPOINT_SET_FILE):
        print("--- Checkpoint found! Resuming from last session. ---")
        all_logs_df = joblib.load(CHECKPOINT_DF_FILE)
        processed_ids = joblib.load(CHECKPOINT_SET_FILE)
        print(f"Already processed {len(processed_ids)} players.")
    else:
        print("--- No checkpoint found. Starting fresh. ---")
        all_logs_df = pd.DataFrame()
        processed_ids = set()

    print(f"Fetching all player game logs for the {season} season. This will take a while...")

    for i, player_id in enumerate(all_player_ids):
        # If we already processed this player in a previous run, skip them.
        if player_id in processed_ids:
            continue

        try:
            # Be polite to the API
            time.sleep(0.6) # Slightly increased delay
            # NEW: Added a 30-second timeout to prevent getting stuck
            log = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=30)
            df_log = log.get_data_frames()[0]

            if not df_log.empty:
                all_logs_df = pd.concat([all_logs_df, df_log], ignore_index=True)

            # Mark this player as processed
            processed_ids.add(player_id)

            # Print progress
            if (i + 1) % 25 == 0:
                print(f"Progress: Processed {i + 1}/{len(all_player_ids)} players...")
                # --- Checkpoint Saving ---
                print("...saving checkpoint...")
                joblib.dump(all_logs_df, CHECKPOINT_DF_FILE)
                joblib.dump(processed_ids, CHECKPOINT_SET_FILE)

        except Exception as e:
            print(f"!!! Error on player_id {player_id}. Skipping. Reason: {e}")
            processed_ids.add(player_id) # Add to processed list so we don't try again
            continue

    print(f"\nFinished fetching player logs. Found {len(all_logs_df)} total game entries.")
    # --- Cleanup Checkpoints ---
    if os.path.exists(CHECKPOINT_DF_FILE): os.remove(CHECKPOINT_DF_FILE)
    if os.path.exists(CHECKPOINT_SET_FILE): os.remove(CHECKPOINT_SET_FILE)
    
    return all_logs_df

# (The rest of the script is the same as before)

print("--- Step 1: Fetching Player Data (Robust Version) ---")
player_stats = get_all_player_logs_robust(SEASON)

# Check if we got any data before proceeding
if player_stats.empty:
    print("\nNo player data was fetched. Exiting.")
else:
    player_stats['GAME_DATE'] = pd.to_datetime(player_stats['GAME_DATE'])
    player_stats.sort_values(by=['Player_ID', 'GAME_DATE'], inplace=True)

    print("\n--- Step 2: Calculating Player Rolling Averages ---")
    stats_to_roll = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    for stat in stats_to_roll:
        player_stats[f'player_{stat}_roll_{ROLLING_WINDOW}'] = player_stats.groupby('Player_ID')[stat].transform(
            lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean()
        )

    print("\n--- Step 3: Loading Processed Game Data ---")
    games_df = pd.read_csv(PROCESSED_GAMES_FILE)
    games_df['GAME_DATE_home'] = pd.to_datetime(games_df['GAME_DATE_home'])
    
    print("\n--- Step 4: Aggregating Player Stats for Each Game ---")
    aggregated_stats = []
    # Drop player duplicates for a game day, keeping the last entry
    player_stats_latest = player_stats.drop_duplicates(subset=['Player_ID', 'GAME_DATE'], keep='last')

    for index, game in games_df.iterrows():
        game_date = game['GAME_DATE_home']
        home_team_abbr = game['TEAM_ABBREVIATION_home']
        away_team_abbr = game['TEAM_ABBREVIATION_away']
        
        relevant_player_stats = player_stats_latest[player_stats_latest['GAME_DATE'] < game_date]
        latest_player_stats = relevant_player_stats.groupby('Player_ID').last()
        
        home_players = latest_player_stats[latest_player_stats['MATCHUP'].str.contains(home_team_abbr)]
        away_players = latest_player_stats[latest_player_stats['MATCHUP'].str.contains(away_team_abbr)]
        
        home_agg = home_players[[f'player_{stat}_roll_{ROLLING_WINDOW}' for stat in stats_to_roll]].sum()
        away_agg = away_players[[f'player_{stat}_roll_{ROLLING_WINDOW}' for stat in stats_to_roll]].sum()
        
        game_agg_stats = home_agg.add_suffix('_home').to_dict()
        game_agg_stats.update(away_agg.add_suffix('_away').to_dict())
        aggregated_stats.append(game_agg_stats)

        if (index + 1) % 100 == 0:
            print(f"Aggregated stats for {index + 1}/{len(games_df)} games...")

    agg_df = pd.DataFrame(aggregated_stats)
    final_df = pd.concat([games_df.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
    final_df.dropna(inplace=True)

    print("\n--- Step 5: Saving Final Dataset ---")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully created new dataset with player stats at '{OUTPUT_FILE}'")
    print(f"New dataset has {final_df.shape[0]} games and {final_df.shape[1]} columns.")