import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time

# --- Configuration ---
# Fetch the last several seasons to build a deep history
SEASONS_TO_FETCH = ['2022', '2023', '2024']
OUTPUT_FILE_NAME = "wnba_games_raw.csv" # We will overwrite the old file with this new, richer one

# --- Main script ---
all_season_logs = []

print(f"Fetching WNBA data for seasons: {SEASONS_TO_FETCH}. This may take a moment...")

for season in SEASONS_TO_FETCH:
    print(f"Fetching data for the {season} WNBA season...")
    try:
        # Key Change: league_id='10' specifies the WNBA
        gamelogs = leaguegamelog.LeagueGameLog(season=season, league_id='10', season_type_all_star='Regular Season')
        df_games = gamelogs.get_data_frames()[0]
        all_season_logs.append(df_games)
        print(f"Successfully fetched {len(df_games)} game records for {season}.")
        # Be polite to the API
        time.sleep(1) 
    except Exception as e:
        print(f"An error occurred fetching season {season}: {e}")

# Combine all the seasons into one big DataFrame
if all_season_logs:
    full_history_df = pd.concat(all_season_logs, ignore_index=True)
    full_history_df.to_csv(OUTPUT_FILE_NAME, index=False)
    print(f"\nSuccessfully combined all seasons into '{OUTPUT_FILE_NAME}'")
    print(f"Total historical records fetched: {len(full_history_df)}")
else:
    print("\nNo data was fetched. Please check your connection and the season list.")