# Import the libraries we need
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# --- Configuration ---
# You can change this season later on.
# '2023-24' is the most recent full season.
SEASON_TO_FETCH = '2023-24'
OUTPUT_FILE_NAME = "nba_games_raw.csv"

# --- Main script ---
print(f"Fetching data for the {SEASON_TO_FETCH} season...")

try:
    # Use the nba_api to get all game logs for the specified season
    gamelogs = leaguegamelog.LeagueGameLog(season=SEASON_TO_FETCH, season_type_all_star='Regular Season')

    # Convert the data into a pandas DataFrame
    df_games = gamelogs.get_data_frames()[0]

    # Check if we got any data
    if df_games.empty:
        print("No data was returned. Check the season format (e.g., '2023-24') or your internet connection.")
    else:
        print(f"Successfully fetched {len(df_games)} game records.")

        # Save the DataFrame to a CSV file
        df_games.to_csv(OUTPUT_FILE_NAME, index=False)

        print(f"Data has been saved to '{OUTPUT_FILE_NAME}'")
        print("\n--- First 5 rows of the data: ---")
        print(df_games.head())

except Exception as e:
    print(f"An error occurred: {e}")