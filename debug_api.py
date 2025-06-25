import pandas as pd
from nba_api.stats.endpoints import scoreboardv2

# --- Main Script ---
print("--- API Debugger ---")
print("Fetching today's WNBA scoreboard data to inspect its structure...")

try:
    # Set pandas to display all columns without truncation
    pd.set_option('display.max_columns', None)

    # Fetch the data for today's WNBA games
    games_scoreboard = scoreboardv2.ScoreboardV2(league_id='10')
    games_df = games_scoreboard.get_data_frames()[0]

    if games_df.empty:
        print("\nNo WNBA games scheduled for today.")
    else:
        # Print all available column names
        print("\nAvailable columns from the API:")
        print(games_df.columns.tolist())

        # Print the first few rows of the data to see the content
        print("\nFirst 5 rows of data:")
        print(games_df.head())

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")