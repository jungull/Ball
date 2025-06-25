from nba_api.stats.endpoints import scoreboardv2
import pandas as pd

print("--- Inspecting Live WNBA Scoreboard API ---")
try:
    # Fetch today's WNBA games
    scoreboard_data = scoreboardv2.ScoreboardV2(league_id='10')
    game_schedule_df = scoreboard_data.get_data_frames()[0]

    if game_schedule_df.empty:
        print("No WNBA games scheduled for today.")
    else:
        print("Available columns in the live scoreboard data:")
        # This will show us the REAL column names
        print(list(game_schedule_df.columns))
except Exception as e:
    print(f"An error occurred: {e}")