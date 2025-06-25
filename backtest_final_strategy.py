import pandas as pd
import joblib
import io
from sklearn.ensemble import RandomForestRegressor

# --- Configuration ---
EWMA_DATA_FILE = "nba_games_ewma_features.csv" # The full dataset of EWMA features

# --- Betting Strategy Configuration ---
BETTING_THRESHOLDS = {
    "High_Confidence": {'edge': 8.0, 'units': 3},
    "Medium_Confidence": {'edge': 5.0, 'units': 2},
    "Low_Confidence": {'edge': 3.0, 'units': 1}
}

# --- Main Script ---
print("--- FINAL Walk-Forward Backtest with Variable Betting ---")

try:
    # 1. LOAD ALL FEATURE DATA AND ODDS DATA
    print(f"Loading all feature data from '{EWMA_DATA_FILE}'...")
    games_df = pd.read_csv(EWMA_DATA_FILE)
    games_df['GAME_DATE_home'] = pd.to_datetime(games_df['GAME_DATE_home']).dt.date
    games_df = games_df.sort_values('GAME_DATE_home')

    print("Loading and processing real odds from embedded data string...")
    odds_data_string = """date,home_team,away_team,spread_line
2024-02-01,Boston Celtics,Los Angeles Lakers,-11.5
2024-02-01,New York Knicks,Indiana Pacers,-4.0
2024-02-01,Memphis Grizzlies,Cleveland Cavaliers,8.0
2024-02-02,Detroit Pistons,Los Angeles Clippers,13.0
2024-02-02,Washington Wizards,Miami Heat,9.0
2024-02-02,Atlanta Hawks,Phoenix Suns,2.5
2024-02-02,Minnesota Timberwolves,Orlando Magic,-6.5
2024-02-02,San Antonio Spurs,New Orleans Pelicans,9.5
2024-02-02,Oklahoma City Thunder,Charlotte Hornets,-14.0
2024-02-02,Denver Nuggets,Portland Trail Blazers,-14.5
2024-02-03,Philadelphia 76ers,Brooklyn Nets,-1.5
2024-02-03,Atlanta Hawks,Golden State Warriors,-2.0
2024-02-03,Chicago Bulls,Sacramento Kings,2.0
2024-02-03,Dallas Mavericks,Milwaukee Bucks,2.0
2024-02-03,New York Knicks,Los Angeles Lakers,-4.0
2024-02-03,San Antonio Spurs,Cleveland Cavaliers,10.0
2024-02-04,Washington Wizards,Phoenix Suns,12.5
2024-02-04,Boston Celtics,Memphis Grizzlies,-16.5
2024-02-04,Charlotte Hornets,Indiana Pacers,8.0
2024-02-04,Miami Heat,Los Angeles Clippers,-1.0
2024-02-04,Minnesota Timberwolves,Houston Rockets,-8.0
2024-02-04,Oklahoma City Thunder,Toronto Raptors,-10.0
2024-02-04,Utah Jazz,Milwaukee Bucks,5.5
2024-02-04,Denver Nuggets,Portland Trail Blazers,-16.0
2024-02-05,Charlotte Hornets,Los Angeles Lakers,10.5
2024-02-05,Cleveland Cavaliers,Sacramento Kings,-5.0
2024-02-05,Brooklyn Nets,Golden State Warriors,6.0
2024-02-05,Atlanta Hawks,Los Angeles Clippers,6.0
2024-02-05,New Orleans Pelicans,Toronto Raptors,-10.0
"""
    odds_df = pd.read_csv(io.StringIO(odds_data_string))
    odds_df['date'] = pd.to_datetime(odds_df['date']).dt.date
    team_name_map = {'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN','Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE','Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET','Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND','Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM','Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN','New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC','Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX','Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS','Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'}
    odds_df['TEAM_ABBREVIATION_home'] = odds_df['home_team'].map(team_name_map)
    odds_df.rename(columns={'spread_line': 'vegas_spread'}, inplace=True)
    
    # 2. MERGE THE DATA TO FIND OUR TEST SET
    test_df = pd.merge(games_df, odds_df, left_on=['GAME_DATE_home', 'TEAM_ABBREVIATION_home'], right_on=['date', 'TEAM_ABBREVIATION_home'])
    print(f"Found {len(test_df)} games with available odds to use for our backtest.")
    
    if len(test_df) < 1: raise ValueError("No overlapping games found between the feature data and the odds data.")
    
    # 3. THE "TIME MACHINE": TRAIN A MODEL ONLY ON PAST DATA
    first_test_date = test_df['GAME_DATE_home'].min()
    print(f"The first game to predict is on {first_test_date}. Training a model on all data BEFORE this date...")
    
    train_df = games_df[games_df['GAME_DATE_home'] < first_test_date]
    features = [col for col in train_df.columns if col.endswith('_diff')]
    
    X_train = train_df[features]
    y_train = train_df['point_differential']

    # Train our temporary, honest model
    honest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    honest_model.fit(X_train, y_train)
    print("Honest model trained successfully.")

    # 4. MAKE PREDICTIONS ON THE TEST SET USING THE HONEST MODEL
    X_test = test_df[features]
    test_df['model_prediction'] = honest_model.predict(X_test)
    test_df['edge'] = test_df['model_prediction'] - test_df['vegas_spread']

    # 5. IMPLEMENT THE VARIABLE BETTING STRATEGY
    bets_df = pd.DataFrame()
    for level, config in sorted(BETTING_THRESHOLDS.items(), key=lambda item: item[1]['edge'], reverse=True):
        bets = test_df[abs(test_df['edge']) > config['edge']].copy()
        if not bets.empty:
            bets['bet_units'] = config['units']
            bets['confidence_level'] = level
            bets_df = pd.concat([bets_df, bets])
    
    bets_df = bets_df.drop_duplicates(subset=['GAME_ID_home'], keep='first')
    
    print(f"\nFound {len(bets_df)} total betting opportunities across all confidence levels.")

    # 6. REPORT THE FINAL, HONEST RESULTS
    if not bets_df.empty:
        bets_df['bet_won'] = (bets_df['edge'] * (bets_df['point_differential'] - bets_df['vegas_spread'])) > 0
        bets_df['profit_units'] = bets_df.apply(lambda row: row['bet_units'] if row['bet_won'] else -row['bet_units'], axis=1)

        total_wins = bets_df['bet_won'].sum()
        total_bets = len(bets_df)
        win_rate = total_wins / total_bets if total_bets > 0 else 0
        total_units_risked = bets_df['bet_units'].sum()
        total_profit = bets_df['profit_units'].sum()
        roi = total_profit / total_units_risked if total_units_risked > 0 else 0

        print("\n--- FINAL HONEST Backtest Results ---")
        print(f"Total Bets Made: {total_bets}")
        print(f"Wins: {total_wins} | Losses: {total_bets - total_wins}")
        print(f"Overall Win Rate: {win_rate:.2%}")
        print("---------------------------------")
        print(f"Total Units Risked: {total_units_risked}")
        print(f"Total Profit: {total_profit:.2f} units")
        print(f"Return on Investment (ROI): {roi:.2%}")
        print("---------------------------------")
        if total_profit > 0:
            print("SUCCESS! The variable betting strategy was PROFITABLE.")
        else:
            print("The variable betting strategy was NOT profitable.")
    else:
        print("No betting opportunities found with the current edge thresholds.")

except FileNotFoundError as e:
    print(f"\nERROR: Could not find a required file. Make sure '{e.filename}' exists.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")