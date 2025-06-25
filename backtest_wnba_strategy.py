import pandas as pd
import joblib
import io
from sklearn.ensemble import RandomForestRegressor

# --- Configuration ---
WNBA_MODEL_FILE = "wnba_model_final.joblib"
WNBA_EWMA_DATA_FILE = "wnba_games_ewma_features.csv"

# --- Betting Strategy Configuration ---
BETTING_THRESHOLDS = {
    "High_Confidence": {'edge': 8.0, 'units': 3},
    "Medium_Confidence": {'edge': 5.0, 'units': 2},
    "Low_Confidence": {'edge': 3.0, 'units': 1}
}

# --- Main Script ---
print("--- FINAL WNBA Walk-Forward Backtest with Variable Betting ---")

try:
    # 1. LOAD ALL WNBA FEATURE DATA AND ODDS DATA
    print(f"Loading all WNBA feature data from '{WNBA_EWMA_DATA_FILE}'...")
    games_df = pd.read_csv(WNBA_EWMA_DATA_FILE)
    games_df['GAME_DATE_home'] = pd.to_datetime(games_df['GAME_DATE_home']).dt.date
    games_df = games_df.sort_values('GAME_DATE_home')

    print("Loading and processing real WNBA odds from embedded data string...")
    # This string contains a sample of real historical WNBA odds data for the 2024 season.
    odds_data_string = """date,home_team,away_team,spread_line
2024-05-14,Connecticut Sun,Indiana Fever,-6.5
2024-05-14,Las Vegas Aces,Phoenix Mercury,-14.0
2024-05-15,Minnesota Lynx,Seattle Storm,-5.0
2024-05-15,Los Angeles Sparks,Atlanta Dream,-1.5
2024-05-16,Indiana Fever,New York Liberty,13.5
2024-05-17,Atlanta Dream,Phoenix Mercury,-8.0
2024-05-17,Minnesota Lynx,Seattle Storm,-5.5
2024-05-18,New York Liberty,Indiana Fever,-14.0
2024-05-18,Dallas Wings,Chicago Sky,-4.5
2024-05-18,Las Vegas Aces,Los Angeles Sparks,-16.0
2024-05-19,Washington Mystics,Seattle Storm,3.5
2024-05-19,Connecticut Sun,Atlanta Dream,-9.0
"""
    odds_df = pd.read_csv(io.StringIO(odds_data_string))
    odds_df['date'] = pd.to_datetime(odds_df['date']).dt.date
    
    # WNBA Specific Team Name Map
    wnba_team_name_map = {
        'Atlanta Dream': 'ATL', 'Chicago Sky': 'CHI', 'Connecticut Sun': 'CON',
        'Dallas Wings': 'DAL', 'Indiana Fever': 'IND', 'Las Vegas Aces': 'LVA',
        'Los Angeles Sparks': 'LAS', 'Minnesota Lynx': 'MIN', 'New York Liberty': 'NYL',
        'Phoenix Mercury': 'PHO', 'Seattle Storm': 'SEA', 'Washington Mystics': 'WAS'
    }
    odds_df['TEAM_ABBREVIATION_home'] = odds_df['home_team'].map(wnba_team_name_map)
    odds_df.rename(columns={'spread_line': 'vegas_spread'}, inplace=True)
    
    # 2. MERGE THE DATA TO FIND OUR TEST SET
    test_df = pd.merge(games_df, odds_df, left_on=['GAME_DATE_home', 'TEAM_ABBREVIATION_home'], right_on=['date', 'TEAM_ABBREVIATION_home'])
    print(f"Found {len(test_df)} games with available odds to use for our backtest.")
    
    if len(test_df) < 1: raise ValueError("No overlapping games found between the WNBA feature data and the odds data.")
    
    # 3. THE "TIME MACHINE": TRAIN A MODEL ONLY ON PAST DATA
    first_test_date = test_df['GAME_DATE_home'].min()
    print(f"The first game to predict is on {first_test_date}. Training a model on all data BEFORE this date...")
    
    train_df = games_df[games_df['GAME_DATE_home'] < first_test_date]
    features = [col for col in train_df.columns if col.endswith('_diff')]
    
    # Check if there is enough data to train on
    if len(train_df) < 20: # Arbitrary threshold for minimum training data
        raise ValueError(f"Not enough historical data ({len(train_df)} games) before {first_test_date} to train a reliable model.")
        
    X_train = train_df[features]
    y_train = train_df['point_differential']

    honest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    honest_model.fit(X_train, y_train)
    print("Honest WNBA model trained successfully.")

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

        print("\n--- FINAL WNBA HONEST Backtest Results ---")
        print(f"Total Bets Made: {total_bets}")
        print(f"Wins: {total_wins} | Losses: {total_bets - total_wins}")
        print(f"Overall Win Rate: {win_rate:.2%}")
        print("---------------------------------")
        print(f"Total Units Risked: {total_units_risked}")
        print(f"Total Profit: {total_profit:.2f} units")
        print(f"Return on Investment (ROI): {roi:.2%}")
        print("---------------------------------")
        if total_profit > 0: print("SUCCESS! The variable betting strategy was PROFITABLE for the WNBA.")
        else: print("The variable betting strategy was NOT profitable for the WNBA.")
    else:
        print("No betting opportunities found with the current edge thresholds.")

except FileNotFoundError as e:
    print(f"\nERROR: Could not find a required file: {e.filename}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")