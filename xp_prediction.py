import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the shot data for expected points prediction"""
    print("Loading shot data...")
    
    # Load the shot data
    df = pd.read_csv('shot.csv')
    
    # Create target variable (points scored)
    # Determine shot value from SHOT_TYPE
    df['SHOT_VALUE'] = df['SHOT_TYPE'].apply(lambda x: 3 if '3PT' in str(x) else 2)
    
    # Create shot made flag (1 for made shots, 0 for missed shots)
    df['SHOT_MADE_FLAG'] = df['SHOT_MADE'].astype(int)
    df['xP_target'] = df['SHOT_MADE_FLAG'] * df['SHOT_VALUE']
    
    # Debug: Check target distribution
    print(f"Target distribution: {df['xP_target'].value_counts().to_dict()}")
    
    # Primary feature engineering - focus on location and shot type
    print("Adding primary features (location and shot type)...")
    
    # === PRIMARY FEATURES: SHOT LOCATION ===
    # Distance features (most important)
    df['SHOT_DISTANCE_SQUARED'] = df['SHOT_DISTANCE'] ** 2
    df['SHOT_DISTANCE_CUBED'] = df['SHOT_DISTANCE'] ** 3
    
    # Location features
    df['LOC_X_ABS'] = abs(df['LOC_X'])
    df['LOC_Y_ABS'] = abs(df['LOC_Y'])
    
    # Angle to basket (critical for shot difficulty)
    df['ANGLE_TO_BASKET'] = np.arctan2(df['LOC_Y'], df['LOC_X_ABS'])
    
    # Distance from rim
    df['DISTANCE_FROM_RIM'] = np.sqrt(df['LOC_X']**2 + df['LOC_Y']**2)
    
    # Court position features
    df['IS_LEFT_SIDE'] = df['LOC_X'] < 0
    df['IS_RIGHT_SIDE'] = df['LOC_X'] > 0
    df['IS_CENTER'] = (df['LOC_X'] >= -50) & (df['LOC_X'] <= 50)
    
    # === PRIMARY FEATURES: SHOT TYPE ===
    # Shot value features
    df['IS_3PT'] = df['SHOT_VALUE'] == 3
    df['IS_2PT'] = df['SHOT_VALUE'] == 2
    df['IS_FT'] = df['SHOT_VALUE'] == 1
    
    # Zone features (based on shot location and type)
    df['IS_CORNER_3'] = (df['BASIC_ZONE'] == 'Corner 3')
    df['IS_ABOVE_BREAK_3'] = (df['BASIC_ZONE'] == 'Above the Break 3')
    df['IS_RESTRICTED_AREA'] = (df['BASIC_ZONE'] == 'Restricted Area')
    df['IS_MID_RANGE'] = (df['BASIC_ZONE'] == 'Mid-Range')
    df['IS_PAINT_NON_RA'] = (df['BASIC_ZONE'] == 'Paint (Non-RA)')
    
    # Action type features (shot mechanics)
    df['IS_DUNK'] = df['ACTION_TYPE'].str.contains('Dunk', na=False)
    df['IS_LAYUP'] = df['ACTION_TYPE'].str.contains('Layup', na=False)
    df['IS_JUMP_SHOT'] = df['ACTION_TYPE'].str.contains('Jump Shot', na=False)
    df['IS_HOOK'] = df['ACTION_TYPE'].str.contains('Hook', na=False)
    df['IS_FLOATER'] = df['ACTION_TYPE'].str.contains('Floater', na=False)
    df['IS_FADEAWAY'] = df['ACTION_TYPE'].str.contains('Fadeaway', na=False)
    df['IS_STEPBACK'] = df['ACTION_TYPE'].str.contains('Step Back', na=False)
    df['IS_DRIVING'] = df['ACTION_TYPE'].str.contains('Driving', na=False)
    df['IS_CUTTING'] = df['ACTION_TYPE'].str.contains('Cutting', na=False)
    df['IS_PUTBACK'] = df['ACTION_TYPE'].str.contains('Putback', na=False)
    df['IS_ALLEY_OOP'] = df['ACTION_TYPE'].str.contains('Alley Oop', na=False)
    
    # === SECONDARY FEATURES: GAME SITUATION ===
    df['GAME_CLOCK'] = df['MINS_LEFT'] * 60 + df['SECS_LEFT']
    df['QUARTER_TIME_REMAINING'] = df['GAME_CLOCK']
    df['IS_LATE_GAME'] = (df['QUARTER'] >= 4) & (df['GAME_CLOCK'] < 300)  # Last 5 minutes of 4th quarter
    
    print(f"Processed {len(df)} shots with primary location and shot type features")
    
    return df

def add_contextual_features(df):
    """Add contextual features from player_context_stats.csv (secondary importance)"""
    print("Adding contextual features (secondary importance)...")
    
    try:
        # Load contextual stats
        context_stats = pd.read_csv('player_context_stats.csv')
        
        # Merge with main dataframe
        df = df.merge(context_stats, on=['PLAYER_NAME', 'TEAM_NAME'], how='left')
        
        # Fill missing values with 0 (neutral)
        df = df.fillna(0)
        
        print(f"Added contextual features, dataset now has {len(df.columns)} columns")
        
    except Exception as e:
        print(f"Could not load contextual features: {e}")
    
    return df

def select_features(df):
    """Select features with balanced focus on location, shot type, and contextual factors"""
    # === PRIMARY FEATURES (High Importance) ===
    primary_features = [
        # Shot location features (most important)
        'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SHOT_DISTANCE_SQUARED', 'SHOT_DISTANCE_CUBED',
        'LOC_X_ABS', 'LOC_Y_ABS', 'ANGLE_TO_BASKET', 'DISTANCE_FROM_RIM',
        'IS_LEFT_SIDE', 'IS_RIGHT_SIDE', 'IS_CENTER',
        
        # Shot type features (most important)
        'IS_3PT', 'IS_2PT', 'IS_FT',
        'IS_CORNER_3', 'IS_ABOVE_BREAK_3', 'IS_RESTRICTED_AREA', 'IS_MID_RANGE', 'IS_PAINT_NON_RA',
        'IS_DUNK', 'IS_LAYUP', 'IS_JUMP_SHOT', 'IS_HOOK', 'IS_FLOATER', 'IS_FADEAWAY',
        'IS_STEPBACK', 'IS_DRIVING', 'IS_CUTTING', 'IS_PUTBACK', 'IS_ALLEY_OOP',
    ]
    
    # === CONTEXTUAL FEATURES (Medium Importance) ===
    contextual_features = [
        # Game situation features
        'QUARTER', 'GAME_CLOCK', 'QUARTER_TIME_REMAINING', 'IS_LATE_GAME',
        
        # Player contextual features (increased importance)
        'FG_PCT', 'FG3_PCT', 'FT_PCT'
    ]
    
    # Combine all features
    all_features = primary_features + contextual_features
    
    # Filter to only include features that exist in the dataframe
    available_features = [col for col in all_features if col in df.columns]
    
    print(f"Using {len(available_features)} features:")
    print(f"  Primary features (location/shot type): {len([f for f in available_features if f in primary_features])}")
    print(f"  Contextual features (game context + player stats): {len([f for f in available_features if f in contextual_features])}")
    print(f"  Primary features: {available_features[:15]}...")
    
    return df[available_features], df['xP_target']

def train_location_shot_type_model(X, y):
    """Train model with focus on location and shot type features"""
    print("Training model focused on location and shot type...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest with focus on primary features
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Random Forest Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Train XGBoost with similar focus
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    
    print(f"\nXGBoost Performance:")
    print(f"MSE: {mse_xgb:.4f}")
    print(f"R²: {r2_xgb:.4f}")
    print(f"MAE: {mae_xgb:.4f}")
    
    # Choose the better model
    if r2_xgb > r2:
        print(f"\nBest model: XGBoost with R² = {r2_xgb:.4f}")
        return xgb_model
    else:
        print(f"\nBest model: Random Forest with R² = {r2:.4f}")
        return rf_model

def generate_predictions(df, model):
    """Generate predictions for all shots"""
    print("Generating predictions...")
    
    # Prepare features
    X, y = select_features(df)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Add predictions to dataframe
    df['xP_pred'] = predictions
    
    # Debug info
    print(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
    print(f"Target range: {y.min():.4f} to {y.max():.4f}")
    
    return df

def create_player_summary(df):
    """Create player summary with expected points in the format expected by the dashboard"""
    print("Creating player summary...")
    
    # Calculate games played for each player
    print("Calculating games played...")
    games_played = df.groupby('PLAYER_NAME')['GAME_ID'].nunique()
    
    # Shot type mapping for grouping similar shot types
    shot_type_mapping = {
        "Alley Oop Dunk Shot": "Dunk",
        "Alley Oop Layup shot": "Layup",
        "Cutting Dunk Shot": "Dunk",
        "Cutting Finger Roll Layup Shot": "Layup",
        "Cutting Layup Shot": "Layup",
        "Driving Bank Hook Shot": "Hook Shot",
        "Driving Dunk Shot": "Dunk",
        "Driving Finger Roll Layup Shot": "Layup",
        "Driving Floating Bank Jump Shot": "Floater",
        "Driving Floating Jump Shot": "Floater",
        "Driving Hook Shot": "Hook Shot",
        "Driving Layup Shot": "Layup",
        "Driving Reverse Dunk Shot": "Dunk",
        "Driving Reverse Layup Shot": "Layup",
        "Dunk Shot": "Dunk",
        "Fadeaway Bank shot": "Fadeaway",
        "Fadeaway Jump Shot": "Fadeaway",
        "Finger Roll Layup Shot": "Layup",
        "Floating Jump shot": "Floater",
        "Hook Bank Shot": "Hook Shot",
        "Hook Shot": "Hook Shot",
        "Jump Bank Shot": "Jump Shot",
        "Jump Shot": "Jump Shot",
        "Layup Shot": "Layup",
        "Pullup Jump shot": "Jump Shot",
        "Putback Dunk Shot": "Dunk",
        "Putback Layup Shot": "Layup",
        "Reverse Dunk Shot": "Dunk",
        "Reverse Layup Shot": "Layup",
        "Running Alley Oop Dunk Shot": "Dunk",
        "Running Alley Oop Layup Shot": "Layup",
        "Running Dunk Shot": "Dunk",
        "Running Finger Roll Layup Shot": "Layup",
        "Running Jump Shot": "Jump Shot",
        "Running Layup Shot": "Layup",
        "Running Pull-Up Jump Shot": "Jump Shot",
        "Running Reverse Dunk Shot": "Dunk",
        "Running Reverse Layup Shot": "Layup",
        "Step Back Bank Jump Shot": "Jump Shot",
        "Step Back Jump shot": "Jump Shot",
        "Tip Dunk Shot": "Dunk",
        "Tip Layup Shot": "Layup",
        "Turnaround Bank Hook Shot": "Hook Shot",
        "Turnaround Bank shot": "Fadeaway",
        "Turnaround Fadeaway Bank Jump Shot": "Fadeaway",
        "Turnaround Fadeaway shot": "Fadeaway",
        "Turnaround Hook Shot": "Hook Shot",
        "Turnaround Jump Shot": "Jump Shot"
    }
    
    # Add shot type category to dataframe
    df['SHOT_TYPE_CATEGORY'] = df['ACTION_TYPE'].map(shot_type_mapping).fillna('Other')
    
    # Group by player and calculate statistics (merge players who played on multiple teams)
    player_stats = df.groupby('PLAYER_NAME').agg({
        'xP_pred': ['sum', 'mean'],
        'xP_target': ['sum', 'mean'],
        'SHOT_MADE': 'count',
        'TEAM_NAME': lambda x: ', '.join(x.unique())  # Combine team names
    }).round(4)
    
    # Flatten column names
    player_stats.columns = ['Total_xP', 'xP_per_Shot', 'Total_Points', 'Points_per_Shot', 'Total_Shots', 'TEAM_NAME']
    
    # Calculate overperformance
    player_stats['Overperformance'] = player_stats['Total_Points'] - player_stats['Total_xP']
    player_stats['Overperf_per_Shot'] = player_stats['Overperformance'] / player_stats['Total_Shots']
    
    # Reset index
    player_stats = player_stats.reset_index()
    
    # Create the format expected by the dashboard
    dashboard_data = []
    
    for _, player in player_stats.iterrows():
        # Get games played for this player
        player_games = games_played.get(player['PLAYER_NAME'], 0)
        
        # Overall stats
        dashboard_data.append({
            'PLAYER_NAME': player['PLAYER_NAME'],
            'TEAM_NAME': player['TEAM_NAME'],
            'Category': 'Overall',
            'Category_Value': 'Overall',
            'Shots': int(player['Total_Shots']),
            'Points': int(player['Total_Points']),
            'xP': round(player['Total_xP'], 2),
            'Overperformance': round(player['Overperformance'], 2),
            'xP_per_Shot': round(player['xP_per_Shot'], 4),
            'Overperf_per_Shot': round(player['Overperf_per_Shot'], 4),
            'GamesPlayed': int(player_games)
        })
        
        # Shot type breakdowns (using mapped categories)
        player_shots = df[df['PLAYER_NAME'] == player['PLAYER_NAME']]
        
        # Shot type aggregations using mapped categories
        shot_types = player_shots.groupby('SHOT_TYPE_CATEGORY').agg({
            'xP_pred': 'sum',
            'xP_target': 'sum',
            'SHOT_MADE': 'count'
        }).reset_index()
        
        for _, shot_type in shot_types.iterrows():
            if shot_type['SHOT_MADE'] >= 5:  # Only include if at least 5 shots
                xp_per_shot = shot_type['xP_pred'] / shot_type['SHOT_MADE']
                overperformance = shot_type['xP_target'] - shot_type['xP_pred']
                overperf_per_shot = overperformance / shot_type['SHOT_MADE']
                
                dashboard_data.append({
                    'PLAYER_NAME': player['PLAYER_NAME'],
                    'TEAM_NAME': player['TEAM_NAME'],
                    'Category': 'Shot_Type',
                    'Category_Value': shot_type['SHOT_TYPE_CATEGORY'],
                    'Shots': int(shot_type['SHOT_MADE']),
                    'Points': int(shot_type['xP_target']),
                    'xP': round(shot_type['xP_pred'], 2),
                    'Overperformance': round(overperformance, 2),
                    'xP_per_Shot': round(xp_per_shot, 4),
                    'Overperf_per_Shot': round(overperf_per_shot, 4),
                    'GamesPlayed': int(player_games)
                })
        
        # Shot zone aggregations (using BASIC_ZONE)
        shot_zones = player_shots.groupby('BASIC_ZONE').agg({
            'xP_pred': 'sum',
            'xP_target': 'sum',
            'SHOT_MADE': 'count'
        }).reset_index()
        
        for _, shot_zone in shot_zones.iterrows():
            if shot_zone['SHOT_MADE'] >= 5:  # Only include if at least 5 shots
                xp_per_shot = shot_zone['xP_pred'] / shot_zone['SHOT_MADE']
                overperformance = shot_zone['xP_target'] - shot_zone['xP_pred']
                overperf_per_shot = overperformance / shot_zone['SHOT_MADE']
                
                dashboard_data.append({
                    'PLAYER_NAME': player['PLAYER_NAME'],
                    'TEAM_NAME': player['TEAM_NAME'],
                    'Category': 'Shot_Zone',
                    'Category_Value': shot_zone['BASIC_ZONE'],
                    'Shots': int(shot_zone['SHOT_MADE']),
                    'Points': int(shot_zone['xP_target']),
                    'xP': round(shot_zone['xP_pred'], 2),
                    'Overperformance': round(overperformance, 2),
                    'xP_per_Shot': round(xp_per_shot, 4),
                    'Overperf_per_Shot': round(overperf_per_shot, 4),
                    'GamesPlayed': int(player_games)
                })
    
    # Convert to DataFrame
    dashboard_df = pd.DataFrame(dashboard_data)
    
    # Sort by overall overperformance
    overall_data = dashboard_df[dashboard_df['Category'] == 'Overall'].sort_values('Overperformance', ascending=False)
    
    print(f"\nPlayer Summary (Top 10):")
    print(overall_data.head(10)[['PLAYER_NAME', 'TEAM_NAME', 'Shots', 'Points', 'xP', 'Overperformance']])
    
    print(f"\nTotal players analyzed: {len(overall_data)}")
    print(f"Total data rows generated: {len(dashboard_df)}")
    
    return dashboard_df

def main():
    """Main function to run the expected points prediction"""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Add contextual features (secondary importance)
    df = add_contextual_features(df)
    
    # Select features and target
    X, y = select_features(df)
    
    # Remove rows with NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset: {len(X)} shots with {len(X.columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train model focused on location and shot type
    model = train_location_shot_type_model(X, y)
    
    # Generate predictions
    df_clean = df[mask].copy()
    df_clean = generate_predictions(df_clean, model)
    
    # Create player summary
    player_summary = create_player_summary(df_clean)
    
    # Save results
    player_summary.to_csv('xp_comprehensive.csv', index=False)
    print("\nResults saved to xp_comprehensive.csv")
    
    print("\nModel training completed with focus on location and shot type!")

if __name__ == "__main__":
    main() 