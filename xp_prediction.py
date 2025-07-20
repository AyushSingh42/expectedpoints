import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ftfy import fix_text
import warnings
warnings.filterwarnings('ignore')

import csv_additions
from xp_contextual_additions import approximate_context_from_action, lookup_fg_pct

# Shot type mapping to group detailed action types into broader categories
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

def add_context_fg_pct(df):
    """
    Add contextual FG% columns using approximated context bins.
    """
    context_features = df.apply(
        lambda row: lookup_fg_pct(
            row['PLAYER_NAME'],
            *approximate_context_from_action(row['ACTION_TYPE'])
        ), axis=1, result_type='expand')
    return pd.concat([df, context_features], axis=1)

def load_and_preprocess_data():
    """Load and preprocess the shot data"""
    print("Loading shot data...")
    df = pd.read_csv("shot.csv", encoding='utf-8')

    # Filter to shot attempts only
    df_clean = df[df["EVENT_TYPE"].isin(["Made Shot", "Missed Shot"])].copy()

    # Remove shots from beyond half court
    df_clean["LOC_X"] = pd.to_numeric(df_clean["LOC_X"], errors="coerce")
    df_clean["LOC_Y"] = pd.to_numeric(df_clean["LOC_Y"], errors="coerce")
    df_clean = df_clean[(df_clean["LOC_Y"] <= 470) & (df_clean["LOC_Y"] >= 0) & (df_clean["LOC_X"].abs() <= 250)]

    # Create target and features
    df_clean["SHOT_TYPE"] = df_clean["SHOT_TYPE"].str.strip()
    df_clean["POINT_VALUE"] = df_clean["SHOT_TYPE"].apply(lambda x: 3 if "3PT" in x else 2)
    df_clean["SHOT_MADE_FLAG"] = df_clean["SHOT_MADE"].apply(lambda x: 1 if str(x).upper() == "TRUE" else 0)
    df_clean["xP_target"] = df_clean["SHOT_MADE_FLAG"] * df_clean["POINT_VALUE"]

    # Create additional features
    df_clean["GAME_CLOCK"] = df_clean["MINS_LEFT"] * 60 + df_clean["SECS_LEFT"]
    df_clean["SHOT_DISTANCE_SQUARED"] = df_clean["SHOT_DISTANCE"] ** 2
    df_clean["LOC_X_ABS"] = df_clean["LOC_X"].abs()

    # Create shot type categories
    df_clean["SHOT_TYPE_CATEGORY"] = df_clean["ACTION_TYPE"].map(shot_type_mapping).fillna("Other")

    # Add contextual FG% stats
    df_clean = add_context_fg_pct(df_clean)

    print(f"Processed {len(df_clean)} shots")
    return df_clean

def train_model_with_cross_validation(X, y):
    """Train model with proper cross-validation"""
    print("Training model with cross-validation...")
    
    # Define features
    categorical_features = ["ACTION_TYPE", "SHOT_TYPE", "QUARTER", "SHOT_TYPE_CATEGORY", "BASIC_ZONE"]
    numerical_features = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_DISTANCE_SQUARED", "GAME_CLOCK", "LOC_X_ABS"]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("scaler", StandardScaler(), numerical_features)
    ])
    
    # Try different models
    models = {
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_score = -np.inf
    best_model_name = ""
    
    for name, model in models.items():
        pipeline = make_pipeline(preprocessor, model)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        mean_score = cv_scores.mean()
        
        print(f"{name} CV R² Score: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with R² = {best_score:.4f}")
    
    # Train the best model on full dataset
    best_model.fit(X, y)
    
    return best_model

def main():
    """Main function to run the improved prediction pipeline"""
    
    # Load and preprocess data
    df_clean = load_and_preprocess_data()
    
    # Prepare features and target
    features = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_DISTANCE_SQUARED", "QUARTER", "GAME_CLOCK", 
                "ACTION_TYPE", "SHOT_TYPE", "SHOT_TYPE_CATEGORY", "BASIC_ZONE", "LOC_X_ABS"]
    target = "xP_target"
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Split data properly (this is for final evaluation, not for training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with cross-validation
    model = train_model_with_cross_validation(X_train, y_train)
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nTest Set Performance:")
    print(f"MSE: {test_mse:.4f}")
    print(f"R²: {test_r2:.4f}")
    
    # Predict on full dataset for analysis
    print("\nGenerating predictions for full dataset...")
    df_clean["xP_pred"] = model.predict(X)
    df_clean["Points"] = df_clean["xP_target"]
    
    # Generate summaries (same as before)
    generate_summaries(df_clean)
    
    print("\nModel training completed with proper cross-validation!")

def generate_summaries(df_clean):
    """Generate the same summaries as the original script"""
    
    # Get current team for each player
    player_current_team = df_clean.groupby("PLAYER_NAME")["TEAM_NAME"].last().reset_index()
    player_current_team.columns = ["PLAYER_NAME", "CURRENT_TEAM"]
    
    # Action Type aggregations
    action_type_summary = df_clean.groupby(["PLAYER_NAME", "ACTION_TYPE"]).agg(
        Shots=("Points", "count"),
        Points=("Points", "sum"),
        xP=("xP_pred", "sum")
    ).reset_index()
    
    action_type_summary["Overperformance"] = action_type_summary["Points"] - action_type_summary["xP"]
    action_type_summary["xP_per_Shot"] = action_type_summary["xP"] / action_type_summary["Shots"]
    action_type_summary["Overperf_per_Shot"] = action_type_summary["Overperformance"] / action_type_summary["Shots"]
    
    # Shot Zone aggregations
    shot_zone_summary = df_clean.groupby(["PLAYER_NAME", "BASIC_ZONE"]).agg(
        Shots=("Points", "count"),
        Points=("Points", "sum"),
        xP=("xP_pred", "sum")
    ).reset_index()
    
    shot_zone_summary["Overperformance"] = shot_zone_summary["Points"] - shot_zone_summary["xP"]
    shot_zone_summary["xP_per_Shot"] = shot_zone_summary["xP"] / shot_zone_summary["Shots"]
    shot_zone_summary["Overperf_per_Shot"] = shot_zone_summary["Overperformance"] / shot_zone_summary["Shots"]
    
    # Player-level summary
    player_summary = df_clean.groupby("PLAYER_NAME").agg(
        Total_Shots=("Points", "count"),
        Total_Points=("Points", "sum"),
        Total_xP=("xP_pred", "sum")
    ).reset_index()
    
    player_summary = player_summary.merge(player_current_team, on="PLAYER_NAME", how="left")
    player_summary = player_summary.rename(columns={"CURRENT_TEAM": "TEAM_NAME"})
    
    player_summary["Overperformance"] = player_summary["Total_Points"] - player_summary["Total_xP"]
    player_summary["xP_per_Shot"] = player_summary["Total_xP"] / player_summary["Total_Shots"]
    player_summary["Overperf_per_Shot"] = player_summary["Overperformance"] / player_summary["Total_Shots"]
    
    player_summary = player_summary.sort_values("Overperf_per_Shot", ascending=False)
    
    # Create comprehensive CSV
    create_comprehensive_csv(player_summary, action_type_summary, shot_zone_summary)
    
    # Preview
    print("Overall Player Summary (Top 10):")
    print(player_summary.head(10))
    print(f"\nTotal players analyzed: {len(player_summary)}")

def create_comprehensive_csv(player_summary, action_type_summary, shot_zone_summary):
    """Create the comprehensive CSV with all breakdowns"""
    comprehensive_data = []
    
    for _, player in player_summary.iterrows():
        player_name = player['PLAYER_NAME']
        
        # Overall stats
        base_row = {
            'PLAYER_NAME': player_name,
            'TEAM_NAME': player['TEAM_NAME'],
            'Category': 'Overall',
            'Category_Value': 'Overall',
            'Shots': player['Total_Shots'],
            'Points': player['Total_Points'],
            'xP': player['Total_xP'],
            'Overperformance': player['Overperformance'],
            'xP_per_Shot': player['xP_per_Shot'],
            'Overperf_per_Shot': player['Overperf_per_Shot']
        }
        comprehensive_data.append(base_row)
        
        # Action type breakdowns
        player_action_types = action_type_summary[action_type_summary['PLAYER_NAME'] == player_name]
        for _, action_row in player_action_types.iterrows():
            action_data = {
                'PLAYER_NAME': player_name,
                'TEAM_NAME': player['TEAM_NAME'],
                'Category': 'Action_Type',
                'Category_Value': action_row['ACTION_TYPE'],
                'Shots': action_row['Shots'],
                'Points': action_row['Points'],
                'xP': action_row['xP'],
                'Overperformance': action_row['Overperformance'],
                'xP_per_Shot': action_row['xP_per_Shot'],
                'Overperf_per_Shot': action_row['Overperf_per_Shot']
            }
            comprehensive_data.append(action_data)
        
        # Shot type breakdowns (using mapping)
        player_action_types = action_type_summary[action_type_summary['PLAYER_NAME'] == player_name]
        shot_type_data = {}
        
        for _, action_row in player_action_types.iterrows():
            action_type = action_row['ACTION_TYPE']
            shot_type = shot_type_mapping.get(action_type, action_type)
            
            if shot_type not in shot_type_data:
                shot_type_data[shot_type] = {'Shots': 0, 'Points': 0, 'xP': 0}
            
            shot_type_data[shot_type]['Shots'] += action_row['Shots']
            shot_type_data[shot_type]['Points'] += action_row['Points']
            shot_type_data[shot_type]['xP'] += action_row['xP']
        
        for shot_type, stats in shot_type_data.items():
            if stats['Shots'] > 0:
                overperformance = stats['Points'] - stats['xP']
                xp_per_shot = stats['xP'] / stats['Shots']
                overperf_per_shot = overperformance / stats['Shots']
                
                shot_type_data = {
                    'PLAYER_NAME': player_name,
                    'TEAM_NAME': player['TEAM_NAME'],
                    'Category': 'Shot_Type',
                    'Category_Value': shot_type,
                    'Shots': stats['Shots'],
                    'Points': stats['Points'],
                    'xP': stats['xP'],
                    'Overperformance': overperformance,
                    'xP_per_Shot': xp_per_shot,
                    'Overperf_per_Shot': overperf_per_shot
                }
                comprehensive_data.append(shot_type_data)
        
        # Shot zone breakdowns
        player_shot_zones = shot_zone_summary[shot_zone_summary['PLAYER_NAME'] == player_name]
        for _, zone_row in player_shot_zones.iterrows():
            zone_data = {
                'PLAYER_NAME': player_name,
                'TEAM_NAME': player['TEAM_NAME'],
                'Category': 'Shot_Zone',
                'Category_Value': zone_row['BASIC_ZONE'],
                'Shots': zone_row['Shots'],
                'Points': zone_row['Points'],
                'xP': zone_row['xP'],
                'Overperformance': zone_row['Overperformance'],
                'xP_per_Shot': zone_row['xP_per_Shot'],
                'Overperf_per_Shot': zone_row['Overperf_per_Shot']
            }
            comprehensive_data.append(zone_data)
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_df.to_csv('xp_comprehensive.csv', index=False, encoding='utf-8')
    csv_additions.add_games_played_to_csv('xp_comprehensive.csv', 'xp_comprehensive.csv')

if __name__ == "__main__":
    main() 