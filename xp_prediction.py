import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from ftfy import fix_text

import csv_additions

# Load CSV
df = pd.read_csv("/Users/ayushsingh/Desktop/repos/shots.csv",  encoding='utf-8')

# Step 1: Filter to shot attempts only
df_clean = df[df["SHOT_ATTEMPTED_FLAG"] == 1].copy()

# Step 2: Remove shots from beyond half court (NBA court is 94 ft long, so Y > 470 = half court)
df_clean["LOC_X"] = pd.to_numeric(df_clean["LOC_X"], errors="coerce")
df_clean["LOC_Y"] = pd.to_numeric(df_clean["LOC_Y"], errors="coerce")
df_clean = df_clean[(df_clean["LOC_Y"] <= 470) & (df_clean["LOC_Y"] >= 0) & (df_clean["LOC_X"].abs() <= 250)]

# Step 3: Create target column (expected points = made_flag * point value)
df_clean["SHOT_TYPE"] = df_clean["SHOT_TYPE"].str.strip()
df_clean["POINT_VALUE"] = df_clean["SHOT_TYPE"].apply(lambda x: 3 if "3PT" in x else 2)
df_clean["xP_target"] = df_clean["SHOT_MADE_FLAG"] * df_clean["POINT_VALUE"]

# Step 4: Create features
df_clean["GAME_CLOCK"] = df_clean["MINUTES_REMAINING"] * 60 + df_clean["SECONDS_REMAINING"]

features = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "PERIOD", "GAME_CLOCK", "ACTION_TYPE", "SHOT_TYPE"]
target = "xP_target"

X = df_clean[features]
y = df_clean[target]

# Step 5: Encode categorical features
categorical = ["ACTION_TYPE", "SHOT_TYPE", "PERIOD"]
numerical = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "GAME_CLOCK"]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder='passthrough')

# Step 6: Build model pipeline
model = make_pipeline(
    preprocessor,
    XGBRegressor(objective="reg:squarederror", max_depth=5, n_estimators=100, learning_rate=0.1)
)

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Optional: Print model error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse:.3f}")

# Step 8: Predict xP for full dataset
df_clean["xP_pred"] = model.predict(X)
df_clean["Points"] = df_clean["xP_target"]

# Step 9: Get the most recent team for each player (assuming data is ordered chronologically)
# If you have a date column, you can use that instead of index
player_current_team = df_clean.groupby("PLAYER_NAME")["TEAM_NAME"].last().reset_index()
player_current_team.columns = ["PLAYER_NAME", "CURRENT_TEAM"]

# Step 10: Create Action Type aggregations
action_type_summary = df_clean.groupby(["PLAYER_NAME", "ACTION_TYPE"]).agg(
    Shots=("Points", "count"),
    Points=("Points", "sum"),
    xP=("xP_pred", "sum")
).reset_index()

action_type_summary["Overperformance"] = action_type_summary["Points"] - action_type_summary["xP"]
action_type_summary["xP_per_Shot"] = action_type_summary["xP"] / action_type_summary["Shots"]
action_type_summary["Overperf_per_Shot"] = action_type_summary["Overperformance"] / action_type_summary["Shots"]

# Step 11: Create Shot Zone aggregations
shot_zone_summary = df_clean.groupby(["PLAYER_NAME", "SHOT_ZONE_BASIC"]).agg(
    Shots=("Points", "count"),
    Points=("Points", "sum"),
    xP=("xP_pred", "sum")
).reset_index()

shot_zone_summary["Overperformance"] = shot_zone_summary["Points"] - shot_zone_summary["xP"]
shot_zone_summary["xP_per_Shot"] = shot_zone_summary["xP"] / shot_zone_summary["Shots"]
shot_zone_summary["Overperf_per_Shot"] = shot_zone_summary["Overperformance"] / shot_zone_summary["Shots"]

# Step 12: Aggregate player-level summary (now only by player, not by team)
player_summary = df_clean.groupby("PLAYER_NAME").agg(
    Total_Shots=("Points", "count"),
    Total_Points=("Points", "sum"),
    Total_xP=("xP_pred", "sum")
).reset_index()

# Step 13: Add current team information
player_summary = player_summary.merge(player_current_team, on="PLAYER_NAME", how="left")
player_summary = player_summary.rename(columns={"CURRENT_TEAM": "TEAM_NAME"})

# Step 14: Calculate performance metrics
player_summary["Overperformance"] = player_summary["Total_Points"] - player_summary["Total_xP"]
player_summary["xP_per_Shot"] = player_summary["Total_xP"] / player_summary["Total_Shots"]
player_summary["Overperf_per_Shot"] = player_summary["Overperformance"] / player_summary["Total_Shots"]

# Sort by overperformance per shot
player_summary = player_summary.sort_values("Overperf_per_Shot", ascending=False)

# Step 15: Save all CSVs
player_summary.to_csv('xp.csv', index=False, encoding='utf-8')
action_type_summary.to_csv('xp_by_action_type.csv', index=False, encoding='utf-8')
shot_zone_summary.to_csv('xp_by_shot_zone.csv', index=False, encoding='utf-8')

# Apply additional processing to main CSV
csv_additions.add_games_played_to_csv('xp.csv')
csv_additions.add_team_to_csv('xp.csv')

# Step 16: Create a comprehensive CSV with all breakdowns
# This will create a master CSV where each player has their overall stats plus breakdowns
comprehensive_data = []

for _, player in player_summary.iterrows():
    player_name = player['PLAYER_NAME']
    
    # Add overall player stats
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
    
    # Add action type breakdowns
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
    
    # Add shot zone breakdowns
    player_shot_zones = shot_zone_summary[shot_zone_summary['PLAYER_NAME'] == player_name]
    for _, zone_row in player_shot_zones.iterrows():
        zone_data = {
            'PLAYER_NAME': player_name,
            'TEAM_NAME': player['TEAM_NAME'],
            'Category': 'Shot_Zone',
            'Category_Value': zone_row['SHOT_ZONE_BASIC'],
            'Shots': zone_row['Shots'],
            'Points': zone_row['Points'],
            'xP': zone_row['xP'],
            'Overperformance': zone_row['Overperformance'],
            'xP_per_Shot': zone_row['xP_per_Shot'],
            'Overperf_per_Shot': zone_row['Overperf_per_Shot']
        }
        comprehensive_data.append(zone_data)

# Create comprehensive DataFrame and save
comprehensive_df = pd.DataFrame(comprehensive_data)
comprehensive_df.to_csv('xp_comprehensive.csv', index=False, encoding='utf-8')

# Preview
print("Overall Player Summary:")
print(player_summary.head(10))
print("\nAction Type Breakdown Sample:")
print(action_type_summary.head(10))
print("\nShot Zone Breakdown Sample:")
print(shot_zone_summary.head(10))
print("\nComprehensive CSV created with all breakdowns!")