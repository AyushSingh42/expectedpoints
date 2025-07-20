from nba_api.stats.static import players
from nba_api.stats.endpoints import CommonPlayerInfo
from nba_api.stats.endpoints import PlayerDashPtShots
import pandas as pd
import time
import os

from xp_contextual_additions import fetch_player_context_stats

# Load existing saved CSV if it exists
csv_path = "active_players_with_teams.csv"

if os.path.exists(csv_path):
    saved_df = pd.read_csv(csv_path)
    saved_player_ids = set(saved_df["PLAYER_ID"])
    active_player_ids = saved_df.to_records(index=False).tolist()
else:
    saved_player_ids = set()
    active_player_ids = []

# Get all active players
active_players = [p for p in players.get_players() if p['is_active']]

# Loop through and only fetch new players
for i, player in enumerate(active_players):
    player_id = player['id']
    full_name = player['full_name']

    if player_id in saved_player_ids:
        print(f"{i+1}/{len(active_players)}: Skipping {full_name} (already saved)")
        continue

    try:
        info = CommonPlayerInfo(player_id=player_id)
        team_id = info.get_data_frames()[0].loc[0, 'TEAM_ID']
        active_player_ids.append((player_id, full_name, team_id))
        print(f"{i+1}/{len(active_players)}: Added {full_name} → TEAM_ID {team_id}")
        time.sleep(1.2)
    except Exception as e:
        print(f"{i+1}/{len(active_players)}: Failed for {full_name}: {e}")
        continue

# Save updated CSV
df = pd.DataFrame(active_player_ids, columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"])
df.to_csv(csv_path, index=False)
print(f"\nSaved {len(df)} players to {csv_path}")

# dash = PlayerDashPtShots(player_id=active_player_ids[0][0], team_id=active_player_ids[0][2])
# dfs = dash.get_data_frames()  # This returns a list of all DataFrames returned

# # Example: print all tables
# for i, df in enumerate(dfs):
#     print(f"\n=== Table {i} ===")
#     print(df)

print("\nFetching contextual shooting stats...")
context_df = fetch_player_context_stats(active_player_ids)
context_df.to_csv("player_context_stats.csv", index=False)
print("Saved contextual stats to player_context_stats.csv")
