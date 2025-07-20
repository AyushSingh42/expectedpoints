# xp_contextual_additions.py

from nba_api.stats.endpoints import PlayerDashPtShots
import pandas as pd
import time
import os

# ----------------------------------------
# A. Get contextual stats for players
# ----------------------------------------
def fetch_player_context_stats(player_ids, season='2024-25', delay=1.2, output_csv='player_context_stats.csv'):
    """
    Query NBA API for player-level contextual FG% stats by touch time, dribbles, shot clock, and closest defender.
    Returns a DataFrame of contextual shooting stats.
    """
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        existing_players = set(existing_df['Player'])
        data = existing_df.to_dict(orient='list')
    else:
        existing_players = set()
        data = {
            'Player': [],
            'ContextType': [],
            'ContextValue': [],
            'FG_PCT': []
        }

    for player_id, player_name, team_id in player_ids:
        if player_name in existing_players:
            print(f"Skipping {player_name} (already fetched)")
            continue

        try:
            dash = PlayerDashPtShots(player_id=player_id, team_id=team_id, season=season)
            dfs = dash.get_data_frames()

            # Touch Time (Table 6)
            try:
                df_touch = dfs[6]
                for _, row in df_touch.iterrows():
                    data['Player'].append(player_name)
                    data['ContextType'].append('TouchTime')
                    data['ContextValue'].append(row['TOUCH_TIME_RANGE'])
                    data['FG_PCT'].append(row['FG_PCT'])
            except Exception:
                print(f"Missing Touch Time for {player_name}")

            # Dribbles (Table 3)
            try:
                df_dribbles = dfs[3]
                for _, row in df_dribbles.iterrows():
                    data['Player'].append(player_name)
                    data['ContextType'].append('DribbleRange')
                    data['ContextValue'].append(row['DRIBBLE_RANGE'])
                    data['FG_PCT'].append(row['FG_PCT'])
            except Exception:
                print(f"Missing Dribble Range for {player_name}")

            # Defender Distance (Table 4)
            try:
                df_defender = dfs[4]
                for _, row in df_defender.iterrows():
                    data['Player'].append(player_name)
                    data['ContextType'].append('DefenderDistance')
                    data['ContextValue'].append(row['CLOSE_DEF_DIST_RANGE'])
                    data['FG_PCT'].append(row['FG_PCT'])
            except Exception:
                print(f"Missing Defender Distance for {player_name}")

            # Shot Clock (Table 2)
            try:
                df_clock = dfs[2]
                for _, row in df_clock.iterrows():
                    data['Player'].append(player_name)
                    data['ContextType'].append('ShotClock')
                    data['ContextValue'].append(row['SHOT_CLOCK_RANGE'])
                    data['FG_PCT'].append(row['FG_PCT'])
            except Exception:
                print(f"Missing Shot Clock for {player_name}")

            print(f"Fetched: {player_name}")
            time.sleep(delay)

        except Exception as e:
            print(f"Failed for {player_name}: {e}")
            continue

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df


# ----------------------------------------
# B. Example: Approximate context per ACTION_TYPE
# ----------------------------------------

def approximate_context_from_action(action_type):
    """
    Map ACTION_TYPE to estimated contextual bins (used to look up FG% later).
    """
    action_type = action_type.lower()

    if "catch" in action_type:
        return ("Touch < 2 Seconds", "0 Dribbles", "6+ Feet - Wide Open", "22-18 Very Early")
    elif "step back" in action_type or "pull" in action_type:
        return ("Touch 2-6 Seconds", "3-6 Dribbles", "2-4 Feet - Tight", "18-15 Early")
    elif "driving" in action_type or "layup" in action_type:
        return ("Touch < 2 Seconds", "2 Dribbles", "0-2 Feet - Very Tight", "< 4")
    elif "floater" in action_type:
        return ("Touch 2-6 Seconds", "1 Dribble", "2-4 Feet - Tight", "14-7 Average")
    else:
        return ("Touch 2-6 Seconds", "1 Dribble", "4-6 Feet - Open", "14-7 Average")
    
def lookup_fg_pct(player, touch_time, dribble_range, defender_dist, shot_clock):
    """
    Lookup contextual FG% using player name and contextual bins.
    If any combination is missing, returns NaN.
    """
    # Load contextual stats once globally
    context_stats = pd.read_csv("player_context_stats.csv")

    try:
        tt = context_stats[(context_stats['Player'] == player) &
                           (context_stats['ContextType'] == 'TouchTime') &
                           (context_stats['ContextValue'] == touch_time)]['FG_PCT'].values[0]
    except IndexError:
        tt = float('nan')

    try:
        dr = context_stats[(context_stats['Player'] == player) &
                           (context_stats['ContextType'] == 'DribbleRange') &
                           (context_stats['ContextValue'] == dribble_range)]['FG_PCT'].values[0]
    except IndexError:
        dr = float('nan')

    try:
        dd = context_stats[(context_stats['Player'] == player) &
                           (context_stats['ContextType'] == 'DefenderDistance') &
                           (context_stats['ContextValue'] == defender_dist)]['FG_PCT'].values[0]
    except IndexError:
        dd = float('nan')

    try:
        sc = context_stats[(context_stats['Player'] == player) &
                           (context_stats['ContextType'] == 'ShotClock') &
                           (context_stats['ContextValue'] == shot_clock)]['FG_PCT'].values[0]
    except IndexError:
        sc = float('nan')

    return {
        'TouchTimeFG%': tt,
        'DribbleRangeFG%': dr,
        'DefenderDistFG%': dd,
        'ShotClockFG%': sc
    }