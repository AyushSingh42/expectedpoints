import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

def get_games_played_dict(season="2024-25", season_type="Regular Season"):
    """
    Returns a dictionary mapping player names to number of games played.
    """
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type
    )
    df = stats.get_data_frames()[0]
    df = df[['PLAYER_NAME', 'GP']]
    return dict(zip(df['PLAYER_NAME'], df['GP']))

def add_games_played_to_csv(csv_path="xp.csv", output_path="xp_with_gp.csv"):
    # Load original CSV
    xp_df = pd.read_csv(csv_path)
    
    # Check if PLAYER_NAME column exists
    if 'PLAYER_NAME' not in xp_df.columns:
        raise ValueError("Input CSV must have a 'PLAYER_NAME' column.")
    
    # Get games played info
    gp_dict = get_games_played_dict()
    
    # Map games played to each player in CSV
    def get_gp(player_name):
        if player_name not in gp_dict:
            raise ValueError(f"Player '{player_name}' not found in NBA stats for 2024-25 season.")
        return gp_dict[player_name]

    xp_df['GamesPlayed'] = xp_df['PLAYER_NAME'].apply(get_gp)
    
    # Save to new CSV
    xp_df.to_csv(output_path, index=False)
    print(f"Updated CSV with GamesPlayed saved to {output_path}")

if __name__ == "__main__":
    add_games_played_to_csv()