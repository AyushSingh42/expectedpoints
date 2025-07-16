from ftfy import fix_text
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
import unicodedata

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
    # Filter out null player names
    df = df.dropna(subset=['PLAYER_NAME'])
    return dict(zip(df['PLAYER_NAME'], df['GP']))

def get_team_dict(season="2024-25", season_type="Regular Season"):
    """
    Returns a dictionary mapping player names to their team abbreviations.
    """
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type
    )
    df = stats.get_data_frames()[0]
    df = df[['PLAYER_NAME', 'TEAM_ABBREVIATION']]
    # Filter out null player names
    df = df.dropna(subset=['PLAYER_NAME'])
    return dict(zip(df['PLAYER_NAME'], df['TEAM_ABBREVIATION']))

def normalize_name(name):
    """Normalize name for matching - remove accents, convert to lowercase, handle punctuation"""
    # Handle None/null values
    if name is None or pd.isna(name):
        return ""
    
    # Convert to string if not already
    name = str(name).strip()
    
    # Handle empty string
    if not name:
        return ""
    
    try:
        # Remove accents
        normalized = unicodedata.normalize('NFD', name)
        ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        # Remove dots and normalize spacing
        ascii_name = ascii_name.replace('.', '').replace('-', ' ').strip()
        # Normalize multiple spaces to single space
        ascii_name = ' '.join(ascii_name.split())
        return ascii_name.lower()
    except Exception as e:
        print(f"Warning: Could not normalize name '{name}': {e}")
        return str(name).lower()

def add_games_played_to_csv(csv_path="xp.csv", output_path="xp.csv"):
    # Load original CSV with proper encoding
    xp_df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Check if PLAYER_NAME column exists
    if 'PLAYER_NAME' not in xp_df.columns:
        raise ValueError("Input CSV must have a 'PLAYER_NAME' column.")
    
    # Fix corrupted text and handle null values
    xp_df['PLAYER_NAME'] = xp_df['PLAYER_NAME'].apply(lambda x: fix_text(str(x)) if x is not None and pd.notna(x) else str(x))
    
    # Get games played info
    gp_dict = get_games_played_dict()
    
    # Create normalized lookup dictionary
    normalized_gp_dict = {}
    for player_name, gp in gp_dict.items():
        if player_name is not None and pd.notna(player_name):
            normalized_key = normalize_name(player_name)
            if normalized_key:  # Only add if normalization was successful
                normalized_gp_dict[normalized_key] = gp
    
    # Track players not found for reporting
    players_not_found = []
    
    def get_gp(player_name):
        # Handle null/empty player names
        if player_name is None or pd.isna(player_name) or str(player_name).strip() == "":
            return None
            
        player_name = str(player_name).strip()
        
        # First try exact match
        if player_name in gp_dict:
            return gp_dict[player_name]
        
        # Try normalized match
        normalized_name = normalize_name(player_name)
        if normalized_name and normalized_name in normalized_gp_dict:
            return normalized_gp_dict[normalized_name]
        
        # Try partial matching (for cases like "D.J. Carton" vs "DJ Carton")
        for stats_name, gp in gp_dict.items():
            if stats_name and normalize_name(stats_name) == normalized_name:
                return gp
        
        # Try fuzzy matching based on last name + first letter
        try:
            parts = player_name.split()
            if len(parts) >= 2:
                last_name = parts[-1].lower()
                first_initial = parts[0][0].lower()
                
                for stats_name, gp in gp_dict.items():
                    if stats_name:
                        stats_parts = stats_name.split()
                        if len(stats_parts) >= 2:
                            stats_last = stats_parts[-1].lower()
                            stats_first_initial = stats_parts[0][0].lower()
                            
                            if (last_name == stats_last and 
                                first_initial == stats_first_initial):
                                print(f"Fuzzy match: '{player_name}' -> '{stats_name}'")
                                return gp
        except (IndexError, AttributeError):
            pass
        
        # If still not found, add to not found list and return None
        players_not_found.append(player_name)
        return None
    
    # Apply the function
    xp_df['GamesPlayed'] = xp_df['PLAYER_NAME'].apply(get_gp)
    
    # Report players not found
    if players_not_found:
        print(f"\nWarning: {len(players_not_found)} players not found in NBA stats:")
        for player in sorted(set(players_not_found)):
            print(f"  - {player}")
            
            # Show potential matches
            print("    Potential matches:")
            player_parts = str(player).lower().split()
            for stats_name in sorted(gp_dict.keys()):
                if stats_name:
                    stats_parts = stats_name.lower().split()
                    # Check if any part of the name matches
                    if any(part in stats_name.lower() for part in player_parts):
                        print(f"      - {stats_name}")
            print()
    
    # Option 1: Fill missing values with 0 (conservative approach)
    xp_df['GamesPlayed'] = xp_df['GamesPlayed'].fillna(0)
    
    # Option 2: Alternative - Drop players not found
    # xp_df = xp_df.dropna(subset=['GamesPlayed'])
    
    # Save to new CSV with proper encoding
    xp_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Updated CSV with GamesPlayed saved to {output_path}")
    
    if players_not_found:
        print(f"Note: {len(set(players_not_found))} players had GamesPlayed set to 0 (not found in stats)")

def add_team_to_csv(csv_path="xp.csv", output_path="xp.csv"):
    # Load original CSV with proper encoding
    xp_df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Check if PLAYER_NAME column exists
    if 'PLAYER_NAME' not in xp_df.columns:
        raise ValueError("Input CSV must have a 'PLAYER_NAME' column.")
    
    # Fix corrupted text and handle null values
    xp_df['PLAYER_NAME'] = xp_df['PLAYER_NAME'].apply(lambda x: fix_text(str(x)) if x is not None and pd.notna(x) else str(x))
    
    # Get team info
    team_dict = get_team_dict()
    
    # Create normalized lookup dictionary
    normalized_team_dict = {}
    for player_name, team in team_dict.items():
        if player_name is not None and pd.notna(player_name):
            normalized_key = normalize_name(player_name)
            if normalized_key:  # Only add if normalization was successful
                normalized_team_dict[normalized_key] = team
    
    # Track players not found for reporting
    players_not_found = []
    
    def get_team(player_name):
        # Handle null/empty player names
        if player_name is None or pd.isna(player_name) or str(player_name).strip() == "":
            return None
            
        player_name = str(player_name).strip()
        
        # First try exact match
        if player_name in team_dict:
            return team_dict[player_name]
        
        # Try normalized match
        normalized_name = normalize_name(player_name)
        if normalized_name and normalized_name in normalized_team_dict:
            return normalized_team_dict[normalized_name]
        
        # Try partial matching (for cases like "D.J. Carton" vs "DJ Carton")
        for stats_name, team in team_dict.items():
            if stats_name and normalize_name(stats_name) == normalized_name:
                return team
        
        # Try fuzzy matching based on last name + first letter
        try:
            parts = player_name.split()
            if len(parts) >= 2:
                last_name = parts[-1].lower()
                first_initial = parts[0][0].lower()
                
                for stats_name, team in team_dict.items():
                    if stats_name:
                        stats_parts = stats_name.split()
                        if len(stats_parts) >= 2:
                            stats_last = stats_parts[-1].lower()
                            stats_first_initial = stats_parts[0][0].lower()
                            
                            if (last_name == stats_last and 
                                first_initial == stats_first_initial):
                                print(f"Fuzzy match: '{player_name}' -> '{stats_name}'")
                                return team
        except (IndexError, AttributeError):
            pass
        
        # If still not found, add to not found list and return None
        players_not_found.append(player_name)
        return None
    
    # Apply the function
    xp_df['Team'] = xp_df['PLAYER_NAME'].apply(get_team)
    
    # Report players not found
    if players_not_found:
        print(f"\nWarning: {len(players_not_found)} players not found in NBA stats:")
        for player in sorted(set(players_not_found)):
            print(f"  - {player}")
            
            # Show potential matches
            print("    Potential matches:")
            player_parts = str(player).lower().split()
            for stats_name in sorted(team_dict.keys()):
                if stats_name:
                    stats_parts = stats_name.lower().split()
                    # Check if any part of the name matches
                    if any(part in stats_name.lower() for part in player_parts):
                        print(f"      - {stats_name}")
            print()
    
    # Option 1: Fill missing values with "Unknown" (conservative approach)
    xp_df['Team'] = xp_df['Team'].fillna("Unknown")
    
    # Option 2: Alternative - Drop players not found
    # xp_df = xp_df.dropna(subset=['Team'])
    
    # Save to new CSV with proper encoding
    xp_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Updated CSV with Team saved to {output_path}")
    
    if players_not_found:
        print(f"Note: {len(set(players_not_found))} players had Team set to 'Unknown' (not found in stats)")