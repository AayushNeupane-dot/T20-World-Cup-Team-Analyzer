import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set the aesthetic style of the plots
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Create sample T20 World Cup match data
# This simulates data from recent T20 World Cups (2021, 2022, 2024)

# Match results data
matches_data = {
    'match_id': list(range(1, 51)),
    'tournament': np.random.choice(['T20 WC 2021', 'T20 WC 2022', 'T20 WC 2024'], 50),
    'date': pd.date_range(start='2021-10-17', periods=50, freq='D'),
    'venue': np.random.choice(['Dubai', 'Sharjah', 'Abu Dhabi', 'Melbourne', 'Sydney', 'Adelaide', 'New York', 'Barbados', 'Trinidad'], 50),
    'team1': np.random.choice(['India', 'Australia', 'England', 'Pakistan', 'New Zealand', 'South Africa', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan'], 50),
    'team2': np.random.choice(['India', 'Australia', 'England', 'Pakistan', 'New Zealand', 'South Africa', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan'], 50),
    'winner': np.random.choice(['team1', 'team2'], 50),
    'margin': np.random.randint(1, 50, 50),
    'margin_type': np.random.choice(['runs', 'wickets'], 50),
    'first_innings_score': np.random.randint(120, 221, 50),
    'second_innings_score': [],
    'toss_winner': np.random.choice(['team1', 'team2'], 50),
    'toss_decision': np.random.choice(['bat', 'field'], 50),
    'player_of_match': []
}

# Fix team2 to ensure different from team1
for i in range(len(matches_data['team1'])):
    while matches_data['team2'][i] == matches_data['team1'][i]:
        matches_data['team2'][i] = np.random.choice(['India', 'Australia', 'England', 'Pakistan', 'New Zealand', 'South Africa', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan'])

# Generate second innings scores based on first innings and match result
for i in range(len(matches_data['match_id'])):
    if matches_data['margin_type'][i] == 'runs':
        matches_data['second_innings_score'].append(matches_data['first_innings_score'][i] - matches_data['margin'][i])
    else:  # wickets
        matches_data['second_innings_score'].append(matches_data['first_innings_score'][i] + np.random.randint(1, 10))
        
# Generate player of match
players = {
    'India': ['Kohli', 'Rohit', 'Bumrah', 'Pandya'],
    'Australia': ['Warner', 'Maxwell', 'Cummins', 'Starc'],
    'England': ['Buttler', 'Stokes', 'Archer', 'Moeen'],
    'Pakistan': ['Babar', 'Rizwan', 'Shaheen', 'Shadab'],
    'New Zealand': ['Williamson', 'Conway', 'Boult', 'Santner'],
    'South Africa': ['de Kock', 'Miller', 'Rabada', 'Nortje'],
    'West Indies': ['Pooran', 'Hetmyer', 'Russell', 'Pollard'],
    'Sri Lanka': ['Hasaranga', 'Rajapaksa', 'Chameera', 'Nissanka'],
    'Bangladesh': ['Shakib', 'Mahmudullah', 'Mustafizur', 'Liton'],
    'Afghanistan': ['Rashid', 'Nabi', 'Gurbaz', 'Mujeeb']
}

for i in range(len(matches_data['match_id'])):
    winner_team = matches_data['team1'][i] if matches_data['winner'][i] == 'team1' else matches_data['team2'][i]
    matches_data['player_of_match'].append(np.random.choice(players[winner_team]))

# Create actual winner column
matches_data['actual_winner'] = [
    matches_data['team1'][i] if matches_data['winner'][i] == 'team1' else matches_data['team2'][i]
    for i in range(len(matches_data['match_id']))
]

# Create matches DataFrame
matches_df = pd.DataFrame(matches_data)

# Create batting performance data
batting_data = []
for i in range(len(matches_df)):
    match_id = matches_df['match_id'][i]
    team1 = matches_df['team1'][i]
    team2 = matches_df['team2'][i]
    tournament = matches_df['tournament'][i]
    
    # First innings
    for player in players[team1]:
        if np.random.random() < 0.8:  # 80% chance player batted
            runs = np.random.randint(0, 120)
            balls = np.random.randint(max(1, runs // 2), max(2, runs * 2))
            fours = np.random.randint(0, runs // 4 + 1)
            sixes = np.random.randint(0, runs // 6 + 1)
            
            batting_data.append({
                'match_id': match_id,
                'tournament': tournament,
                'player': player,
                'team': team1,
                'opposition': team2,
                'innings': 1,
                'runs': runs,
                'balls': balls,
                'fours': fours,
                'sixes': sixes,
                'strike_rate': round((runs / balls) * 100, 2) if balls > 0 else 0
            })
    
    # Second innings
    for player in players[team2]:
        if np.random.random() < 0.8:  # 80% chance player batted
            runs = np.random.randint(0, 120)
            balls = np.random.randint(max(1, runs // 2), max(2, runs * 2))
            fours = np.random.randint(0, runs // 4 + 1)
            sixes = np.random.randint(0, runs // 6 + 1)
            
            batting_data.append({
                'match_id': match_id,
                'tournament': tournament,
                'player': player,
                'team': team2,
                'opposition': team1,
                'innings': 2,
                'runs': runs,
                'balls': balls,
                'fours': fours,
                'sixes': sixes,
                'strike_rate': round((runs / balls) * 100, 2) if balls > 0 else 0
            })

batting_df = pd.DataFrame(batting_data)

# Create bowling performance data
bowling_data = []
for i in range(len(matches_df)):
    match_id = matches_df['match_id'][i]
    team1 = matches_df['team1'][i]
    team2 = matches_df['team2'][i]
    tournament = matches_df['tournament'][i]
    
    # First innings (team2 bowls)
    for player in players[team2]:
        if np.random.random() < 0.7:  # 70% chance player bowled
            overs = np.random.randint(1, 5)
            balls = overs * 6 + np.random.randint(0, 6)
            runs = np.random.randint(balls // 2, balls * 2)
            wickets = np.random.randint(0, 4)
            
            bowling_data.append({
                'match_id': match_id,
                'tournament': tournament,
                'player': player,
                'team': team2,
                'opposition': team1,
                'innings': 1,
                'overs': overs + (balls % 6) / 10,
                'balls': balls,
                'runs': runs,
                'wickets': wickets,
                'economy': round(runs / (balls / 6), 2) if balls > 0 else 0
            })
    
    # Second innings (team1 bowls)
    for player in players[team1]:
        if np.random.random() < 0.7:  # 70% chance player bowled
            overs = np.random.randint(1, 5)
            balls = overs * 6 + np.random.randint(0, 6)
            runs = np.random.randint(balls // 2, balls * 2)
            wickets = np.random.randint(0, 4)
            
            bowling_data.append({
                'match_id': match_id,
                'tournament': tournament,
                'player': player,
                'team': team1,
                'opposition': team2,
                'innings': 2,
                'overs': overs + (balls % 6) / 10,
                'balls': balls,
                'runs': runs,
                'wickets': wickets,
                'economy': round(runs / (balls / 6), 2) if balls > 0 else 0
            })

bowling_df = pd.DataFrame(bowling_data)

# Data Analysis Functions

def analyze_team_performance(matches_df):
    """Analyze overall team performance in T20 World Cups"""
    # Create a dictionary to store team stats
    team_stats = {}
    
    all_teams = list(set(matches_df['team1'].tolist() + matches_df['team2'].tolist()))
    
    for team in all_teams:
        # Matches where team was involved
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        
        # Matches won
        team_wins = team_matches[team_matches['actual_winner'] == team]
        
        # Calculate win percentage
        win_percentage = round((len(team_wins) / len(team_matches)) * 100, 2) if len(team_matches) > 0 else 0
        
        # Average score when batting first
        batting_first = matches_df[(matches_df['team1'] == team) & (matches_df['toss_decision'] == 'bat') | 
                                   (matches_df['team2'] == team) & (matches_df['toss_decision'] == 'field')]
        
        avg_batting_first = round(batting_first['first_innings_score'].mean(), 2) if len(batting_first) > 0 else 0
        
        # Toss win percentage
        toss_wins = matches_df[((matches_df['team1'] == team) & (matches_df['toss_winner'] == 'team1')) | 
                               ((matches_df['team2'] == team) & (matches_df['toss_winner'] == 'team2'))]
        
        toss_win_pct = round((len(toss_wins) / len(team_matches)) * 100, 2) if len(team_matches) > 0 else 0
        
        # Store stats in dictionary
        team_stats[team] = {
            'matches': len(team_matches),
            'wins': len(team_wins),
            'win_percentage': win_percentage,
            'avg_score_batting_first': avg_batting_first,
            'toss_win_percentage': toss_win_pct
        }
    
    # Convert to DataFrame
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    team_stats_df = team_stats_df.sort_values('win_percentage', ascending=False)
    
    return team_stats_df

def analyze_batting_performance(batting_df):
    """Analyze top batting performances in T20 World Cups"""
    # Group by player and calculate stats
    player_batting_stats = batting_df.groupby(['player', 'team']).agg({
        'runs': ['sum', 'mean', 'max'],
        'balls': 'sum',
        'fours': 'sum',
        'sixes': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    
    # Rename columns
    player_batting_stats.columns = ['player', 'team', 'total_runs', 'avg_runs', 'highest_score', 
                                    'balls_faced', 'fours', 'sixes', 'innings']
    
    # Calculate strike rate
    player_batting_stats['strike_rate'] = round((player_batting_stats['total_runs'] / player_batting_stats['balls_faced']) * 100, 2)
    
    # Sort by total runs
    player_batting_stats = player_batting_stats.sort_values('total_runs', ascending=False)
    
    return player_batting_stats

def analyze_bowling_performance(bowling_df):
    """Analyze top bowling performances in T20 World Cups"""
    # Group by player and calculate stats
    player_bowling_stats = bowling_df.groupby(['player', 'team']).agg({
        'wickets': ['sum', 'mean', 'max'],
        'runs': 'sum',
        'balls': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    
    # Rename columns
    player_bowling_stats.columns = ['player', 'team', 'total_wickets', 'avg_wickets', 'best_bowling', 
                                    'runs_conceded', 'balls_bowled', 'innings']
    
    # Calculate economy rate
    player_bowling_stats['economy_rate'] = round((player_bowling_stats['runs_conceded'] / (player_bowling_stats['balls_bowled'] / 6)), 2)
    
    # Calculate bowling average
    player_bowling_stats['bowling_avg'] = round((player_bowling_stats['runs_conceded'] / player_bowling_stats['total_wickets']), 2)
    
    # Sort by total wickets
    player_bowling_stats = player_bowling_stats.sort_values('total_wickets', ascending=False)
    
    return player_bowling_stats

def analyze_tournament_trends(matches_df):
    """Analyze tournament trends across different editions"""
    # Tournament stats
    tournament_stats = matches_df.groupby('tournament').agg({
        'first_innings_score': ['mean', 'min', 'max'],
        'second_innings_score': ['mean', 'min', 'max'],
        'match_id': 'count'
    }).reset_index()
    
    # Rename columns
    tournament_stats.columns = ['tournament', 'avg_first_innings', 'min_first_innings', 'max_first_innings',
                               'avg_second_innings', 'min_second_innings', 'max_second_innings', 'matches']
    
    # Calculate win percentages by batting first vs second
    tournament_wins = []
    
    for tournament in matches_df['tournament'].unique():
        tournament_matches = matches_df[matches_df['tournament'] == tournament]
        
        # Count wins batting first
        batting_first_wins = tournament_matches[
            ((tournament_matches['team1'] == tournament_matches['actual_winner']) & (tournament_matches['toss_decision'] == 'bat')) |
            ((tournament_matches['team2'] == tournament_matches['actual_winner']) & (tournament_matches['toss_decision'] == 'field'))
        ]
        
        batting_first_win_pct = round((len(batting_first_wins) / len(tournament_matches)) * 100, 2)
        
        tournament_wins.append({
            'tournament': tournament,
            'batting_first_win_pct': batting_first_win_pct,
            'batting_second_win_pct': 100 - batting_first_win_pct
        })
    
    tournament_wins_df = pd.DataFrame(tournament_wins)
    
    return tournament_stats, tournament_wins_df

def analyze_match_phases(batting_df, bowling_df):
    """Analyze performance in different match phases (powerplay, middle, death)"""
    # This is a simplified simulation since we don't have over-by-over data
    # We'll create synthetic data for demonstration
    
    powerplay_stats = []
    death_overs_stats = []
    
    for team in batting_df['team'].unique():
        # Powerplay runs (random simulation)
        powerplay_runs = np.random.randint(35, 65, 20)
        powerplay_avg = np.mean(powerplay_runs)
        
        # Death overs runs (random simulation)
        death_runs = np.random.randint(40, 80, 20)
        death_avg = np.mean(death_runs)
        
        powerplay_stats.append({
            'team': team,
            'avg_powerplay_runs': round(powerplay_avg, 2),
            'powerplay_run_rate': round(powerplay_avg / 6, 2)
        })
        
        death_overs_stats.append({
            'team': team,
            'avg_death_runs': round(death_avg, 2),
            'death_run_rate': round(death_avg / 4, 2)
        })
    
    powerplay_df = pd.DataFrame(powerplay_stats)
    death_overs_df = pd.DataFrame(death_overs_stats)
    
    return powerplay_df, death_overs_df

def plot_team_performance(team_stats_df):
    """Plot team performance comparison"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x=team_stats_df.index, y='win_percentage', data=team_stats_df)
    plt.title('Team Win Percentage in T20 World Cups', fontsize=16)
    plt.xlabel('Teams', fontsize=14)
    plt.ylabel('Win Percentage (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot average scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x=team_stats_df.index, y='avg_score_batting_first', data=team_stats_df)
    plt.title('Average Score When Batting First', fontsize=16)
    plt.xlabel('Teams', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_batting_stats(player_batting_stats):
    """Plot top batting performances"""
    # Top 10 run scorers
    top_batsmen = player_batting_stats.head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='player', y='total_runs', hue='team', data=top_batsmen)
    plt.title('Top 10 Run Scorers in T20 World Cups', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Total Runs', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Team')
    plt.tight_layout()
    plt.show()
    
    # Strike rates of top batsmen
    plt.figure(figsize=(12, 8))
    sns.barplot(x='player', y='strike_rate', hue='team', data=top_batsmen)
    plt.title('Strike Rates of Top 10 Run Scorers', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Strike Rate', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Team')
    plt.tight_layout()
    plt.show()

def plot_bowling_stats(player_bowling_stats):
    """Plot top bowling performances"""
    # Top 10 wicket takers
    top_bowlers = player_bowling_stats.head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='player', y='total_wickets', hue='team', data=top_bowlers)
    plt.title('Top 10 Wicket Takers in T20 World Cups', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Total Wickets', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Team')
    plt.tight_layout()
    plt.show()
    
    # Economy rates of top bowlers
    plt.figure(figsize=(12, 8))
    sns.barplot(x='player', y='economy_rate', hue='team', data=top_bowlers)
    plt.title('Economy Rates of Top 10 Wicket Takers', fontsize=16)
    plt.xlabel('Players', fontsize=14)
    plt.ylabel('Economy Rate', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Team')
    plt.tight_layout()
    plt.show()

def plot_tournament_trends(tournament_stats, tournament_wins_df):
    """Plot tournament trends across editions"""
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(tournament_stats['tournament']))
    width = 0.35
    
    plt.bar(x - width/2, tournament_stats['avg_first_innings'], width, label='First Innings')
    plt.bar(x + width/2, tournament_stats['avg_second_innings'], width, label='Second Innings')
    
    plt.xlabel('Tournament', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores by Tournament', fontsize=16)
    plt.xticks(x, tournament_stats['tournament'])
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot batting first vs second win percentage
    plt.figure(figsize=(10, 6))
    
    # Create stacked bars
    plt.bar(tournament_wins_df['tournament'], tournament_wins_df['batting_first_win_pct'], 
            label='Batting First Wins')
    plt.bar(tournament_wins_df['tournament'], tournament_wins_df['batting_second_win_pct'], 
            bottom=tournament_wins_df['batting_first_win_pct'], label='Batting Second Wins')
    
    plt.xlabel('Tournament', fontsize=14)
    plt.ylabel('Win Percentage', fontsize=14)
    plt.title('Batting First vs Second Win Percentage by Tournament', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_match_phases(powerplay_df, death_overs_df):
    """Plot match phase analysis"""
    # Combine dataframes
    phases_df = pd.merge(powerplay_df, death_overs_df, on='team')
    
    # Sort by powerplay run rate
    phases_df = phases_df.sort_values('powerplay_run_rate', ascending=False)
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(phases_df['team']))
    width = 0.35
    
    plt.bar(x - width/2, phases_df['powerplay_run_rate'], width, label='Powerplay (Overs 1-6)')
    plt.bar(x + width/2, phases_df['death_run_rate'], width, label='Death Overs (Overs 17-20)')
    
    plt.xlabel('Teams', fontsize=14)
    plt.ylabel('Run Rate', fontsize=14)
    plt.title('Team Performance in Different Match Phases', fontsize=16)
    plt.xticks(x, phases_df['team'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the analysis functions
team_stats_df = analyze_team_performance(matches_df)
player_batting_stats = analyze_batting_performance(batting_df)
player_bowling_stats = analyze_bowling_performance(bowling_df)
tournament_stats, tournament_wins_df = analyze_tournament_trends(matches_df)
powerplay_df, death_overs_df = analyze_match_phases(batting_df, bowling_df)

# Print overall team performance
print("T20 World Cup Team Performance")
print("-----------------------------")
print(team_stats_df)
print("\n")

# Print top 5 batsmen by runs
print("Top 5 Batsmen by Total Runs")
print("-------------------------")
print(player_batting_stats[['player', 'team', 'total_runs', 'avg_runs', 'strike_rate']].head(5))
print("\n")

# Print top 5 bowlers by wickets
print("Top 5 Bowlers by Total Wickets")
print("----------------------------")
print(player_bowling_stats[['player', 'team', 'total_wickets', 'economy_rate', 'bowling_avg']].head(5))
print("\n")

# Print tournament trends
print("Tournament Trends")
print("----------------")
print(tournament_stats)
print("\n")
print("Batting First vs Second Win Percentage")
print(tournament_wins_df)
print("\n")

# Print match phase analysis
print("Team Performance in Match Phases")
print("------------------------------")
print("Powerplay Performance (Overs 1-6)")
print(powerplay_df.sort_values('powerplay_run_rate', ascending=False))
print("\n")
print("Death Overs Performance (Overs 17-20)")
print(death_overs_df.sort_values('death_run_rate', ascending=False))

# Generate visualizations
plot_team_performance(team_stats_df)
plot_batting_stats(player_batting_stats)
plot_bowling_stats(player_bowling_stats)
plot_tournament_trends(tournament_stats, tournament_wins_df)
plot_match_phases(powerplay_df, death_overs_df)

# Advanced Analysis: Predictive model for match outcome

def build_match_prediction_model(matches_df):
    """Build a simple predictive model for match outcomes"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Create features
    matches_df['team1_is_winner'] = (matches_df['winner'] == 'team1').astype(int)
    
    # One-hot encode teams
    team1_dummies = pd.get_dummies(matches_df['team1'], prefix='team1')
    team2_dummies = pd.get_dummies(matches_df['team2'], prefix='team2')
    
    # Create features DataFrame
    features = pd.concat([
        team1_dummies, 
        team2_dummies,
        matches_df[['toss_winner']].replace({'team1': 0, 'team2': 1}),
        pd.get_dummies(matches_df['toss_decision'], prefix='toss')
    ], axis=1)
    
    # Target variable
    target = matches_df['team1_is_winner']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Match Prediction Model Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Features for Predicting Match Outcome', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return model, feature_importance

# Run the predictive model
try:
    model, feature_importance = build_match_prediction_model(matches_df)
    print("\nTop 5 Important Features for Match Prediction")
    print(feature_importance.head(5))
except ImportError:
    print("Scikit-learn not available, skipping predictive model")

# Final Insights
print("\n\nKEY INSIGHTS FROM T20 WORLD CUP ANALYSIS")
print("========================================")
print(f"1. {team_stats_df.index[0]} has the highest win percentage at {team_stats_df['win_percentage'].iloc[0]:.2f}%")
print(f"2. {player_batting_stats['player'].iloc[0]} ({player_batting_stats['team'].iloc[0]}) is the top run scorer with {player_batting_stats['total_runs'].iloc[0]} runs")
print(f"3. {player_bowling_stats['player'].iloc[0]} ({player_bowling_stats['team'].iloc[0]}) is the top wicket taker with {player_bowling_stats['total_wickets'].iloc[0]} wickets")
