import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle

# look_back = 3 # games
def get_data(data_type, start_year, end_year=2020, week=None):
    website = 'https://api.collegefootballdata.com/' + data_type
    frames = []
    for year in range(start_year, end_year+1):
        url = website + '?year=%d' % year
        if week:
          # start = max(week - look_back, 1)
          # url += '&startWeek=%d' % start
          url += '&endWeek=%d' % week
        frames.append(pd.read_json(url))
    return pd.concat(frames)

def get_weekly_stats():
  frames = []
  for week in range(1, 16):
    week_frame = get_data('stats/season', 2004, week=week)
    week_frame['week'] = week+1
    frames.append(week_frame)
  return pd.concat(frames)


def get_games_df(data_type='train'):
    '''
    Parameters
    ----------
    data_type : str
        - 'train' returns all years except 2019
        - 'test' returns 2019

    Returns
    -------
    games : pandas df
        data for all games
    '''

    try:
        with open('cfb_data.pkl', 'rb') as f:
            original_stats, original_games, original_records, original_teams = pickle.load(f)
    except:
        raise Exception('Call save_data to create pickle files of data')

    # get the games and calculate spread (result)
    games = original_games
    games = games.assign(result = (games.home_points-games.away_points))
    games = games.loc[:, games.columns.intersection(['season', 'week', 'home_team', 'away_team', 'result'])]

    # get winning percentages for the season
    # TODO: make this win percentage up to that week's game
    records = original_records
    records = records.assign(total_games = [d.get('games') for d in records.total])
    records = records.assign(total_wins = [d.get('wins') for d in records.total])
    records = records.loc[:, records.columns.intersection(['year', 'team', 'total_games', 'total_wins'])]

    # add winning percentage difference
    games = pd.merge(games, records.rename(columns={'total_games': 'home_total_games', 'total_wins': 'home_wins'}), how='left', left_on=['season', 'home_team'], right_on=['year', 'team']).drop(['year', 'team'], axis=1)
    games = pd.merge(games, records.rename(columns={'total_games': 'away_total_games', 'total_wins': 'away_wins'}), how='left', left_on=['season', 'away_team'], right_on=['year', 'team']).drop(['year', 'team'], axis=1)
    games = games.assign(win_difference = ((games.home_wins / games.home_total_games)-(games.away_wins / games.away_total_games)))
    games = games.drop(['home_wins', 'away_wins'], axis=1)
    games = games.dropna()
    games = games.drop(columns=['home_total_games', 'away_total_games'])

    stats = original_stats
    stats = stats.drop(columns=['conference'])

    # get the list of stats
    w = stats.loc[(stats.team == 'Washington') & (stats.season == 2004)]
    stat_cols = w.statName.to_list()
    stat_cols = list(set(stat_cols)) # remove duplicates
    stat_cols.sort()
    stat_cols.remove('games')
    # print(stat_cols)

    # add each stat of both teams
    s = stats.loc[(stats.statName == 'games')]
    s = s.drop(columns=['statName'])
    games = pd.merge(games, s.rename(columns={'statValue': 'home_games'}), how='left', left_on=['season', 'home_team', 'week'], right_on=['season', 'team', 'week']).drop(['team'], axis=1)
    games = pd.merge(games, s.rename(columns={'statValue': 'away_games'}), how='left', left_on=['season', 'away_team', 'week'], right_on=['season', 'team', 'week']).drop(['team'], axis=1)

    for stat in stat_cols:
        s = stats.loc[(stats.statName == stat)]
        s = s.drop(columns=['statName'])
        games = pd.merge(games, s.rename(columns={'statValue': 'home_'+stat}), how='left', left_on=['season', 'home_team', 'week'], right_on=['season', 'team', 'week']).drop(['team'], axis=1)
        games = pd.merge(games, s.rename(columns={'statValue': 'away_'+stat}), how='left', left_on=['season', 'away_team', 'week'], right_on=['season', 'team', 'week']).drop(['team'], axis=1)
        games = games.fillna(0)
        games[stat+'_spread'] = (games['home_'+stat]/games.home_games) - (games['away_'+stat]/games.away_games)
        games = games.drop(columns=['home_'+stat, 'away_'+stat])

    games = games.fillna(0)

    # create the table for total points a team scored for an against before a given week
    points_per_game = original_teams.loc[:, original_teams.columns.intersection(['school'])]
    points_per_game['year'] = 2020
    one_season = points_per_game.copy()
    for year in range(2004, 2020):
        temp = one_season.copy()
        temp['year'] = year
        points_per_game = pd.concat([points_per_game, temp], ignore_index=True)

    points_per_game[('points_for_week1')] = 0.0
    points_per_game[('points_against_week1')] = 0.0
    scores = original_games.loc[:, original_games.columns.intersection(['season', 'week', 'home_team', 'away_team', 'home_points', 'away_points'])]
    scores['week'] += 1
    for week in range(2, 16):
        this_scores = scores.loc[scores.week == week]
        this_scores = this_scores.rename(columns={'home_points': ('points_for_week%d' % week), 'away_points': ('points_against_week%d' % week)})
        points_per_game = pd.merge(points_per_game, this_scores, how='left', left_on=['school', 'year'], right_on=['home_team', 'season']).drop(['home_team', 'away_team', 'season', 'week'], axis=1)
        this_scores = scores.loc[scores.week == week]
        this_scores = this_scores.rename(columns={'away_points': ('points_for_week%d' % week), 'home_points': ('points_against_week%d' % week)})
        points_per_game = pd.merge(points_per_game, this_scores, how='left', left_on=['school', 'year'], right_on=['away_team', 'season']).drop(['home_team', 'away_team', 'season', 'week'], axis=1)
        points_per_game = points_per_game.fillna(0)
        points_per_game[('points_for_week%d' % week)] = points_per_game[('points_for_week%d_x' % week)] + points_per_game[('points_for_week%d_y' % week)]
        points_per_game[('points_against_week%d' % week)] = points_per_game[('points_against_week%d_x' % week)] + points_per_game[('points_against_week%d_y' % week)]
        points_per_game = points_per_game.drop(columns=[('points_for_week%d_x' % week), ('points_for_week%d_y' % week), ('points_against_week%d_x' % week), ('points_against_week%d_y' % week)])

    for week in range(2, 16):
        points_per_game[('points_for_week%d' % week)] += points_per_game[('points_for_week%d' % (week-1))]
        points_per_game[('points_against_week%d' % week)] += points_per_game[('points_against_week%d' % (week-1))]

    for week in range(1,16):
        points = points_per_game.loc[:, points_per_game.columns.intersection(['year', 'school', ('points_for_week%d' % week), ('points_against_week%d' % week)])]
        points = points.rename(columns={('points_for_week%d' % week): 'home_for', ('points_against_week%d' % week): 'home_against'})
        points['week'] = week
        games = pd.merge(games, points, how='left', left_on=['home_team', 'season', 'week'], right_on=['school', 'year', 'week']).drop(['school', 'year'], axis=1)
        points = points.rename(columns={'home_for': 'away_for', 'home_against': 'away_against'})
        games = pd.merge(games, points, how='left', left_on=['away_team', 'season', 'week'], right_on=['school', 'year', 'week']).drop(['school', 'year'], axis=1)
        if (week > 1):
            games['home_for'] = games['home_for_x'].fillna(0) + games['home_for_y'].fillna(0)
            games['home_against'] = games['home_against_x'].fillna(0) + games['home_against_y'].fillna(0)
            games['away_for'] = games['away_for_x'].fillna(0) + games['away_for_y'].fillna(0)
            games['away_against'] = games['away_against_x'].fillna(0) + games['away_against_y'].fillna(0)
            games = games.drop(columns=['home_for_x', 'home_for_y', 'home_against_x', 'home_against_y', 'away_for_x', 'away_for_y', 'away_against_x', 'away_against_y'])

    games['home_for'] /= games['home_games']
    games['home_against'] /= games['home_games']
    games['away_for'] /= games['away_games']
    games['away_against'] /= games['away_games']
    games = games.fillna(0)

    games = games.drop(['win_difference'],axis=1)

    # Drop 2020
    games = games[games['season'] != 2020]

    # Add record up to game
    # Create list of teams
    teams = pd.concat((games['home_team'], games['away_team'])).unique()

    # Create Team ID
    team_id = np.arange(0,len(teams))
    team2id = dict(zip(teams, team_id))

    num_seasons = len(games['season'].unique())
    season2id = dict(zip(games['season'].unique(), np.arange(0,num_seasons)))
    # Create Wins Lookup Table
    win_data = np.zeros((num_seasons, len(teams),17))
    #print(win_data.shape) #(season, team, week)

    # loop through every game in games df
    for k in range(len(games)):
        home_team = games['home_team'][k]
        away_team = games['away_team'][k]
        result = np.sign(games['result'][k])
        week = games['week'][k]
        season = games['season'][k]
        
        home_id = team2id[home_team]
        away_id = team2id[away_team]
        seas_id = season2id[season]
        if result == 1: # Home team won
            win_data[seas_id, home_id, (week+1):] += 1
        else: # Away team won
            win_data[seas_id, away_id, (week+1):] += 1

    # populate games df
    away_wins = []
    home_wins = []
    for k in range(len(games)):
        week = games['week'][k]
        home = games['home_team'][k]
        away = games['away_team'][k]
        season = games['season'][k]
        away_wins.append(win_data[season2id[season],team2id[away],week])
        home_wins.append(win_data[season2id[season],team2id[home],week])
        
    games['away_wins'] = away_wins
    games['home_wins'] = home_wins


    if data_type == 'train':
        games = games[games['season'] != 2019]
    elif data_type == 'test':
        games = games[games['season'] == 2019]
    else:
        raise Exception('Invalid data_type string ("train", "test" is valid)')
    return games