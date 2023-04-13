import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_data():
    df_res = pd.read_csv('results_2223.csv')

    team_encoder = preprocessing.LabelEncoder()
    team_encoder.fit(df_res['HOME_TEAM'])
    df_res['AWAY_ID'] = team_encoder.transform(df_res['HOME_TEAM'])
    df_res['HOME_ID'] = team_encoder.transform(df_res['AWAY_TEAM'])
    
    # two matrices, a binary one for W/L and another one for point diff
    wl_mat = (df_res['HOME_WIN'] == 'W').to_numpy()
    point_diff_mat = ((df_res['HOME_PTS'].replace('TBP', 0).astype(int) - df_res['AWAY_PTS'].replace('TBP', 0).astype(int))).to_numpy()

    ot_mask = (df_res['MIN'] != '200').to_numpy()

    # mask of games already played
    played_mat = (df_res['HOME_WIN'] != 'TBP').to_numpy()
    tbp_mat = (df_res['HOME_WIN'] == 'TBP').to_numpy()

    # matches
    matches = np.flip(df_res[['HOME_ID', 'AWAY_ID']].to_numpy(), axis=1)
    rounds = df_res['ROUND'].to_numpy()

    return (wl_mat, point_diff_mat, ot_mask), (matches, rounds), (played_mat, tbp_mat), team_encoder
