import numpy as np
from itertools import groupby
from dataclasses import dataclass

from utils import load_data

def compute_mov_multiplier(mov, elo_diff):
    return ((mov + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff)

def sample_from_Y_given_X(X, mean, cov):
    mu_Y_given_X = mean[1] + cov[1][0] * (X - mean[0]) / cov[0][0]
    var_Y_given_X = cov[1][1] - cov[1][0] * cov[0][1] / cov[0][0]

    # Sample from the conditional distribution of Y given X
    return np.random.normal(mu_Y_given_X, np.sqrt(var_Y_given_X), size=len(X))

def get_order(teams, criteria=0, wins_mat=None, point_diff_mat=None):
    # compute the tiebreaker criteria of the different teams (either number of wins or point diff)
    if criteria == 0: team_values = wins_mat.sum(axis=1)
    elif criteria == 1: team_values = wins_mat[teams, :][:, teams].sum(axis=1)
    elif criteria == 2: team_values = point_diff_mat[teams, :][:, teams].sum(axis=1)
    elif criteria == 3: team_values = point_diff_mat[teams, :].sum(axis=1)
    else:
        raise Exception(f'Criteria {criteria} not implemented')
    
    prov_order = team_values.argsort()  # provisional order
    
    # check for subgroups of tied teams 
    curr_idx = 0
    for v, g in groupby(team_values[prov_order]):
        group_size = len(list(g))

        if group_size == 1:  # no tie
            curr_idx += group_size
            continue
        
        # tie involving multiple teams: move on recursively to next tiebreaker criteria
        subgroup_teams = teams[prov_order[curr_idx:curr_idx+group_size]]  # teams[team_values == v]
        next_criteria = 1 if group_size < len(teams) else criteria + 1
        subgroup_order = get_order(subgroup_teams, criteria=next_criteria,
                                   wins_mat=wins_mat, point_diff_mat=point_diff_mat)
        
        prov_order[curr_idx:curr_idx+group_size] = prov_order[curr_idx + subgroup_order]
        curr_idx += group_size
    
    # return the relative order of the teams
    return prov_order       


@dataclass
class EloSettings:
    K: float = 20.0  # K factor in ELO systems
    hca: float = 100.0  # home court advantage
    avg_rating: float = 1500.0
    initial_rating: float = 1500.0

class Simulator:
    def __init__(self,
                 last_played_round: int = 32,
                 num_teams: int = 18,
                 elo_settings: EloSettings = EloSettings(),
                 **kwargs):
        super().__init__(**kwargs)
        self.num_teams = num_teams
        self.last_played_round = last_played_round
        self.elo_settings = elo_settings

        # load base wins and point diff (up to the last played round)
        self.load_base_season()

        # compute base ELO ratings
        self.initialize_ratings()  # set base elo for the simulations


    def load_base_season(self):
        (home_wl_mat, point_diff_mat, ot_mask), \
        (matches, rounds), \
        (played_mat, tbp_mat), team_encoder = load_data()

        self.home_wl_mat = home_wl_mat
        self.point_diff_mat = point_diff_mat
        self.ot_mask = ot_mask
        self.matches = matches
        self.rounds = rounds
        self.played_mat = played_mat
        self.tbp_mat = tbp_mat
        self.team_encoder = team_encoder

        self.base_wins, self.base_point_diff = self.recreate_season_results()

    def recreate_season_results(self):
        wins = np.zeros((self.num_teams, self.num_teams), dtype=np.int8)
        point_diff = np.zeros((self.num_teams, self.num_teams), dtype=np.int32)

        # para mirar la acumulada de un equipo se hace con .sum(axis=1)
        for game_idx in np.where(self.played_mat[self.rounds <= self.last_played_round])[0]:
            team_h, team_a = self.matches[game_idx, :]
            home_win = self.home_wl_mat[game_idx]
            pd = self.point_diff_mat[game_idx]
            ot = self.ot_mask[game_idx]
            
            wins[team_h, team_a] += 1 if home_win else 0
            wins[team_a, team_h] += 1 if not home_win else 0
            home_mov = 0 if ot else pd
            point_diff[team_h, team_a] += home_mov
            point_diff[team_a, team_h] -= home_mov

        return wins, point_diff
    
    def initialize_ratings(self):
        initial_elo = np.ones(self.matches[:, 0].max() + 1) * self.elo_settings.initial_rating

        # store the elo_diff and mov of each game (to be used later for the simulations)
        self.elo_diff_history = np.zeros((self.last_played_round, self.num_teams//2))
        self.mov_history = self.elo_diff_history.copy()

        self.base_elo = self.simulate_rounds(initial_elo, simulate=False, start_round=1)

        if len(self.elo_diff_history.shape) > 1: 
            self.elo_diff_history = self.elo_diff_history.flatten()
            self.elo_diff_history = np.concatenate([self.elo_diff_history, -self.elo_diff_history])
        
        if len(self.mov_history.shape) > 1: 
            self.mov_history = self.mov_history.flatten()
            self.mov_history = np.concatenate([self.mov_history, -self.mov_history])

        # compute elo_diff and mov distribution parameters
        self.sim_mean = [np.mean(self.elo_diff_history), np.mean(self.mov_history)]
        self.sim_cov = np.cov(self.elo_diff_history, self.mov_history)

    def simulate_rounds(self, initial_elo, simulate=True, start_round=1,
                        partial_round=False, record_simulated_games=False):
        
        base_elo = initial_elo.copy()
        if simulate:
            sim_wins = self.base_wins.copy()
            sim_point_diff = self.base_point_diff.copy()

        record_simulated_games = record_simulated_games and simulate
        if record_simulated_games:
            simulated_games = np.zeros_like(self.tbp_mat, dtype=np.int8)

        if not simulate and start_round > 1: raise Exception(f'Not simulating but starting in a round different than 1')
        
        # simulate round
        for round_idx in range(start_round, self.rounds.max()+1):
            if not simulate and round_idx > self.last_played_round: break
            round_mask = self.rounds==round_idx

            if partial_round:
                round_mask = np.logical_and(round_mask, self.tbp_mat==1 if simulate else 0)

            # compute elo difference (negative when underdog wins)
            elo_diff = base_elo[self.matches[round_mask, 0]] + self.elo_settings.hca - base_elo[self.matches[round_mask, 1]]
            elo_diff_sign_win = (self.point_diff_mat[round_mask] < 0) * (-2) + 1

            # compute expected win prob for home team
            expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400))
            
            if round_idx > self.last_played_round and simulate:
                # simulate games
                sim_movs = np.round(sample_from_Y_given_X(elo_diff, self.sim_mean, self.sim_cov)).astype(np.int32)
                tied_mask = sim_movs == 0
                
                sim_point_diff[self.matches[round_mask][:, 0], self.matches[round_mask][:, 1]] += sim_movs
                sim_point_diff[self.matches[round_mask][:, 1], self.matches[round_mask][:, 0]] += -sim_movs

                while tied_mask.sum() > 0:
                    sim_movs[tied_mask] = np.round(sample_from_Y_given_X(elo_diff[tied_mask], 
                                                                         self.sim_mean, 
                                                                         self.sim_cov))
                    tied_mask = sim_movs == 0
                
                sim_wins[self.matches[round_mask][:, 0], self.matches[round_mask][:, 1]] += (sim_movs > 0) * 1
                sim_wins[self.matches[round_mask][:, 1], self.matches[round_mask][:, 0]] += (sim_movs < 0) * 1

                if record_simulated_games:
                    simulated_games[round_mask] = (sim_movs > 0) * 2 - 1

            elif round_idx <= self.last_played_round and not simulate:
                # game played, no simulation, but keep stats for simulation
                self.elo_diff_history[round_idx-1] = elo_diff
                self.mov_history[round_idx-1] = self.point_diff_mat[round_mask]
            
            if not simulate:  # dont update ELOs using simulated results
                # compute mov multiplier
                mov = np.abs(self.point_diff_mat[round_mask])  # margin of victory
                mov_mult = compute_mov_multiplier(mov, elo_diff*elo_diff_sign_win)

                # compute ELO points at stake
                home_result = self.home_wl_mat[round_mask]
                shift = self.elo_settings.K * mov_mult * (home_result - expected_home)

                base_elo[self.matches[round_mask, 0]] += shift
                base_elo[self.matches[round_mask, 1]] -= shift

        if simulate and not record_simulated_games:
            return sim_wins, sim_point_diff
        elif simulate and record_simulated_games:
            return sim_wins, sim_point_diff, simulated_games

        return base_elo
