from data_preperator import Processor_Efficient
from utils import *
from config import data_map_X_1, weapon_map_f_MI, tick_rate
from debug import debug_seq_data
from plot_module import plot_move_seq

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
import time

import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix



class Expected_Win_Rate:
    """
    Information needed:    
        - The health of the player & opponent (Continueous var)
        - The armor of the players (Boolean var)
        - The helmet of the players (Boolean var)
        - Pitch and Yaw values of this player and the opponent (Continuous var)
        - Diff of Pitch and Yaw values from the target Player (Continuous var)
        - Distance between 2 players (Continuous var)
        - Ticks after flashed of this player and the opponent (Continuous var)
        - The weapon of the player & opponent (Discrete var)
        
        
    TODO:
        - Categorical From 1 to 9 (Order of death)
        - 
        
    Not considered:
        
        
    TODO:
        - High Risk / Low Risk Zone
        - The life state of 10 players alive or not (Boolean var) 
        - The positions of 10 players (Continuous var)
                
        - Certain Situations and uncertain situations (delta map) & Kill map
        
        - Which one is contributing the most
            - Learning mechanism (dynamics)
    """
    wanted_props = ['team_num', 'flash_duration', 'pitch', 'yaw', 'X', 'Y', 'Z', 'armor_value', 'active_weapon_name', 'flash_max_alpha', 'health', 'is_alive', 'has_helmet', 'move_type']
    bool_to_int = ['is_alive', 'has_helmet']
    folder_path = "demo_analysis\demo\demos_extracted"
    train_data_path = "demo_analysis\intermediate_data\model_training"
    train_sequence_data_path = "demo_analysis\intermediate_data\model_training_sequence"
        
    def __init__(self, the_map="de_dust2"):
        self.map = the_map
        self.data_map = data_map_X_1
        
    def collect_demo_files(self):
        """Collects all of the demo files that will be used to compute the expected Win_rate of fights

        Returns:
            files (list): a list of the demo files in the folder
        """

        files = os.listdir(os.path.join(self.folder_path,self.map))
        return files
    
    def create_data_save_path(self):
        """creates the folder to store data for training the model
        
        returns:
            path (str): path to the folder
        """
        folders = os.listdir(self.train_data_path)
        if len(folders) == 0:
            the_folder = os.path.join(self.train_data_path, '0')
            os.mkdir(the_folder)
            return the_folder
        index = max([int(folder) for folder in folders])
        the_folder = os.path.join(self.train_data_path, str(index+1))
        os.mkdir(the_folder)
        return the_folder
    
    def create_sequence_data_save_path(self):
        """creates the folder to store data for training the model
        
        returns:
            path (str): path to the folder
        """        
        folders = os.listdir(self.train_sequence_data_path)
        if len(folders) == 0:
            the_folder = os.path.join(self.train_sequence_data_path, '0')
            os.mkdir(the_folder)
            return the_folder
        index = max([int(folder) for folder in folders])
        the_folder = os.path.join(self.train_sequence_data_path, str(index+1))
        os.mkdir(the_folder)
        return the_folder
        
    def prepare_data(self, num=None, resume=False, resume_dir=None, resume_idx=None):
        """reads out the desired data from the demo files and prepares it for the model
        """
        demo_files = self.collect_demo_files()
        if num is not None:
            demo_files = demo_files[:num]
        
        if resume == True and resume_dir is not None and resume_idx is not None:  
            demo_files = demo_files[resume_idx:]
            data_save_path = resume_dir    
        else:      
            data_save_path = self.create_data_save_path()
        print(f"saving to {data_save_path}")

        for demo_file in tqdm(demo_files, desc=f"Processing demo"):
            try:
                parser = Processor_Efficient(os.path.join(self.folder_path, self.map, demo_file))
                df_all_ticks = parser.parser.parse_ticks(wanted_props=self.wanted_props)
            except Exception as e:
                print(e)
                continue
            
            # Assign players to teams
            team_A, team_B = get_teams(df_all_ticks)        
            if team_A is None:
                continue
            team_dict = {}
            for member in team_A:
                team_dict[member] = 'A'
            for member in team_B:
                team_dict[member] = 'B'
            
            team_A = set(team_A)
            team_B = set(team_B)
            
            # Preprocessing
            try:
                # Convert boolean values to int
                for column in self.bool_to_int:
                    c = column                    
                    df_all_ticks[column] = df_all_ticks[column].astype(int)
            except TypeError:
                print(f"Error in {demo_file} at {c}")
                rows_with_none = df_all_ticks[df_all_ticks[c].isnull()]
                print(rows_with_none)
                continue
            
            # Get the ticks for training
            ticks_train_ranges, attackers, attackees = parser.ticks_for_training()
            
            # To save Memory:
            # # Only keep the ticks in ticks_train_ranges
            # df_all_ticks = df_all_ticks[df_all_ticks['tick'].isin([x for x in range(ticks_train_ranges[0][0], ticks_train_ranges[-1][1])])] 
            
            players = set(attackers).union(set(attackees))
            player_dfs = {player: None for player in players}
            for player in players:
                player_df = df_all_ticks[df_all_ticks['name'] == player]
                flash_duration = 0
                # Flash var
                for index, row in player_df.iterrows():
                    if row['flash_duration'] == 0.0:
                        flash_duration = 0
                    else:
                        # When the player is in flash status
                        if flash_duration == 0:
                            # First tick flashed
                            flash_duration = row['flash_duration']
                        else:
                            flash_duration -= 1/tick_rate
                            row['flash_duration'] = flash_duration

                player_dfs[player] = player_df
                
                
            for i, tick_range in enumerate(ticks_train_ranges):
                start_tick, end_tick = tick_range
                ### Get the desired info into string format seperated by '\t'
                attacker = attackers[i]
                attackee = attackees[i]
                choice = random.choice([0, 1])
                Y = choice
                # Y == 1 indicates that the player at the 2nd position wins
                if Y == 1: attacker, attackee = attackee, attacker
                
                def grab_X_info(start_index, end_index, df, name_player):
                    # Grab only the start_index row
                    start_tick_df = df_all_ticks[df_all_ticks['tick'] == start_tick]
                    X_info = []
                    X_info.append(df.loc[start_index]['X'])
                    X_info.append(df.loc[start_index]['Y'])
                    X_info.append(df.loc[start_index]['Z'])
                    
                    if team_dict[name_player] == 'A':
                        teammates = start_tick_df[start_tick_df['name'].isin(team_A) & start_tick_df['is_alive']]
                    else:
                        teammates = start_tick_df[start_tick_df['name'].isin(team_B) & start_tick_df['is_alive']]
                    
                    teammates = teammates[teammates['name'] != name_player]
                    teammates = teammates.drop_duplicates()
                    
                    alive_num = len(teammates)
                    for i, row in teammates.iterrows():
                        X_info.append(row['X'])
                        X_info.append(row['Y'])
                        X_info.append(row['Z'])
                    
                    for i in range(4-alive_num):
                        X_info.append(0)
                        X_info.append(0)
                        X_info.append(0)
                    
                    assert len(X_info) == 15, print(len(X_info), "not equal to 15")

                    X_info.append(df.loc[start_index]['health'])
                    armor_value = df.loc[start_index]['armor_value']
                    X_info.append(1 if armor_value > 0 else 0)
                    X_info.append(df.loc[start_index]['has_helmet'])
                    X_info.append(df.loc[start_index]['pitch'])
                    X_info.append(df.loc[start_index]['yaw'])
                    X_info.append(df.loc[start_index]['flash_duration'])
                    weapon = weapon_one_hot_vector(df.loc[start_index]['active_weapon_name'])
                    X_info += weapon
                    return [X_info]
                
                def add_pitch_diff(X_vec_0, X_vec_1):
                    att_X, att_Y, att_Z = X_vec_0[:3]
                    oppo_X, oppo_Y, oppo_Z = X_vec_1[:3]
                    att_pitch, att_yaw = X_vec_0[18], X_vec_0[19]
                    oppo_pitch, oppo_yaw = X_vec_1[18], X_vec_1[19]
                    att_pitch_yaw_diff = pitch_yaw_diff_from_enemy(att_pitch, att_yaw, att_X, att_Y, att_Z, oppo_X, oppo_Y, oppo_Z)
                    oppo_pitch_yaw_diff = pitch_yaw_diff_from_enemy(oppo_pitch, oppo_yaw, oppo_X, oppo_Y, oppo_Z, att_X, att_Y, att_Z)
                    X_vec_0 += list(att_pitch_yaw_diff)
                    X_vec_1 += list(oppo_pitch_yaw_diff)
                    
                try:                
                    start_index_att = player_dfs[attacker].loc[player_dfs[attacker]['tick'] == start_tick].index[0]
                    end_index_att = player_dfs[attacker].loc[player_dfs[attacker]['tick'] == end_tick].index[0]
                    start_index_oppo = player_dfs[attackee].loc[player_dfs[attackee]['tick'] == start_tick].index[0]
                    end_index_oppo = player_dfs[attackee].loc[player_dfs[attackee]['tick'] == end_tick].index[0]

                except IndexError:
                    print(f"Error in {demo_file} at {start_tick} - {end_tick}")
                    with open(os.path.join(data_save_path, 'data.txt'), 'a') as f:
                        f.write(f"Error in {demo_file} at {start_tick} - {end_tick}\n")
                    continue
                
                try:
                    att_info = grab_X_info(start_index_att, end_index_att, player_dfs[attacker], attacker)
                    oppo_info = grab_X_info(start_index_oppo, end_index_oppo, player_dfs[attackee], attackee)
                except:
                    continue
                
                all_info = []
                
                X_vec_0 = att_info[0]
                X_vec_1 = oppo_info[0]
                add_pitch_diff(X_vec_0, X_vec_1)
                X = X_vec_0 + X_vec_1 + [Y]
                all_info.append(X)
                
                with open(os.path.join(data_save_path, 'data.txt'), 'a') as f:
                    for X in all_info:
                        f.write('\t'.join([str(x) for x in X]) + '\n')
                
            with open(os.path.join(data_save_path, 'log.txt'), 'a') as f_log:
                f_log.write(f"Finished processing {demo_file}" + "\n")
                
            del df_all_ticks, player_dfs
            gc.collect()       
            print("cleaned the df_all_ticks, player_dfs")
            
    def prepare_sequence_data(self, num=None, resume=False, resume_dir=None, resume_idx=None, demo_lst=None):
        """reads out the desired data from the demo files and prepares it for the model
        """
        demo_files = self.collect_demo_files()
        if num is not None:
            demo_files = demo_files[:num]
        
        if resume == True and resume_dir is not None and resume_idx is not None:  
            demo_files = demo_files[resume_idx:]
            data_save_path = resume_dir    
        else:      
            data_save_path = self.create_sequence_data_save_path()
        print(f"saving to {data_save_path}")

        for demo_file in tqdm(demo_files, desc=f"Processing demo"):
            try:
                parser = Processor_Efficient(os.path.join(self.folder_path, self.map, demo_file))
                df_all_ticks = parser.parser.parse_ticks(wanted_props=self.wanted_props)
            except Exception as e:
                print(e)
                continue
            
            # Assign players to teams
            team_A, team_B = get_teams(df_all_ticks)        
            if team_A is None:
                continue
            team_dict = {}
            for member in team_A:
                team_dict[member] = 'A'
            for member in team_B:
                team_dict[member] = 'B'
            
            team_A = set(team_A)
            team_B = set(team_B)
            
            # Preprocessing
            try:
                # Convert boolean values to int
                for column in self.bool_to_int:
                    c = column                    
                    df_all_ticks[column] = df_all_ticks[column].astype(int)
            except TypeError:
                print(f"Error in {demo_file} at {c}")
                rows_with_none = df_all_ticks[df_all_ticks[c].isnull()]
                print(rows_with_none)
                continue
            
            # Get the ticks for training
            ticks_train_ranges, attackers, attackees = parser.ticks_for_training(fixed_range=True)
            
            # To save Memory:
            # # Only keep the ticks in ticks_train_ranges
            # df_all_ticks = df_all_ticks[df_all_ticks['tick'].isin([x for x in range(ticks_train_ranges[0][0], ticks_train_ranges[-1][1])])] 
            
            players = set(attackers).union(set(attackees))    
            player_dfs = {player: None for player in players}
            for player in players:
                player_df = df_all_ticks[df_all_ticks['name'] == player]
                flash_duration = 0
                # Flash var
                for index, row in player_df.iterrows():
                    if row['flash_duration'] == 0.0:
                        flash_duration = 0
                    else:
                        # When the player is in flash status
                        if flash_duration == 0:
                            # First tick flashed
                            flash_duration = row['flash_duration']
                        else:
                            flash_duration -= 1/tick_rate
                            row['flash_duration'] = flash_duration

                player_dfs[player] = player_df
            print("player_dfs ready")    
            
            for i, tick_range in tqdm(enumerate(ticks_train_ranges), total=len(ticks_train_ranges), desc=f"Processing {demo_file}"):
                start_tick, end_tick = tick_range
                ### Get the desired info into string format seperated by '\t'
                attacker = attackers[i]
                attackee = attackees[i]
                if attacker is None or attackee is None:
                    continue
                choice = random.choice([0, 1])
                Y = choice
                # Y == 1 indicates that the player at the 2nd position wins
                if Y == 1: attacker, attackee = attackee, attacker
                
                def grab_X_info_seq(start_tick, end_tick, df, name_player):
                    tick_interval = 10
                    X_info_output = []
                    df_new = df.drop_duplicates()
                    try:
                        start_tick_idx = df_new.index.get_loc(df_new.loc[df_new['tick'] == start_tick].index[0])
                        end_tick_idx = df_new.index.get_loc(df_new.loc[df_new['tick'] == end_tick].index[0])
                    except IndexError:
                        return X_info_output
                    
                    for tick_idx in range(0, end_tick_idx-start_tick_idx, tick_interval):
                        curr_idx = tick_idx + start_tick_idx
                        curr_tick = df_new.iloc[tick_idx]['tick']
                        X_info = []
                        
                        tick_df = df_all_ticks[df_all_ticks['tick'] == curr_tick]
                        if team_dict[name_player] == 'A':
                            teammates = tick_df[tick_df['name'].isin(team_A) & tick_df['is_alive']]
                        else:
                            teammates = tick_df[tick_df['name'].isin(team_B) & tick_df['is_alive']]
                                
                        teammates = teammates[teammates['name'] != name_player]
                        teammates = teammates.drop_duplicates()
                        alive_num = len(teammates)
                        
                        X_info.append(df_new.iloc[curr_tick]['X'])
                        X_info.append(df_new.iloc[curr_tick]['Y'])
                        X_info.append(df_new.iloc[curr_tick]['Z'])
                        
                        for i, row in teammates.iterrows():
                            X_info.append(row['X'])
                            X_info.append(row['Y'])
                            X_info.append(row['Z'])
                        
                        for i in range(4-alive_num):
                            X_info.append(0)
                            X_info.append(0)
                            X_info.append(0)
                            
                        assert len(X_info) == 15, print(len(X_info), "not equal to 15")
                        X_info.append(df_new.iloc[curr_tick]['health'])
                        armor_value = df_new.iloc[curr_tick]['armor_value']
                        X_info.append(1 if armor_value > 0 else 0)
                        X_info.append(df_new.iloc[curr_tick]['has_helmet'])
                        X_info.append(df_new.iloc[curr_tick]['pitch'])
                        X_info.append(df_new.iloc[curr_tick]['yaw'])
                        X_info.append(df_new.iloc[curr_tick]['flash_duration'])
                        weapon = weapon_one_hot_vector(df_new.iloc[curr_tick]['active_weapon_name'])
                        
                        X_info += weapon
                        
                        X_info_output.append(X_info)
                        
                    return X_info_output
                
                def add_pitch_diff(X_vec_0, X_vec_1):
                    att_X, att_Y, att_Z = X_vec_0[:3]
                    oppo_X, oppo_Y, oppo_Z = X_vec_1[:3]
                    att_pitch, att_yaw = X_vec_0[18], X_vec_0[19]
                    oppo_pitch, oppo_yaw = X_vec_1[18], X_vec_1[19]
                    att_pitch_yaw_diff = pitch_yaw_diff_from_enemy(att_pitch, att_yaw, att_X, att_Y, att_Z, oppo_X, oppo_Y, oppo_Z)
                    oppo_pitch_yaw_diff = pitch_yaw_diff_from_enemy(oppo_pitch, oppo_yaw, oppo_X, oppo_Y, oppo_Z, att_X, att_Y, att_Z)
                    X_vec_0 += list(att_pitch_yaw_diff)
                    X_vec_1 += list(oppo_pitch_yaw_diff)
                
                try:
                    
                    att_info = grab_X_info_seq(start_tick, end_tick, player_dfs[attacker], attacker)
                    oppo_info = grab_X_info_seq(start_tick, end_tick, player_dfs[attackee], attackee)
                    
                    assert len(att_info) == len(oppo_info), print(len(att_info), len(oppo_info), "length not match")
                    
                except KeyError:
                    print(f"Error in {demo_file} at {start_tick} - {end_tick}")
                    with open(os.path.join(data_save_path, 'log.txt'), 'a') as f_log:
                        f_log.write(f"Error in {demo_file} at {start_tick} - {end_tick}\n")
                    continue
                
                all_info = []
                
                for index_seq, (X_vec_0, X_vec_1) in enumerate(zip(att_info, oppo_info)):
                    add_pitch_diff(X_vec_0, X_vec_1)
                    # index 121: index_seq
                    # index 122: demo_file
                    # index 123: start_tick
                    # index 124: end_tick
                    X = X_vec_0 + X_vec_1 + [Y] + [index_seq, demo_file, start_tick, end_tick]
                    all_info.append(X)
                
                with open(os.path.join(data_save_path, 'data.txt'), 'a') as f:
                    for X in all_info:
                        f.write('\t'.join([str(x) for x in X]) + '\n')
                
            with open(os.path.join(data_save_path, 'log.txt'), 'a') as f_log:
                f_log.write(f"Finished processing {demo_file}" + "\n")
                
            del df_all_ticks, player_dfs
            gc.collect()       
            print("cleaned the df_all_ticks, player_dfs")
        

    def load_data(self, data_path=None):
        """Loads the data from the data.txt file

        Args:
            data_path (string): path to the data file
        
        Returns:
            X, Y: X, Y dataframes of player information
        """

        print(os.path.join(data_path,'data.txt'))
        with open(os.path.join(data_path, 'data.txt'), 'r') as f:
            data = f.readlines()
        
        # Only use the first 10_000 data
        # data = data[:10_000]
        
        # Remove lines starting with "Error"
        data = [x for x in data if (not x.startswith("Error"))]
        
    
        data = [x.strip().split('\t') for x in data]
        
        data = [x for x in data if (len(x) == 121)]
        # Make that into csv
        df = pd.DataFrame(data)
        # Drop all columns after column 127
        print(f"the length of the dataframe: {len(df)}")
        
        # df.replace({'True': 1, 'False': 0}, inplace=True)
        
        df.iloc[:, 15:-1] = df.iloc[:, 15:-1].replace({'True': 1, 'False': 0})
        
        X = df.iloc[:, :-1]
        print(f"value counts: {df.iloc[:, -1].value_counts()}")
        Y = df.iloc[:, -1]
        return X, Y
    
    def load_seq_data(self, data_path=None):
        print(os.path.join(data_path,'data.txt'))
        with open(os.path.join(data_path, 'data.txt'), 'r') as f:
            data = f.readlines()
        # Remove lines starting with "Error"
        data = [x for x in data if (not x.startswith("Error"))]
        data = [x.strip().split('\t')[:-3] for x in data]
        data = [x for x in data if (len(x) == 122)]
        # Make that into csv
        df = pd.DataFrame(data)
        # Drop all columns after column 127
        print(f"the length of the dataframe: {len(df)}")
                
        df.iloc[:, 15:-1] = df.iloc[:, 15:-1].replace({'True': 1, 'False': 0})
        
        Y = df.iloc[:, -2]
        X = df.drop(df.columns[-2], axis=1)
        print(f"value counts: {df.iloc[:, -1].value_counts()}")
        return X, Y
        
        # Note: 
        # Index 120 indicates the Winner, 0 means the former, 1 indicates the latter
        # Index 121 indicates the sequence index, 0 represents the start of the sequence
        # Index 122 is the demo_file name, for debug use
        # Index 123, 124 are the start_tick, end_tick range for this sequence, for debug us
        
    def debug_seq_data(self, X_seqs, Y_seqs, traj_indexes):
        """
            After self.prepare_seq, ensure that X_seq and Y_seq are correct by manually plotting the sequence information on the map using a annimation
            
            Args:
            X_seq (np.array): sequences of predictor information shape(num of seqs, tick_per_seq, X_dim)
            Y_seq (np.array): sequences of winner boolean (num of seqs, 1)
            traj_index (int): index of the sequence to be visualized
        
        """    
        for traj_index in traj_indexes:
            seq_to_visualize = X_seqs[traj_index]
            plot_move_seq(seq_to_visualize, self.data_map)

    def case_study(self, demo_path, player_to_checkout):
        """Computes the expected win rate of a player in a demo file

        Args:
            demo_path (string): path to the demo file
            player_to_checkout (string): player name to checkout
        """
        parser =  Processor_Efficient(demo_path)
        df_all_ticks = parser.parser.parse_ticks(wanted_props=self.wanted_props)
        for column in self.bool_to_int:
            try:
                df_all_ticks[column] = df_all_ticks[column].astype(int)
            except TypeError:
                print(f"Error in {demo_file} at {column}")
                rows_with_none = df_all_ticks[df_all_ticks[column].isnull()]
                print(rows_with_none)
                break
        ticks_train_ranges, attackers, attackees = parser.ticks_for_training()
        players = list(set(attackers).union(set(attackees)))
        
        # Assign players to teams
        team_A, team_B = get_teams(df_all_ticks)        
        if team_A is None:
            return
        team_dict = {}
        for member in team_A:
            team_dict[member] = 'A'
        for member in team_B:
            team_dict[member] = 'B'
        
        team_A = set(team_A)
        team_B = set(team_B)
        
        print(players)
        print(f"entered a valid player name: {player_to_checkout in players}")
        player_dfs = {player: None for player in players}
        for player in players:
            player_df = df_all_ticks[df_all_ticks['name'] == player]
            flash_duration = 0
            # Flash var
            for index, row in player_df.iterrows():
                if row['flash_duration'] == 0.0:
                    flash_duration = 0
                else:
                    if flash_duration == 0:
                        flash_duration = row['flash_duration']
                    else:
                        flash_duration -= 1/16
                        row['flash_duration'] = flash_duration

            player_dfs[player] = player_df

        xWinRate_map = {}
        for player_to_checkout in players:
            if player_to_checkout is None:
                continue
            player_to_checkout_expected_win_times = 0
            player_to_checkout_actual_win_times = 0
            player_to_checkout_engaged_times = 0
            for i, tick_range in enumerate(ticks_train_ranges):
                start_tick, end_tick = tick_range
                attacker = attackers[i]
                attackee = attackees[i]
                
                def grab_X_info(start_index, end_index, df, name_player):
                    # Grab only the start_index row
                    start_tick_df = df_all_ticks[df_all_ticks['tick'] == start_tick]
                    X_info = []
                    X_info.append(df.loc[start_index]['X'])
                    X_info.append(df.loc[start_index]['Y'])
                    X_info.append(df.loc[start_index]['Z'])
                    
                    if team_dict[name_player] == 'A':
                        teammates = start_tick_df[start_tick_df['name'].isin(team_A) & start_tick_df['is_alive']]
                    else:
                        teammates = start_tick_df[start_tick_df['name'].isin(team_B) & start_tick_df['is_alive']]
                    
                    teammates = teammates[teammates['name'] != name_player]
                    teammates = teammates.drop_duplicates()
                    
                    alive_num = len(teammates)
                    for i, row in teammates.iterrows():
                        X_info.append(row['X'])
                        X_info.append(row['Y'])
                        X_info.append(row['Z'])
                    
                    for i in range(4-alive_num):
                        X_info.append(0)
                        X_info.append(0)
                        X_info.append(0)
                    
                    assert len(X_info) == 15, print(len(X_info), "not equal to 15")

                    X_info.append(df.loc[start_index]['health'])
                    armor_value = df.loc[start_index]['armor_value']
                    X_info.append(1 if armor_value > 0 else 0)
                    X_info.append(df.loc[start_index]['has_helmet'])
                    X_info.append(df.loc[start_index]['pitch'])
                    X_info.append(df.loc[start_index]['yaw'])
                    X_info.append(df.loc[start_index]['flash_duration'])
                    weapon = weapon_one_hot_vector(df.loc[start_index]['active_weapon_name'])
                    X_info += weapon
                    
                    return [X_info]
                
                def add_pitch_diff(X_vec_0, X_vec_1):
                    att_X, att_Y, att_Z = X_vec_0[:3]
                    oppo_X, oppo_Y, oppo_Z = X_vec_1[:3]
                    att_pitch, att_yaw = X_vec_0[18], X_vec_0[19]
                    oppo_pitch, oppo_yaw = X_vec_1[18], X_vec_1[19]
                    att_pitch_yaw_diff = pitch_yaw_diff_from_enemy(att_pitch, att_yaw, att_X, att_Y, att_Z, oppo_X, oppo_Y, oppo_Z)
                    oppo_pitch_yaw_diff = pitch_yaw_diff_from_enemy(oppo_pitch, oppo_yaw, oppo_X, oppo_Y, oppo_Z, att_X, att_Y, att_Z)
                    X_vec_0 += list(att_pitch_yaw_diff)
                    X_vec_1 += list(oppo_pitch_yaw_diff)
                    
                try:
                    start_index_att = player_dfs[attacker].loc[player_dfs[attacker]['tick'] == start_tick].index[0]
                    end_index_att = player_dfs[attacker].loc[player_dfs[attacker]['tick'] == end_tick].index[0]
                    start_index_oppo = player_dfs[attackee].loc[player_dfs[attackee]['tick'] == start_tick].index[0]
                    end_index_oppo = player_dfs[attackee].loc[player_dfs[attackee]['tick'] == end_tick].index[0]
                except IndexError:
                    continue

                if attacker == player_to_checkout:
                    player_to_checkout_actual_win_times += 1
                    player_to_checkout_engaged_times += 1
                elif attackee == player_to_checkout:
                    player_to_checkout_engaged_times += 1
                else: 
                    continue
                

                att_info = grab_X_info(start_index_att, end_index_att, player_dfs[attacker], attacker)
                oppo_info = grab_X_info(start_index_oppo, end_index_oppo, player_dfs[attackee], attackee)
                X_vec_0 = att_info[0]
                X_vec_1 = oppo_info[0]
                add_pitch_diff(X_vec_0, X_vec_1)
                X = X_vec_0 + X_vec_1
                assert len(X) == 120, f"len(X) = {len(X)} not equal to 120"

                X = [X]
                
                y = self.predict_w_model(X)
                
                y = int(y[0])
                
                if y == 0 and attacker == player_to_checkout or y == 1 and attackee == player_to_checkout:
                    player_to_checkout_expected_win_times += 1
                if player_to_checkout_engaged_times == 0:
                    continue
                
            # print(f"win_times {player_to_checkout_expected_win_times}")
            # print(f"total_times {player_to_checkout_engaged_times}")
            x_rate = player_to_checkout_expected_win_times / player_to_checkout_engaged_times
            actual_rate = player_to_checkout_actual_win_times / player_to_checkout_engaged_times
            print(f"{player_to_checkout} expected to win {x_rate}, actually won {actual_rate}")
            xWinRate_map[player_to_checkout] = (x_rate, actual_rate)
            
        return xWinRate_map
    
    def load_sample_demo(self, demo_path):
        parser = Processor_Efficient(demo_path)
        df_all_ticks = parser.parser.parse_ticks(wanted_props=self.wanted_props)

        team_A, team_B = get_teams(df_all_ticks)   
        
        if team_A is None:
            raise ValueError
        team_dict = {}
        for member in team_A:
            team_dict[member] = 'A'
        for member in team_B:
            team_dict[member] = 'B'
        
        team_A = set(team_A)
        team_B = set(team_B)
        
        try:
        # Convert boolean values to int
            for column in self.bool_to_int:
                c = column                    
                df_all_ticks[column] = df_all_ticks[column].astype(int)
        except TypeError:
            print(f"Error at {c}")
        
        # Get the ticks for training
        ticks_train_ranges, attackers, attackees = parser.ticks_for_training()
        players = set(attackers).union(set(attackees))
        player_dfs = {player: None for player in players}
        for player in players:
            player_df = df_all_ticks[df_all_ticks['name'] == player]
            flash_duration = 0
            # Flash var
            for index, row in player_df.iterrows():
                if row['flash_duration'] == 0.0:
                    flash_duration = 0
                else:
                    # When the player is in flash status
                    if flash_duration == 0:
                        # First tick flashed
                        flash_duration = row['flash_duration']
                    else:
                        flash_duration -= 1/tick_rate
                        row['flash_duration'] = flash_duration

            player_dfs[player] = player_df
        
        # Pick one index here
        self.ticks_train_ranges = ticks_train_ranges
        self.attackers = attackers
        self.attackees = attackees
        self.df_all_ticks = df_all_ticks
        self.player_dfs = player_dfs
        self.team_dict = team_dict
        self.team_A = team_A
        self.team_B = team_B
    
    def gather_predict_sample(self, index_checkout):
        i = index_checkout
        tick_range = self.ticks_train_ranges[index_checkout]
        start_tick, end_tick = tick_range
        attacker = self.attackers[i]
        attackee = self.attackees[i]
                
        def grab_X_info(index_todo, df, df_all_ticks, name_player, team_dict, team_A, team_B):
            """Gets the desired X info to feed into the model

            Args:
                index_todo (int): index to collect
                df (data frame): player df
                df_all_ticks (data frame): df to get teammate information
                name_player (str): name of the player

            Returns:
                list: prepared X data ready to be feed into the model
            """
            # Grab only the index row            
            start_tick_df = df_all_ticks[df_all_ticks['tick'] == start_tick]
            X_info = []
            X_info.append(df.loc[index_todo]['X'])
            X_info.append(df.loc[index_todo]['Y'])
            X_info.append(df.loc[index_todo]['Z'])
                        
            if team_dict[name_player] == 'A':
                teammates = start_tick_df[start_tick_df['name'].isin(team_A) & start_tick_df['is_alive']]
            else:
                teammates = start_tick_df[start_tick_df['name'].isin(team_B) & start_tick_df['is_alive']]
                
            teammates = teammates[teammates['name'] != name_player]
            teammates = teammates.drop_duplicates()
                        
            alive_num = len(teammates)
            # print(f"teammates: {teammates}")
            for i, row in teammates.iterrows():
                X_info.append(row['X'])
                X_info.append(row['Y'])
                X_info.append(row['Z'])
                        
            for i in range(4-alive_num):
                X_info.append(0)
                X_info.append(0)
                X_info.append(0)
            
            assert len(X_info) == 15, print(len(X_info), "not equal to 15")
            
            X_info.append(df.loc[index_todo]['health'])
            armor_value = df.loc[index_todo]['armor_value']
            X_info.append(1 if armor_value > 0 else 0)
            X_info.append(df.loc[index_todo]['has_helmet'])
            X_info.append(df.loc[index_todo]['pitch'])
            X_info.append(df.loc[index_todo]['yaw'])
            X_info.append(df.loc[index_todo]['flash_duration'])
            if X_info[-1] != 0:
                print(f"flash: {X_info[-1]}")
            weapon = weapon_one_hot_vector(df.loc[index_todo]['active_weapon_name'])
            X_info += weapon
            return [X_info]
        
        def add_pitch_diff(X_vec_0, X_vec_1):
            att_X, att_Y, att_Z = X_vec_0[:3]
            oppo_X, oppo_Y, oppo_Z = X_vec_1[:3]
            att_pitch, att_yaw = X_vec_0[18], X_vec_0[19]
            oppo_pitch, oppo_yaw = X_vec_1[18], X_vec_1[19]
            att_pitch_yaw_diff = pitch_yaw_diff_from_enemy(att_pitch, att_yaw, att_X, att_Y, att_Z, oppo_X, oppo_Y, oppo_Z)
            oppo_pitch_yaw_diff = pitch_yaw_diff_from_enemy(oppo_pitch, oppo_yaw, oppo_X, oppo_Y, oppo_Z, att_X, att_Y, att_Z)
            X_vec_0 += list(att_pitch_yaw_diff)
            X_vec_1 += list(oppo_pitch_yaw_diff)
        
        player_dfs = self.player_dfs
        # ----- This blocks handles exceptions --------------------------
        try:                
            start_index_att = player_dfs[attacker].loc[player_dfs[attacker]['tick'] == start_tick].index[0]
            end_index_att = player_dfs[attacker].loc[player_dfs[attacker]['tick'] == end_tick].index[0]
            start_index_oppo = player_dfs[attackee].loc[player_dfs[attackee]['tick'] == start_tick].index[0]
            end_index_oppo = player_dfs[attackee].loc[player_dfs[attackee]['tick'] == end_tick].index[0]
        except IndexError:
            print(f"Error at {start_tick} - {end_tick}")
            return
        # ----------------------------------------------------------------    

        att_info = grab_X_info(start_index_att, player_dfs[attacker], self.df_all_ticks, attacker, self.team_dict, self.team_A, self.team_B)
        oppo_info = grab_X_info(start_index_oppo, player_dfs[attackee], self.df_all_ticks, attackee, self.team_dict, self.team_A, self.team_B)
        
        X_vec_0 = att_info[0]
        X_vec_1 = oppo_info[0]

        add_pitch_diff(X_vec_0, X_vec_1)
        X = X_vec_0 + X_vec_1
        assert len(X) == 120, "X should have length of 120"
        
        return X
    
    def visualize_predict_sample(self, X):
        assert len(X) == 120, "X should have length of 120"
                        
class xFight_Linear_Approximation(Expected_Win_Rate):
    """This is a linear approximation of the expected win rate of fights in CSGO
    
    What is it measuring:
        For each tick, depending on several factors, what is the probability of the player winning this fight (Prob of killing the other one)
    
    How to determine a fight:
        The fight is found using the "death_event"
    """
    def __init__(self, the_map="de_dust2"):
        super().__init__(the_map)
        
    def model_training(self, data_path=None):
        X, y = self.load_data(data_path)
        print(len(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        print(f"Training MSE: {mse_train}")
        print(f"Test MSE: {mse_test}")
        print(f"Training R2: {r2_train}")
        print(f"Test R2: {r2_test}")
        
class xFight_Logistic_Regression(Expected_Win_Rate):
    def __init__(self, the_map="de_dust2"):
        super().__init__(the_map)
        
    def model_training(self, data_path=None):
        X, y = self.load_data(data_path)
        print(len(X))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Logistic Regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Predict probabilities
        y_train_prob = self.model.predict_proba(X_train)[:, 1]
        y_test_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Evaluate the model using Log Loss
        self.log_model(y_train, y_test, y_train_prob, y_test_prob, y_train_pred, y_test_pred)
        
    def log_model(self, y_train, y_test, y_train_prob, y_test_prob, y_train_pred, y_test_pred):
        
        log_loss_train = log_loss(y_train, y_train_prob)
        log_loss_test = log_loss(y_test, y_test_prob)

        # Additional metrics
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # precision_test = precision_score(y_test, y_test_pred)
        # recall_test = recall_score(y_test, y_test_pred)
        # f1_test = f1_score(y_test, y_test_pred)
        # roc_auc_test = roc_auc_score(y_test, y_test_prob)
        confusion_test = confusion_matrix(y_test, y_test_pred)

        # Print evaluation metrics
        print(f"Training Log Loss: {log_loss_train}")
        print(f"Test Log Loss: {log_loss_test}")
        print(f"Test Accuracy: {accuracy_test}")
        # print(f"Test Precision: {precision_test}")
        # print(f"Test Recall: {recall_test}")
        # print(f"Test F1-Score: {f1_test}")
        # print(f"Test ROC AUC: {roc_auc_test}")
        print(f"Test Confusion Matrix:\n{confusion_test}")
        
    def predict_w_model(self, X):
        # print("predicting")
        return self.model.predict(X)

class NeuralNetLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(NeuralNetLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
    
class xFight_LSTM(Expected_Win_Rate):
    def __init__(self, the_map="de_dust2"):
        super().__init__(the_map)
        self.data_map = data_map_X_1
        
    def prepare_seq(self, X, y):
        
        # Ensure data is in NumPy array format
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        X_seqs = []
        y_seqs = []
        
        X_seq = []
        y_seq = []
        
        max_seq_length = 5
        seq_idx = 0
        for index, x_data in enumerate(X):
            data_seq_idx = x_data[-1]
            if data_seq_idx == seq_idx:
                X_seq.append(x_data[:-1])
                y_seq.append(y[index])
                seq_idx += 1
            # Not matching, indicating end of seq
            # Possiblity 1: reached length of 5 (max size)
            # Possibilty 2: < length of 5
            else: 
                if seq_idx != max_seq_length:
                    for _ in range(seq_idx, max_seq_length):
                        X_seq.append([0] * 120)
                        y_seq.append(0)
                        
                assert len(X_seq) == max_seq_length, "X_seq length not equal to 5"
                assert len(y_seq) == max_seq_length, "y_seq length not equal to 5"
                X_seqs.append(X_seq)
                y_seqs.append(y_seq)
                X_seq = []
                y_seq = []
                X_seq.append(x_data[:-1])
                y_seq.append(y[index])
                seq_idx = 1
                
        X_seqs = np.array(X_seqs, dtype=np.float32)
        y_seqs = np.array(y_seqs, dtype=np.float32)

        debug_index_lst = [0, 111, 5557, 1241, 5422]
        self.debug_seq_data(X_seqs, y_seqs, debug_index_lst)

        return X_seqs, y_seqs
    
    # Read this
    # https://colah.github.io/posts/2015-08-Understanding-LSTMs/
            
    def model_training(self, data_path=None):
        # Load data
        X, y = self.load_seq_data(data_path)
        print(f"Total samples: {len(X)}")
        
        # Prepare sequences for LSTM
        X_seq, y_seq = self.prepare_seq(X, y)
                
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize the LSTM model
        input_dim = X_train.shape[2]  # Number of features per time step
        self.model = NeuralNetLSTM(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
                
        # Evaluate the model
        self.model.eval()
        with torch.no_grad():
            y_train_prob = self.model(X_train_tensor).cpu().numpy().ravel()
            y_test_prob = self.model(X_test_tensor).cpu().numpy().ravel()
            
        y_train_pred = (y_train_prob > 0.5).astype(int)
        y_test_pred = (y_test_prob > 0.5).astype(int)
        
        # Log evaluation metrics
        self.log_model(y_train, y_test, y_train_prob, y_test_prob, y_train_pred, y_test_pred)
        
    def log_model(self, y_train, y_test, y_train_prob, y_test_prob, y_train_pred, y_test_pred):
        # Calculate evaluation metrics
        log_loss_train = log_loss(y_train, y_train_prob)
        log_loss_test = log_loss(y_test, y_test_prob)

        accuracy_test = accuracy_score(y_test, y_test_pred)
        precision_test = precision_score(y_test, y_test_pred, zero_division=0)
        recall_test = recall_score(y_test, y_test_pred, zero_division=0)
        f1_test = f1_score(y_test, y_test_pred, zero_division=0)
        roc_auc_test = roc_auc_score(y_test, y_test_prob)
        confusion_test = confusion_matrix(y_test, y_test_pred)
        
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        ss_res = ((y_test - y_test_prob) ** 2).sum()
        r2_score_value = 1 - (ss_res / ss_tot)

        # Print evaluation metrics
        print(f"Training Log Loss: {log_loss_train:.4f}")
        print(f"Test Log Loss: {log_loss_test:.4f}")
        print(f"Test Accuracy: {accuracy_test:.4f}")
        print(f"Test Precision: {precision_test:.4f}")
        print(f"Test Recall: {recall_test:.4f}")
        print(f"Test F1-Score: {f1_test:.4f}")
        print(f"Test ROC AUC: {roc_auc_test:.4f}")
        print(f"Test Confusion Matrix:\n{confusion_test}")
        print(f"Test R² Score: {r2_score_value:.4f}")
        
    def predict_w_model(self, X, output_prob=False, seq_length=10):
        self.model.eval()
        if isinstance(X, list):
            X = np.array(X, dtype=np.float32)
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        else:
            raise TypeError("Input X must be a list or numpy array.")
        
        # Prepare sequences
        if X.ndim == 1:
            # Single sample input
            if len(X) < seq_length:
                raise ValueError("Input sequence is shorter than the required sequence length.")
            X_seq = [X[-seq_length:]]
        elif X.ndim == 2:
            # Multiple samples input
            if X.shape[0] < seq_length:
                raise ValueError("Number of samples is less than the sequence length.")
            X_seq = []
            for i in range(len(X) - seq_length + 1):
                X_seq.append(X[i:i+seq_length])
        else:
            raise ValueError("Input X must be 1D or 2D array.")
        
        X_seq = np.array(X_seq, dtype=np.float32)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        with torch.no_grad():
            output = self.model(X_tensor)
        
        probabilities = output.cpu().numpy().ravel()
        if output_prob:
            return probabilities
            
        return (probabilities > 0.5).astype(int)
        
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class xFight_Neural_Network(Expected_Win_Rate):
    def __init__(self, the_map="de_dust2"):
        super().__init__(the_map)
        
    def model_training(self, data_path=None):
        X, y = self.load_data(data_path)
        print(len(X))
        
        # Ensure all data is numerical
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_dim = X_train.shape[1]
        self.model = NeuralNet(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            y_train_prob = self.model(X_train_tensor).numpy().ravel()
            y_test_prob = self.model(X_test_tensor).numpy().ravel()

        y_train_pred = (y_train_prob > 0.5).astype(int)
        y_test_pred = (y_test_prob > 0.5).astype(int)
        
        # Evaluate the model using Log Loss
        self.log_model(y_train, y_test, y_train_prob, y_test_prob, y_train_pred, y_test_pred)
        
    def log_model(self, y_train, y_test, y_train_prob, y_test_prob, y_train_pred, y_test_pred):
        log_loss_train = log_loss(y_train, y_train_prob)
        log_loss_test = log_loss(y_test, y_test_prob)

        # Additional metrics
        accuracy_test = accuracy_score(y_test, y_test_pred)
        precision_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_prob)
        confusion_test = confusion_matrix(y_test, y_test_pred)
        
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        ss_res = ((y_test - y_test_prob) ** 2).sum()
        r2_score = 1 - (ss_res / ss_tot)

        # Print evaluation metrics
        print(f"Training Log Loss: {log_loss_train}")
        print(f"Test Log Loss: {log_loss_test}")
        print(f"Test Accuracy: {accuracy_test}")
        print(f"Test Precision: {precision_test}")
        print(f"Test Recall: {recall_test}")
        print(f"Test F1-Score: {f1_test}")
        print(f"Test ROC AUC: {roc_auc_test}")
        print(f"Test Confusion Matrix:\n{confusion_test}")
        print(f"Test R²: {r2_score}")
        
    def predict_w_model(self, X, output_prob=False):
        self.model.eval()
        if isinstance(X, list):
            X = np.array(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(X_tensor)
        
        if output_prob:
            return output.numpy().ravel()
            
        return (output.numpy().ravel() > 0.5).astype(int)
    
# Run the Mutual Information Analysis
class Mutual_Information_Analysis(Expected_Win_Rate):
    """Analysis how factors contribute to the winning of gun fights.
    
    Factors that matter:
        - Relative Distance between players
        - Player Health
        - Armor
        - Helmet
        - Flashed or not
        - Weapon Cate (pitol, smg, rifle, sniper)
        - Pitch & Yaw Diff
    """
    
    def __init__(self, the_map="de_dust2"):
        super().__init__(the_map)

    def prepare_data(self, data_path=None):
        self.X, self.y = self.load_data(data_path)
        self.X, self.y = self._data_transform()
    
    def _data_transform(self):
        """need invalid data flag
        """
        data_map = data_map_X_1
        mid_point = data_map['mid_point']
        X_data = self.X
        y_data = self.y

        X_out = []
        y_out = []
        
        err_coun = 0 
        
        for index, row in X_data.iterrows():
            row_out = []
            # 0. Get data
            p1_coord = [float(row[i]) for i in data_map['player_coordinates']]
            p2_coord = [float(row[i+mid_point]) for i in data_map['player_coordinates']]
            
            p1_weapon_list = [int(row[i]) for i in data_map["weapon"]]
            p2_weapon_list = [int(row[i+mid_point]) for i in data_map["weapon"]]
            
            # 1. Compute Relative Distance
            distance = euclidean_distance(p1_coord, p2_coord)
            if distance == 0:
                continue
            distance = distance // 500
            row_out.append(distance)
            
            # 2. Weapon -> Weapon Cate
            try:
                p1_weapon_idx = p1_weapon_list.index(1)
                p2_weapon_idx = p2_weapon_list.index(1)
            
                p1_weapon_cate_name = weapon_map_f_MI[p1_weapon_idx]
                p2_weapon_cate_name = weapon_map_f_MI[p2_weapon_idx]
                
                if p1_weapon_cate_name == 5 or p2_weapon_cate_name == 5:
                    continue # skip this weapon
                
                row_out.append(p1_weapon_cate_name)
                row_out.append(p2_weapon_cate_name) # ? Is it Okay to represent the weapon diff using 1 single value?
                
                weapon_advantage = p1_weapon_cate_name - p2_weapon_cate_name
                row_out.append(weapon_advantage)
            
            except ValueError: # Sometimes, no weapon is available, skip that now
                err_coun += 1
                continue
            
            # 3. Get Player Health (continuous)
            p1_health = [float(row[i]) for i in data_map["player_health"]]
            p2_health = [float(row[i+mid_point]) for i in data_map["player_health"]]
            health_advantage = extract_single_element(p1_health) - extract_single_element(p2_health) # ranges from -99 to 99
            row_out.append(health_advantage)
            
            # 4. Get Armor bool (bool)
            p1_armor = [float(row[i]) for i in data_map["armor_value"]]
            p2_armor = [float(row[i+mid_point]) for i in data_map["armor_value"]]
            armor_advantage = extract_single_element(p1_armor) - extract_single_element(p2_armor) # either -1 or 0 or 1
            row_out.append(armor_advantage)
            
            # 5. Helmet
            p1_helmet = [float(row[i]) for i in data_map["has_helmet"]]
            p2_helmet = [float(row[i+mid_point]) for i in data_map["has_helmet"]]
            helmet_advantage = extract_single_element(p1_helmet) - extract_single_element(p2_helmet) # either -1 or 0 or 1
            row_out.append(helmet_advantage)
            
            # 6. Flashed bool (bool)
            p1_flashed = extract_single_element([float(row[i]) for i in data_map["flash_duration"]])
            p2_flashed = extract_single_element([float(row[i+mid_point]) for i in data_map["flash_duration"]])
            if p1_flashed != 0: p1_flashed = 1 
            if p2_flashed != 0: p2_flashed = 1
            flash_advantage = p2_flashed - p1_flashed # either -1 or 0 or 1
            row_out.append(flash_advantage)
            
            # 7. Pitch & Yaw Diff
            p1_pitch_diff, p1_yaw_diff = [float(row[i]) for i in data_map["pitch_&_yaw_diff"]]
            p2_pitch_diff, p2_yaw_diff = [float(row[i+mid_point]) for i in data_map["pitch_&_yaw_diff"]]
            yaw_diff_advantage = abs(p2_yaw_diff) - abs(p1_yaw_diff) # Measures which one has the crossair futhur from target
            yaw_diff_advantage = yaw_diff_advantage // 10
            row_out.append(yaw_diff_advantage)
            
            X_out.append(row_out)
            y_out.append(int(y_data.loc[index]))
            
        
        print(f"Error Count: {err_coun}")
                    
        return X_out, y_out
    

def prepare_data(demo_num=None):
    xfight = Expected_Win_Rate()
    xfight.prepare_data(demo_num)
    
def prepare_sequence_data(demo_num=None):
    xfight = Expected_Win_Rate()
    resume_dir = r'demo_analysis\intermediate_data\model_training_sequence\17'
    # xfight.prepare_sequence_data(demo_num, resume=True, resume_dir=resume_dir, resume_idx=47)
    xfight.prepare_sequence_data()
    

def train(data_path):
    # xfight = xFight_Linear_Approximation()
    # xfight = xFight_Logistic_Regression()
    # xfight = xFight_Neural_Network()
    xfight = xFight_LSTM()
    # xfight.model_training(data_path)
    # demo_path_case = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2\g2-vs-vitality-m1-dust2.dem"
    # demo_path_case = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2\spirit-vs-aurora-m1-dust2.dem"
    xfight.model_training(data_path)
    # output = xfight.case_study(demo_path_case, 'donk')
    # return output
    


def case_study():
    demo_path_case = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2\g2-vs-vitality-m1-dust2.dem"
    # Parse the demo
    processor = Processor(demo_path_case)
    death_df = processor.query_death_df()
    
    
    
    # Identify a firing event
    
    # Predict the outcome
    
    # Visualize the scenario with features displayed
    
    pass


if __name__ == '__main__':
    # prepare_data()
    # prepare_sequence_data()
    
    # xfight = Expected_Win_Rate()
    # xfight.prepare_data(resume=True, resume_dir=r'demo_analysis\intermediate_data\model_training\19', resume_idx=6+2+1)
    
    
    # data_path = r'demo_analysis\intermediate_data\model_training\19'
    data_path = r'demo_analysis\intermediate_data\model_training_sequence\18'
    train(data_path)