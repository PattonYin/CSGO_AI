from data_preperator import Processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import StringIO
from utils import *
from plot_module import *
import pickle

class Utility_Analysis(Processor):
    def __init__(self, path="demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"):
        super().__init__(path)
        
    def flash_distribution_on_map(self):
        """Computes the number and distribution of the flashes on the map 
        Constraints:
        From: Terrorists (to attack)
        Before: The bomb is planted
        
        Note: Teamnum -- 2 = terrorists

        Returns:
            _type_: _description_
        """
        
        side_length = 50 
        
        round_start_ticks, round_end_ticks = self.get_round_ticks()
        
        bomb_df = self.query_bomb_plant_info()
        bomb_plant_ticks = bomb_df['tick'].tolist()
        
        # compute ticks before bomb planted
        before_bomb_planted_ticks = [] # whichever comes first
        
        idx = 0
        for index, start_tick in enumerate(round_start_ticks):
            bomb_plant_tick = bomb_plant_ticks[idx]
            if bomb_plant_tick > start_tick and bomb_plant_tick < round_end_ticks[index]:
                before_bomb_planted_ticks += (list(range(start_tick, bomb_plant_tick+1)))
                idx += 1
            else:
                before_bomb_planted_ticks += (list(range(start_tick, round_end_ticks[index]+1)))
        
        # only keep the data in any of the ranges specified by before_bomb_planted_ticks
        flash_df = self.query_flash_info()
        print(len(before_bomb_planted_ticks))
        flash_df = flash_df[flash_df['tick'].isin(before_bomb_planted_ticks)]
        print(len(flash_df))
        
        # filter the utilities from terrorists
        flash_df = flash_df[flash_df['user_team_num'] == 2]
        
        flash_count = self.data_to_pixel_map(flash_df, side_length=side_length)
        plot_pixel_map(flash_count, side_length, title="flash_count_01")
        
    def smoke_distribution_on_map(self):
        """Computes the number and distribution of the smokes on the map 
        Constraints:
        From: Terrorists (to attack)
        Before: The bomb is planted

        Returns:
            _type_: _description_
        """
        
        side_length = 50 
        
        round_start_ticks, round_end_ticks = self.get_round_ticks()
        
        bomb_df = self.query_bomb_plant_info()
        bomb_plant_ticks = bomb_df['tick'].tolist()
        
        # compute ticks before bomb planted
        before_bomb_planted_ticks = [] # whichever comes first
        
        idx = 0
        for index, start_tick in enumerate(round_start_ticks):
            bomb_plant_tick = bomb_plant_ticks[idx]
            if bomb_plant_tick > start_tick and bomb_plant_tick < round_end_ticks[index]:
                before_bomb_planted_ticks += (list(range(start_tick, bomb_plant_tick+1)))
                idx += 1
            else:
                before_bomb_planted_ticks += (list(range(start_tick, round_end_ticks[index]+1)))
        
        # only keep the data in any of the ranges specified by before_bomb_planted_ticks
        smoke_df = self.query_smoke_info()
        print(len(before_bomb_planted_ticks))
        smoke_df = smoke_df[smoke_df['tick'].isin(before_bomb_planted_ticks)]
        print(len(smoke_df))
        
        # filter the utilities from terrorists
        smoke_df = smoke_df[smoke_df['user_team_num'] == 2]
        
        smoke_count = self.data_to_pixel_map(smoke_df, side_length=side_length)
        plot_pixel_map(smoke_count, side_length, title="smoke_count_01")
        
        
        

class Gun_Fight_Analysis(Processor):        
    def __init__(self, path="demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"):
        super().__init__(path)
    
    def win_rate_of_fights(self, datas, area_togo=None):
        """
        Computes the win rate of gunfights on the basis of location of attacker on the map
        
        How to determine a fight:
        Factors considered:
        - A player dies
        
        Factors not yet considered:
        - HP of players
        - Weapons of players
        - Blind or not
        - Smoked or not

        The unit of location:
        square of size 50x50

        Possible_nextStep:
        Use NN to learn the gunfight win rate on the continuous level

        Args:
            datas (list): list of dataframes to be processed
                the dataframe has to be the output from 
                    `parser.parse_event("player_death")`
            area_togo: e.g. ((min_x, max_x), (min_y, max_y))
        """
        side_length = 100
        fight_count = self.generate_pixel_map(side_length=side_length)
        win_count = self.generate_pixel_map(side_length=side_length)
        square_ids = fight_count.keys()
        
        for data in datas:
            for index in range(len(data)):
                if (data.iloc[index]['attackerblind'] == True 
                    or data.iloc[index]['thrusmoke'] == True):
                    continue
                
                att_true_x = data.iloc[index]['attacker_X']
                att_true_y = data.iloc[index]['attacker_Y']
                dead_true_x = data.iloc[index]['user_X']
                dead_true_y = data.iloc[index]['user_Y']
                
                if area_togo is not None:
                    
                    if att_true_x < area_togo[0][0] or att_true_x > area_togo[0][1] or att_true_y < area_togo[1][0] or att_true_y > area_togo[1][1]:
                        continue
                    if dead_true_x < area_togo[0][0] or dead_true_x > area_togo[0][1] or dead_true_y < area_togo[1][0] or dead_true_y > area_togo[1][1]:
                        continue
                    
                attacker_x = (att_true_x // side_length * side_length)
                attacker_y = (att_true_y // side_length * side_length)

                dead_x = (dead_true_x // side_length * side_length)
                dead_y = (dead_true_y // side_length * side_length)
                
                try:
                    attacker_coord = (int(attacker_x), int(attacker_y))
                    dead_coord = (int(dead_x), int(dead_y))
                except ValueError:
                    # print(attacker_x, attacker_y)
                    # print(dead_x, dead_y)
                    continue    
                
                if attacker_coord not in square_ids:
                    print(data.iloc[index]['attacker_X'])
                    print(data.iloc[index]['attacker_Y'])
                
                win_count[attacker_coord] += 1
                fight_count[attacker_coord] += 1
                fight_count[dead_coord] += 1
                
        win_rate = {}
        
        for square_id in square_ids:
            if fight_count[square_id] == 0:
                win_rate[square_id] = None
            else:
                win_rate[square_id] = win_count[square_id]/fight_count[square_id]
                
        return win_rate
    
    def num_death_on_map(self, datas, area_togo=None):
        side_length = 100
        
        dead_count = self.generate_pixel_map(side_length=side_length)
        square_ids = dead_count.keys()
        
        for data in datas:
            for index in range(len(data)):
                # if (data.iloc[index]['attackerblind'] == True 
                #     or data.iloc[index]['thrusmoke'] == True):
                #     continue
                dead_true_x = data.iloc[index]['user_X']
                dead_true_y = data.iloc[index]['user_Y']
                
                if area_togo is not None:                    
                    if dead_true_x < area_togo[0][0] or dead_true_x > area_togo[0][1] or dead_true_y < area_togo[1][0] or dead_true_y > area_togo[1][1]:
                        continue

                dead_x = (dead_true_x // side_length * side_length)
                dead_y = (dead_true_y // side_length * side_length)
                
                try:
                    dead_coord = (int(dead_x), int(dead_y))
                except ValueError:
                    continue                   

                dead_count[dead_coord] += 1
                
        return dead_count
    
    def num_death_att_entropy(self, datas, area_togo=None):
        """
        att_position_freq: maps coordinates to another dictionary, which records the number of attacks from each position
        """
        side_length = 100
        
        att_position_freq = self.generate_pixel_map_to_pixel_map(side_length=side_length)
        att_position_entropy = self.generate_pixel_map(side_length=side_length)
        square_ids = att_position_freq.keys()
        
        for data in datas:
            for index in range(len(data)):

                dead_true_x = data.iloc[index]['user_X']
                dead_true_y = data.iloc[index]['user_Y']

                att_true_x = data.iloc[index]['attacker_X']
                att_true_y = data.iloc[index]['attacker_Y']
                
                if area_togo is not None:                    
                    if dead_true_x < area_togo[0][0] or dead_true_x > area_togo[0][1] or dead_true_y < area_togo[1][0] or dead_true_y > area_togo[1][1]:
                        continue

                dead_x = (dead_true_x // side_length * side_length)
                dead_y = (dead_true_y // side_length * side_length)
                
                att_x = (att_true_x // side_length * side_length)
                att_y = (att_true_y // side_length * side_length)
                
                try:
                    dead_coord = (int(dead_x), int(dead_y))
                    att_coord = (int(att_x), int(att_y))
                except ValueError:
                    continue                   
                
                # update the map in the corresponding coordinate
                dict_to_update = att_position_freq[dead_coord]
                dict_to_update[att_coord] = dict_to_update.get(att_coord, 0) + 1
                att_position_freq[dead_coord] = dict_to_update
                
        # compute entropy basing on the attacking coordinates and 
        for square_id in square_ids:
            att_position_entropy = compute_entropy(att_position_freq[square_id])
        
        return att_position_entropy
    
class Aim_Analysis(Processor):
    tick_range = 100 # 100 ticks before the death event
    props_to_query = ["pitch", "yaw", "X", "Y", "Z"]         

    def get_a_death_event(self, data, idx):
        attacker_name = data.iloc[idx]['attacker_name']
        attackee_name = data.iloc[idx]['user_name']
        tick_sample = data.iloc[idx]['tick']
        weapon_name = data.iloc[idx]['weapon']
        return attacker_name, attackee_name, tick_sample, weapon_name
    
    def get_fire_ticks(self, names, tick_sample, tick_range):
        fire_df = self.query_weapon_fire_df()
        fire_df = fire_df[(fire_df["tick"] >= tick_sample - tick_range) & (fire_df["tick"] <= tick_sample)]
        if isinstance(names, list):
            output = []
            for name in names:
                user_fire_df = fire_df[fire_df['user_name'] == f'{name}']
                user_fire_ticks = user_fire_df.tick.tolist()
                output.append(user_fire_ticks)
        elif isinstance(names, str):
            user_fire_df = fire_df[fire_df['user_name'] == f'{names}']
            user_fire_ticks = user_fire_df.tick.tolist()
            output = user_fire_ticks
        else:
            raise ValueError("names should be either a string or a list of strings")
        return output
        
    def get_pitch_yaw_ticks(self, names, tick_sample, tick_range):
        output = []
        ticks = self.get_ticks(tick_sample-tick_range, tick_sample, self.props_to_query)
        for name in names:
            user_ticks = ticks[ticks['name'] == f'{name}']
            user_yaw_list = user_ticks.yaw.tolist()
            user_pitch_list = user_ticks.pitch.tolist()
            output.append((user_yaw_list, user_pitch_list))
        return output  
    
    def get_pitch_yaw_XYZ(self, name, tick_sample, tick_range):
        ticks = self.get_ticks(tick_sample-tick_range, tick_sample, self.props_to_query)
        user_ticks = ticks[ticks['name'] == f'{name}']
        user_yaw_list = user_ticks.yaw.tolist()
        user_pitch_list = user_ticks.pitch.tolist()
        user_X_list = user_ticks.X.tolist()
        user_Y_list = user_ticks.Y.tolist()
        user_Z_list = user_ticks.Z.tolist()
        return user_pitch_list, user_yaw_list, user_X_list, user_Y_list, user_Z_list
    
    def compute_pitch_yaw_diff(self, attacker_name, attackee_name, tick_sample):
        """Compute ho far the crosshair is from the enemy given the tick of death and attacker, attackee names.

        Args:
            attacker_name (_type_): _description_
            attackee_name (_type_): _description_
            tick_sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            start_tick = tick_sample-self.tick_range
        
            attacker_fire_ticks = self.get_fire_ticks(attacker_name, tick_sample, self.tick_range)
        
            first_tick = attacker_fire_ticks[0]
            
            attacker_pitch_list, attacker_yaw_list, attacker_X_list, attacker_Y_list, attacker_Z_list = self.get_pitch_yaw_XYZ(attacker_name, tick_sample, self.tick_range)
            attackee_pitch_list, attackee_yaw_list, attackee_X_list, attackee_Y_list, attackee_Z_list = self.get_pitch_yaw_XYZ(attackee_name, tick_sample, self.tick_range)

            attacker_pitchs_diff = []
            attacker_yaws_diff = []
            
            for i in range(first_tick-start_tick-1, self.tick_range):
                attacker_pitch_diff, attacker_yaw_diff = pitch_yaw_diff_from_enemy(attacker_pitch_list[i], attacker_yaw_list[i], attacker_X_list[i], attacker_Y_list[i], attacker_Z_list[i], attackee_X_list[i], attackee_Y_list[i], attackee_Z_list[i])
                
                attacker_pitchs_diff.append(attacker_pitch_diff)
                attacker_yaws_diff.append(attacker_yaw_diff)
            
            return attacker_pitchs_diff, attacker_yaws_diff, attacker_fire_ticks
    
        except IndexError:
            print(f"attacker_name: {attacker_name}, attackee_name: {attackee_name}, tick_sample: {tick_sample}")
            
            return None, None, None
    
    def highlight_ticks(self, ax, data_list, fire_ticks, title, lim=None, fix_lim=True):
        if lim is not None:
            ax.set_ylim(lim)
        else:
            if fix_lim == True: 
                min_y = min(data_list)
                ax.set_ylim(min_y-0.2, min_y+3)
        ax.plot(data_list)
        
        first_tick = fire_ticks[0]
        xs_togo = [tick - first_tick for tick in fire_ticks]
        ys_togo = [data_list[tick - first_tick] for tick in fire_ticks]
        ax.scatter(xs_togo, ys_togo, color='red', s=40)
        ax.set_title(title)
                        
class Aim_Analysis_Group(Aim_Analysis):
    tick_range = 100 # 100 ticks before the death event
    props_to_query = ["pitch", "yaw", "X", "Y", "Z"]   
    yaw_lim = (-15, 15)
    pitch_lim = (-15, 15)
    weapons_main = {'glock': 0, 'usp_silencer': 1, 'deagle': 2, 'ak47': 3, 'm4a1':4 ,'m4a1_silencer':5, 'awp':6}
    
    def __init__(self, path="demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"):
        super().__init__(path)
        self.death_df = self.query_death_df()
        self.players = self.death_df['user_name'].unique().tolist()
    
    def first_shot_accuracy(self):
        """This function quantitatively measures how far the crosshair is from the target
        """
        accuracy_dict = {player: [] for player in self.players}
        for player in tqdm(self.players):
            death_df_player = self.death_df[self.death_df['attacker_name'] == player]
            for index, row in death_df_player.iterrows():
                attackee_name = row['user_name']
                tick_sample = row['tick']
                attacker_pitchs_diff, attacker_yaws_diff, attacker_fire_ticks = self.compute_pitch_yaw_diff(player, attackee_name, tick_sample)
                if attacker_pitchs_diff is None:
                    continue
                diff_tuple = (attacker_pitchs_diff[0], attacker_yaws_diff[0])
                accuracy_dict[player].append(diff_tuple)
        
        # Use pickle to save the dict
        with open('demo_analysis/intermediate_data/accuracy_dict_sample.pickle', 'wb') as handle:
            pickle.dump(accuracy_dict, handle)
        
        self.first_shot_acc_freq_plot(accuracy_dict)
        
    def first_shot_acc_freq_plot(self, accuracy_dict={}, load_path=None):
        if load_path is not None:
            with open(load_path, 'rb') as handle:
                accuracy_dict = pickle.load(handle)
                
        # clip the yaw vals > 8 or < -8
        for player in accuracy_dict.keys():
            accuracy_dict[player] = [item for item in accuracy_dict[player] if abs(item[1]) < 8]    
                        
        # visualize the dict for each player, 10 in total
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        bin_size = 1
        bin_edges = np.arange(-8, 8+bin_size, bin_size)
        
        for i, player in enumerate(accuracy_dict.keys()):
            ax = axes[i]
            # freq plots for the data
            ax.hist([item[1] for item in accuracy_dict[player]], bins=bin_edges, edgecolor='teal', alpha=0.7)
            ax.set_xlabel("diff_yaw")
            ax.set_ylabel("Frequency")
            ax.set_title(player)
            ax.set_ylim(0, 15)
            ax.set_xlim(-8, 8)
        
        plt.tight_layout()
        plt.show()
            
    # Visualization
    def aim_accuracy_individual_plot(self):
        death_df = self.query_death_df()
        players = death_df['user_name'].unique().tolist()
        
        df_weapons = [death_df[death_df['weapon'] == weapon] for weapon in self.weapons_main.keys()]        
        
        # Investigate the aim_acc for ak47
        weapon_ticks = [df_weapon['tick'].tolist() for df_weapon in df_weapons]
        
        ak_df = df_weapons[self.weapons_main['ak47']]
        weapon_todo = 'ak47'
        
        for player in players[:5]:
            # Step 1: get AK df for each player
            # Step 2: get the pitch, yaw, X, Y, Z for each player for each fight
            # Step 3: plot
            ak_player_df = ak_df[ak_df['attacker_name'] == player]
            ak_player_ticks = ak_player_df.tick.tolist()
            attackee_names = ak_player_df.user_name.tolist()
            
            player_all_pitchs_diff = []
            player_all_yaws_diff = []
            attacker_fire_ticks_list = []
            
            for index, tick_sample in enumerate(ak_player_ticks):
                attacker_pitchs_diff, attacker_yaws_diff, attacker_fire_ticks = self.compute_pitch_yaw_diff(player, attackee_names[index], tick_sample)
                player_all_pitchs_diff.append(attacker_pitchs_diff)
                player_all_yaws_diff.append(attacker_yaws_diff)
                attacker_fire_ticks_list.append(attacker_fire_ticks)
                
            num_plots = 6
            ax_len = 3
                
            fig, axes = plt.subplots(2, num_plots, figsize=(num_plots*ax_len, ax_len*2))
                        
            itrs = min(num_plots, len(player_all_pitchs_diff))
            for i in range(itrs):
                self.highlight_ticks(axes[0, i], player_all_pitchs_diff[i], attacker_fire_ticks_list[i], f"{player}_pitch_diff, {weapon_todo}", lim=self.pitch_lim)
                self.highlight_ticks(axes[1, i], player_all_yaws_diff[i], attacker_fire_ticks_list[i], f"{player}_yaw_diff, {weapon_todo}", lim=self.yaw_lim)
                
            for j in range(itrs, num_plots):
                fig.delaxes(axes[0, j])
                fig.delaxes(axes[1, j])
                
            plt.tight_layout()
            plt.show()
                
                
class Aim_Analysis_Case_Study(Aim_Analysis):
    # Constants
    sample_idx = 21
    tick_range = 100 # 100 ticks before the death event
    dyaw_lim = (-500, 500)
    dpitch_lim = (-1, 1)
    props_to_query = ["pitch", "yaw", "X", "Y", "Z"]         
        
    def aim_speed(self):
        dval = True
        data = self.query_death_df()
        
        attacker_name, attackee_name, tick_sample, weapon = self.get_a_death_event(data, self.sample_idx)
            
        attacker_fire_ticks, attackee_fire_ticks = self.get_fire_ticks([attacker_name, attackee_name], tick_sample, self.tick_range)
                
        (attacker_yaw_list, attacker_pitch_list), (attackee_yaw_list, attackee_pitch_list) = self.get_pitch_yaw_ticks([attacker_name, attackee_name], tick_sample, self.tick_range)
        
        attacker_dyaw_list = np.diff(attacker_yaw_list)
        attacker_dpitch_list = np.diff(attacker_pitch_list)
        attackee_dyaw_list = np.diff(attackee_yaw_list)
        attackee_dpitch_list = np.diff(attackee_pitch_list)

        # Plot out the dyaw/dt and dpitch/dt
        # Highlight fire ticks
        def highlight_ticks(ax, data_list, fire_ticks, title, lim=None):
            if lim is not None:
                ax.set_ylim(lim)
            ax.plot(data_list)
            ax.scatter([i+100-tick_sample for i in fire_ticks], [data_list[i+100-tick_sample] for i in fire_ticks], color='red', s=40)
            ax.set_title(title)
        
        if dval:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            highlight_ticks(ax[0, 0], attacker_dyaw_list, attacker_fire_ticks, "attacker_dyaw_list", self.dyaw_lim)
            highlight_ticks(ax[0, 1], attacker_dpitch_list, attacker_fire_ticks, "attacker_dpitch_list", self.dpitch_lim)
            highlight_ticks(ax[1, 0], attackee_dyaw_list, attackee_fire_ticks, "attackee_dyaw_list", self.dyaw_lim)
            highlight_ticks(ax[1, 1], attackee_dpitch_list, attackee_fire_ticks, "attackee_dpitch_list", self.dpitch_lim)
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            highlight_ticks(ax[0, 0], attacker_yaw_list, attacker_fire_ticks, "attacker_yaw_list")
            highlight_ticks(ax[0, 1], attacker_pitch_list, attacker_fire_ticks, "attacker_pitch_list")
            highlight_ticks(ax[1, 0], attackee_yaw_list, attackee_fire_ticks, "attackee_yaw_list")
            highlight_ticks(ax[1, 1], attackee_pitch_list, attackee_fire_ticks, "attackee_pitch_list")
            plt.tight_layout()
            plt.show()
            
    def aim_accuracy(self):
        data = self.query_death_df()
        
        attacker_name, attackee_name, tick_sample, weapon = self.get_a_death_event(data, self.sample_idx)
        start_tick = tick_sample-self.tick_range
            
        attacker_fire_ticks, attackee_fire_ticks = self.get_fire_ticks([attacker_name, attackee_name], tick_sample, self.tick_range)
        
        attacker_pitch_list, attacker_yaw_list, attacker_X_list, attacker_Y_list, attacker_Z_list = self.get_pitch_yaw_XYZ(attacker_name, tick_sample, self.tick_range) 
        attackee_pitch_list, attackee_yaw_list, attackee_X_list, attackee_Y_list, attackee_Z_list = self.get_pitch_yaw_XYZ(attackee_name, tick_sample, self.tick_range)
        
        first_fire_tick = min(attacker_fire_ticks[0], attackee_fire_ticks[0])
        print(first_fire_tick, start_tick)
        first_tick = max(first_fire_tick - 10, start_tick)
        print(first_tick)
            
        attacker_pitchs_diff = []
        attacker_yaws_diff = []
        attackee_pitchs_diff = []
        attackee_yaws_diff = []
        
        for i in range(first_tick-start_tick, self.tick_range):
            attacker_pitch_diff, attacker_yaw_diff = pitch_yaw_diff_from_enemy(attacker_pitch_list[i], attacker_yaw_list[i], attacker_X_list[i], attacker_Y_list[i], attacker_Z_list[i], attackee_X_list[i], attackee_Y_list[i], attackee_Z_list[i])
            
            attackee_pitch_diff, attackee_yaw_diff = pitch_yaw_diff_from_enemy(attackee_pitch_list[i], attackee_yaw_list[i], attackee_X_list[i], attackee_Y_list[i], attackee_Z_list[i], attacker_X_list[i], attacker_Y_list[i], attacker_Z_list[i])
            
            attacker_pitchs_diff.append(attacker_pitch_diff)
            attacker_yaws_diff.append(attacker_yaw_diff)
            attackee_pitchs_diff.append(attackee_pitch_diff)
            attackee_yaws_diff.append(attackee_yaw_diff)            

        # Highlight fire ticks
        def highlight_ticks(ax, data_list, fire_ticks, title, lim=None, fix_lim=True):
            if lim is not None:
                ax.set_ylim(lim)
            if fix_lim == True: 
                min_y = min(data_list)
                ax.set_ylim(min_y-0.2, min_y+3)
                
            ax.plot(data_list)
            xs_togo = [tick - first_tick - 1 for tick in fire_ticks]
            ys_togo = [data_list[tick - first_tick - 1] for tick in fire_ticks]
            ax.scatter(xs_togo, ys_togo, color='red', s=40)
            ax.set_title(title)
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        highlight_ticks(ax[0, 0], attacker_yaws_diff, attacker_fire_ticks, f"attacker_yaw_diff, {weapon}")
        highlight_ticks(ax[0, 1], attacker_pitchs_diff, attacker_fire_ticks, f"attacker_pitch_diff, {weapon}")
        highlight_ticks(ax[1, 0], attackee_yaws_diff, attackee_fire_ticks, "attackee_yaw_diff")
        highlight_ticks(ax[1, 1], attackee_pitchs_diff, attackee_fire_ticks, "attackee_pitch_diff")
        plt.tight_layout()
        plt.show()
        
    def aim_accuracy_individual(self):
        data = self.query_death_df()
        
        attacker_name, attackee_name, tick_sample, weapon = self.get_a_death_event(data, self.sample_idx)
        start_tick = tick_sample-self.tick_range
            
        attacker_fire_ticks, attackee_fire_ticks = self.get_fire_ticks([attacker_name, attackee_name], tick_sample, self.tick_range)
        
        attacker_pitch_list, attacker_yaw_list, attacker_X_list, attacker_Y_list, attacker_Z_list = self.get_pitch_yaw_XYZ(attacker_name, tick_sample, self.tick_range) 
        attackee_pitch_list, attackee_yaw_list, attackee_X_list, attackee_Y_list, attackee_Z_list = self.get_pitch_yaw_XYZ(attackee_name, tick_sample, self.tick_range)
        
        first_tick = attacker_fire_ticks[0]
        
        attacker_pitchs_diff = []
        attacker_yaws_diff = []
        
        for i in range(first_tick-start_tick-1, self.tick_range):
            attacker_pitch_diff, attacker_yaw_diff = pitch_yaw_diff_from_enemy(attacker_pitch_list[i], attacker_yaw_list[i], attacker_X_list[i], attacker_Y_list[i], attacker_Z_list[i], attackee_X_list[i], attackee_Y_list[i], attackee_Z_list[i])
            
            attacker_pitchs_diff.append(attacker_pitch_diff)
            attacker_yaws_diff.append(attacker_yaw_diff)
 

        # Highlight fire ticks
        def highlight_ticks(ax, data_list, fire_ticks, title, lim=None, fix_lim=True):
            if lim is not None:
                ax.set_ylim(lim)
            if fix_lim == True: 
                min_y = min(data_list)
                ax.set_ylim(min_y-0.2, min_y+3)
                
            ax.plot(data_list)
            xs_togo = [tick - first_tick for tick in fire_ticks]
            ys_togo = [data_list[tick - first_tick] for tick in fire_ticks]
            ax.scatter(xs_togo, ys_togo, color='red', s=40)
            ax.set_title(title)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        highlight_ticks(ax[0], attacker_yaws_diff, attacker_fire_ticks, f"attacker_yaw_diff, {weapon}")
        highlight_ticks(ax[1], attacker_pitchs_diff, attacker_fire_ticks, f"attacker_pitch_diff, {weapon}")
        plt.tight_layout()
        plt.show()        
        
