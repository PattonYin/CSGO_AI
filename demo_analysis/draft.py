def grab_X_info_seq(start_tick, end_tick, df, name_player):
    tick_interval = 10
    X_info_output = []
    df_new = df.drop_duplicates()
    start_tick_idx = df_new.index.get_loc(df_new.loc[df_new['tick'] == start_tick].index[0])
    end_tick_idx = df_new.index.get_loc(df_new.loc[df_new['tick'] == end_tick].index[0])
    
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