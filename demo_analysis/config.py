max_x = 1787.9722900390625
min_x = -2185.96875
max_y = 3098.52978515625
min_y = -997.5834350585938

max_x = 1900
min_x = -2200
max_y = 3200
min_y = -1200

side_length = 50

tick_rate = 64

data_map_X_1 = {
    "player_coordinates": list(range(0, 3)),
    "teammate_coordinates": list(range(3, 15)),
    "player_health": [15],
    "armor_value": [16],
    "has_helmet": [17],
    "pitch_&_yaw": list(range(18,20)),
    "flash_duration": [20],
    "weapon": list(range(21, 58)),
    "pitch_&_yaw_diff": list(range(58, 60)),
    
    "mid_point": 60,
}

weapon_categories = [
    'FAMAS','P90','SSG 08','M4A4','Tec-9','Desert Eagle','AWP','USP-S','MAC-10','Zeus x27','XM1014','knife','P2000','Galil AR','SG 553','MP9','Smoke Grenade','Decoy Grenade','AK-47','UMP-45','P250','Incendiary Grenade','Flashbang','Five-SeveN','High Explosive ','0','M4A1-S','MAG-7','Nova','Negev','C4','AUG','MP7','Molotov','CZ75-Auto','Glock-18','Dual Berettas'
]   


weapon_meta_cate_map_1 = {
    1: [4,5,7,12,20,23,34,35,36], # pistol
    2: [1,8,15,19,32], # smg
    3: [0,3,13,14,18,26,31], # rifle
    4: [6], # sniper
    5: [2,9,10,11,16,17,21,22,24,25,27,28,29,30,33], # not considered
}

weapon_map_f_MI = dict()
for cate_name, index_list in weapon_meta_cate_map_1.items():
    for i in index_list:
        weapon_map_f_MI[i] = cate_name    

# weapon_map_f_MI = {
#     "pistol": [4,5,7,12,20,23,34,35,36],
#     "smg": [1,8,15,19,32],
#     "rifle": [0,3,13,14,18,26,31],
#     "sniper": [6],
#     "not considered": [2,9,10,11,16,17,21,22,24,25,27,28,29,30,33],
# }