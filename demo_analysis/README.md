## Packages installed
```
pip install demoparser2
```

## Python version
```
PS X:\code\CSGO_AI\demo_analysis\demo_info> python -V
Python 3.11.5
```


# Documentation
## Conversions
- Tick to time: 16 ticks = 1 second
- Teamnum: 2 = terrorist
## References
parser.parse_event("player_death") 
```
['assistedflash', 'assister_X', 'assister_Y', 'assister_name',
'assister_steamid', 'attacker_X', 'attacker_Y', 'attacker_name',
'attacker_steamid', 'attackerblind', 'distance', 'dmg_armor',
'dmg_health', 'dominated', 'headshot', 'hitgroup', 'noreplay',
'noscope', 'penetrated', 'revenge', 'thrusmoke', 'tick', 'weapon', 
'weapon_fauxitemid', 'weapon_itemid', 'weapon_originalowner_xuid', 
'wipe']
```