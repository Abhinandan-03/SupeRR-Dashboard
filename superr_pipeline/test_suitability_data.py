import polars as pl
import numpy as np
import json
from pathlib import Path

# Load data base
base = pl.read_parquet('pipeline_outputs/modeling/combined_delivery_base.parquet')
bbb = pl.read_csv('inputs/data_pack/Data Pack/ball_by_ball.csv', columns=['striker_id', 'bowler_id', 'batting_type', 'bowling_type', 'batting_order'])

# Load squads for nationality
squads = pl.read_csv('inputs/squads.csv')
squads_nat = dict(zip(squads['player'].str.to_lowercase().to_list(), squads['nationality'].to_list()))

# Load mapping for nationality
mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
indian_teams = {'India', 'India A', 'Chennai Super Kings', 'Mumbai Indians', 'Delhi Capitals', 'Lucknow Super Giants', 'Punjab Kings', 'Rajasthan Royals', 'Sunrisers Hyderabad', 'Royal Challengers Bengaluru', 'Gujarat Titans', 'Kolkata Knight Riders'}
player_nat_map = {}
for r in mp.iter_rows(named=True):
    pid = r['player_id']
    tm = r['team_name']
    if pid not in player_nat_map:
        if tm in ['Pakistan', 'Pakistan A']:
            player_nat_map[pid] = 'Pakistan'
        elif tm in ['India', 'India A']:
            player_nat_map[pid] = 'Indian'
        elif tm in ['Australia', 'England', 'South Africa', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan', 'Ireland', 'Zimbabwe', 'Netherlands', 'Scotland', 'Namibia', 'Nepal', 'Oman', 'UAE', 'USA']:
            player_nat_map[pid] = 'Overseas'

# Add squads nationality
for r in mp.filter(pl.col('comp_name') == 'IPL 2026').iter_rows(named=True):
    pid = r['player_id']
    pname = r['player_name'].lower()
    if pname in squads_nat:
        player_nat_map[pid] = squads_nat[pname]

# Helper for nationality
def get_nationality(pid_str):
    if pid_str.startswith('PID_'):
        pid_num = int(pid_str.replace('PID_', ''))
        nat = player_nat_map.get(pid_num)
        if nat: return nat
        return 'Indian / Domestic' # fallback for PID
    return 'Unknown' # for CS_

# Extract batting profiles
# Compute batting order across base matches
print("Building ball-level batting order map...")
# For original data, use bbb batting order
bbb_order_map = {}
for r in bbb.filter(pl.col('striker_id').is_not_null() & pl.col('batting_order').is_not_null()).iter_rows(named=True):
    pid = f"PID_{r['striker_id']}"
    if pid not in bbb_order_map:
        bbb_order_map[pid] = []
    bbb_order_map[pid].append(r['batting_order'])

# Compute modal batting type and bowling type for PID
pid_bat_type = {}
for r in bbb.filter(pl.col('striker_id').is_not_null() & pl.col('batting_type').is_not_null()).iter_rows(named=True):
    pid = f"PID_{r['striker_id']}"
    if pid not in pid_bat_type:
        pid_bat_type[pid] = r['batting_type']

pid_bowl_type = {}
for r in bbb.filter(pl.col('bowler_id').is_not_null() & pl.col('bowling_type').is_not_null()).iter_rows(named=True):
    pid = f"PID_{r['bowler_id']}"
    if pid not in pid_bowl_type:
        pid_bowl_type[pid] = r['bowling_type']

# Bowler phases from base
bowler_phases = base.filter(pl.col('legal_ball') == True).group_by('bowler_player_id').agg([
    pl.len().alias('legal_balls_bowled'),
    (pl.col('over') < 6).mean().alias('powerplay_overs_1_6_share'),
    ((pl.col('over') >= 6) & (pl.col('over') < 16)).mean().alias('middle_overs_7_16_share'),
    (pl.col('over') >= 16).mean().alias('death_overs_17_20_share'),
    pl.col('pace_or_spin').mode().first().alias('base_pace_or_spin')
])

bowler_phases_dict = {r['bowler_player_id']: r for r in bowler_phases.to_dicts()}

# Batter power from base
batter_power = base.filter(pl.col('batter_faced') == True).group_by('striker_player_id').agg([
    pl.len().alias('balls_faced'),
    (pl.col('over') >= 16).mean().alias('death_balls_faced_share'),
    (pl.col('batter_runs') >= 4.0).mean().alias('boundary_rate')
])

batter_power_dict = {r['striker_player_id']: r for r in batter_power.to_dicts()}

print(f"Extracted bowler profiles: {len(bowler_phases_dict)}, batter profiles: {len(batter_power_dict)}")
