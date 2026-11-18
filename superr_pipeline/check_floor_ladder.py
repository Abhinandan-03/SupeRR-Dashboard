import polars as pl
from pathlib import Path
import json

# Load released IDs
rel_df = pl.read_csv('pipeline_outputs/config/released_players_mapping.csv')
released_ids = set(rel_df['formatted_id'].to_list())

# Load baseline exclusions
mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
pak_ids = set(mp.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().is_in(['pakistan', 'pakistan a']))['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids = set(mp.filter(pl.col('comp_name').cast(pl.Utf8) == 'IPL 2026')['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids.add(6628) # Coetzee

excluded_pids = {f"PID_{int(x)}" for x in pak_ids | ipl26_ids}

# Load Pakistan registry
t20s_pak = set()
try:
    with open('pipeline_outputs/tables/pak_registry_ids.json', 'r') as f:
        t20s_pak = set(json.load(f))
except Exception:
    pass
for rid in t20s_pak:
    excluded_pids.add(f"CS_{rid}")

# Retired players
retired = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237'}
excluded_pids.update(retired)

print(f"Total exclusion set before removing released: {len(excluded_pids)}")

# Remove released players
final_excluded = excluded_pids - released_ids
print(f"Final exclusion set after removing released: {len(final_excluded)}")

# Check eligible players under floor ladder
b_diag = pl.read_csv('pipeline_outputs/tables/batter_all_player_recentered_eb_diagnostics.csv')
w_diag = pl.read_csv('pipeline_outputs/tables/bowler_all_player_recentered_eb_diagnostics.csv')

for role, df_d in [('batter', b_diag), ('bowler', w_diag)]:
    unexcl = df_d.filter(~pl.col('player_id').is_in(final_excluded))
    print(f"\nRole {role}: total unexcluded in data = {len(unexcl)}")
    
    # Primary floor: >= 100 balls, >= 12 matches
    f100 = unexcl.filter((pl.col('pressure_balls') >= 100) & (pl.col('matches_with_pressure_balls') >= 12))
    print(f"  Floor 100/12: {len(f100)} eligible")
    
    # Fallback 80/10
    f80 = unexcl.filter((pl.col('pressure_balls') >= 80) & (pl.col('matches_with_pressure_balls') >= 10))
    print(f"  Floor 80/10:  {len(f80)} eligible")
    
    # Fallback 60/8
    f60 = unexcl.filter((pl.col('pressure_balls') >= 60) & (pl.col('matches_with_pressure_balls') >= 8))
    print(f"  Floor 60/8:   {len(f60)} eligible")
