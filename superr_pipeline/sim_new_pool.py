import polars as pl
from pathlib import Path
import json

# 1. Load mapping and build baseline exclusions
mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
pak_ids = set(mp.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().is_in(['pakistan', 'pakistan a']))['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids = set(mp.filter(pl.col('comp_name').cast(pl.Utf8) == 'IPL 2026')['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids.add(6628) # Gerald Coetzee

# Load 76 released IDs
from match_released_teams import all_matched_ids
released_set = set(all_matched_ids)
print(f"Total IPL 2026 IDs before release: {len(ipl26_ids)}")
print(f"Total released IDs to remove from exclusion: {len(released_set)}")

retained_ipl_ids = ipl26_ids - released_set
print(f"Retained IPL 2026 IDs (stay excluded): {len(retained_ipl_ids)}")

final_excluded_canonical = pak_ids | retained_ipl_ids
excluded = {f"PID_{int(x)}" for x in final_excluded_canonical}

# Add Pakistan registry
# load pak registry from 05_empirical_bayes or identity
# For safety, load pak_registry if available
t20s_pak = set()
try:
    with open('pipeline_outputs/tables/pak_registry_ids.json', 'r') as f:
        t20s_pak = set(json.load(f))
except Exception:
    pass
for rid in t20s_pak:
    excluded.add(f"CS_{rid}")

# Retired players
retired = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237'}
excluded.update(retired)

print(f"Final exclusion set size: {len(excluded)}")

# 2. Check how many batters and bowlers meet floors
b_diag = pl.read_csv('pipeline_outputs/tables/batter_all_player_recentered_eb_diagnostics.csv')
w_diag = pl.read_csv('pipeline_outputs/tables/bowler_all_player_recentered_eb_diagnostics.csv')

# Check floors: >= 100 balls, >= 12 matches
b_sel = b_diag.filter(
    (pl.col('pressure_balls') >= 100) &
    (pl.col('matches_with_pressure_balls') >= 12) &
    (~pl.col('player_id').is_in(excluded))
)

w_sel = w_diag.filter(
    (pl.col('pressure_balls') >= 100) &
    (pl.col('matches_with_pressure_balls') >= 12) &
    (~pl.col('player_id').is_in(excluded))
)

print(f"\nEligible Batters (>=100b, >=12m): {len(b_sel)}")
print(f"Eligible Bowlers (>=100b, >=12m): {len(w_sel)}")

# Check which released players entered eligible pool
b_rel_in = [r for r in b_sel.iter_rows(named=True) if int(r['player_id'].replace('PID_', '')) in released_set if r['player_id'].startswith('PID_')]
w_rel_in = [r for r in w_sel.iter_rows(named=True) if int(r['player_id'].replace('PID_', '')) in released_set if r['player_id'].startswith('PID_')]

print(f"\nReleased Batters qualifying for primary floor: {len(b_rel_in)}")
for r in b_rel_in:
    print(f"  Batter: {r['player_name']} ({r['player_id']}) | balls={r['pressure_balls']} | matches={r['matches_with_pressure_balls']} | raw_p={r['raw_pressure_value_per_ball']:.3f} | p_dev={r['recentered_shrunk_pressure_deviation']:.3f}")

print(f"\nReleased Bowlers qualifying for primary floor: {len(w_rel_in)}")
for r in w_rel_in:
    print(f"  Bowler: {r['player_name']} ({r['player_id']}) | balls={r['pressure_balls']} | matches={r['matches_with_pressure_balls']} | raw_p={r['raw_pressure_value_per_ball']:.3f} | p_dev={r['recentered_shrunk_pressure_deviation']:.3f}")
