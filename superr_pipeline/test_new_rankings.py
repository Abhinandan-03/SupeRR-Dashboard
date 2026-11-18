import polars as pl
import numpy as np

# Load EB diagnostics
b_diag = pl.read_csv('pipeline_outputs/tables/batter_all_player_recentered_eb_diagnostics.csv')
w_diag = pl.read_csv('pipeline_outputs/tables/bowler_all_player_recentered_eb_diagnostics.csv')

# Load final exclusions
rel_df = pl.read_csv('pipeline_outputs/config/released_players_mapping.csv')
released_ids = set(rel_df['formatted_id'].to_list())

mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
pak_ids = set(mp.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().is_in(['pakistan', 'pakistan a']))['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids = set(mp.filter(pl.col('comp_name').cast(pl.Utf8) == 'IPL 2026')['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids.add(6628)
excluded_pids = {f"PID_{int(x)}" for x in pak_ids | ipl26_ids}
retired = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237'}
excluded_pids.update(retired)

final_excluded = excluded_pids - released_ids

# Filter eligible
b_sel = b_diag.filter(
    (pl.col('pressure_balls') >= 100) &
    (pl.col('matches_with_pressure_balls') >= 12) &
    (~pl.col('player_id').is_in(final_excluded))
).sort('rating', descending=True)

w_sel = w_diag.filter(
    (pl.col('pressure_balls') >= 100) &
    (pl.col('matches_with_pressure_balls') >= 12) &
    (~pl.col('player_id').is_in(final_excluded))
).sort('rating', descending=True)

print(f"Eligible Batters: {len(b_sel)}")
print("Performance Top 10 Batters (before bootstrap/near-tie):")
for r in b_sel.head(10).iter_rows(named=True):
    print(f"  {r['player_name']:<20} ({r['player_id']}) | rating={r['rating']:.3f} | balls={r['pressure_balls']}")

print(f"\nEligible Bowlers: {len(w_sel)}")
print("Performance Top 10 Bowlers (before bootstrap/near-tie):")
for r in w_sel.head(10).iter_rows(named=True):
    print(f"  {r['player_name']:<20} ({r['player_id']}) | rating={r['rating']:.3f} | balls={r['pressure_balls']}")
