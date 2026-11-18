import polars as pl

rel_df = pl.read_csv('pipeline_outputs/config/released_players_mapping.csv')
released_ids = set(rel_df['formatted_id'].to_list())

b_diag = pl.read_csv('pipeline_outputs/tables/batter_all_player_recentered_eb_diagnostics.csv')
w_diag = pl.read_csv('pipeline_outputs/tables/bowler_all_player_recentered_eb_diagnostics.csv')

# Load final excluded
import json
mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
pak_ids = set(mp.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().is_in(['pakistan', 'pakistan a']))['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids = set(mp.filter(pl.col('comp_name').cast(pl.Utf8) == 'IPL 2026')['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids.add(6628)
excluded_pids = {f"PID_{int(x)}" for x in pak_ids | ipl26_ids}
retired = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237'}
excluded_pids.update(retired)
final_excluded = excluded_pids - released_ids

b_sel = b_diag.filter((pl.col('pressure_balls') >= 100) & (pl.col('matches_with_pressure_balls') >= 12) & (~pl.col('player_id').is_in(final_excluded)))
w_sel = w_diag.filter((pl.col('pressure_balls') >= 100) & (pl.col('matches_with_pressure_balls') >= 12) & (~pl.col('player_id').is_in(final_excluded)))

b_rel = b_sel.filter(pl.col('player_id').is_in(released_ids))
w_rel = w_sel.filter(pl.col('player_id').is_in(released_ids))

print(f"Released Batters qualifying (>=100b, >=12m): {len(b_rel)}")
for r in b_rel.iter_rows(named=True):
    print(f"  {r['player_name']:<22} ({r['player_id']}) | balls={r['pressure_balls']:<5} | matches={r['matches_with_pressure_balls']:<3} | p_dev={r['recentered_shrunk_pressure_deviation']:.3f} | raw_p={r['raw_pressure_value_per_ball']:.3f}")

print(f"\nReleased Bowlers qualifying (>=100b, >=12m): {len(w_rel)}")
for r in w_rel.iter_rows(named=True):
    print(f"  {r['player_name']:<22} ({r['player_id']}) | balls={r['pressure_balls']:<5} | matches={r['matches_with_pressure_balls']:<3} | p_dev={r['recentered_shrunk_pressure_deviation']:.3f} | raw_p={r['raw_pressure_value_per_ball']:.3f}")
