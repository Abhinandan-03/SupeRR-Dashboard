import polars as pl
from pathlib import Path

mp_file = Path('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
mp_df = pl.read_csv(mp_file)
pak_ids = set(mp_df.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().is_in(['pakistan', 'pakistan a']))['player_id'].drop_nulls().cast(pl.Int64).to_list())
ipl26_ids = set(mp_df.filter(pl.col('comp_name').cast(pl.Utf8) == 'IPL 2026')['player_id'].drop_nulls().cast(pl.Int64).to_list())
excluded_canonical = pak_ids | ipl26_ids
excluded_canonical.add(6628)
excl = {f'PID_{x}' for x in excluded_canonical}
retired = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237'}
excl.update(retired)

name_map = {}
pe_file = Path('inputs/data_pack/Data Pack/playing_eleven.csv')
if pe_file.exists():
    pe = pl.read_csv(pe_file)
    for r in pe.iter_rows(named=True):
        if r.get('player_id') and r.get('player_name'):
            name_map[f"PID_{r['player_id']}"] = str(r['player_name'])

audit_file = Path('pipeline_outputs/tables/smat_player_pressure_exposure_audit.csv')
if audit_file.exists():
    aud = pl.read_csv(audit_file)
    for r in aud.iter_rows(named=True):
        if r.get('player_id') and r.get('player_name'):
            name_map[str(r['player_id'])] = str(r['player_name'])

b_df = pl.read_csv('pipeline_outputs/tables/batter_player_comp_empirical_bayes.csv')
b_smat = b_df.filter((pl.col('competition_type') == 'Syed Mushtaq Ali Trophy') & (pl.col('pressure_balls') >= 15) & (pl.col('matches_with_pressure_balls') >= 2))
b_unexcl = b_smat.filter(~pl.col('player_id').is_in(excl)).with_columns(
    pl.col('player_id').map_elements(lambda x: name_map.get(str(x), str(x)), return_dtype=pl.Utf8).alias('player_name')
).sort('recentered_pressure_deviation', descending=True)

print("=== SMAT UNEXCLUDED BATTERS (>=15b, >=2m) ===")
for r in b_unexcl.head(15).iter_rows(named=True):
    print(f"{r['player_id']}: {r['player_name']} | balls={r['pressure_balls']} | matches={r['matches_with_pressure_balls']} | raw_p={r['raw_pressure_value_per_ball']:.3f} | shrunk_p={r['shrunk_pressure_value_per_ball']:.3f} | p_dev={r['recentered_pressure_deviation']:.3f}")

w_df = pl.read_csv('pipeline_outputs/tables/bowler_player_comp_empirical_bayes.csv')
w_smat = w_df.filter((pl.col('competition_type') == 'Syed Mushtaq Ali Trophy') & (pl.col('pressure_balls') >= 15) & (pl.col('matches_with_pressure_balls') >= 2))
w_unexcl = w_smat.filter(~pl.col('player_id').is_in(excl)).with_columns(
    pl.col('player_id').map_elements(lambda x: name_map.get(str(x), str(x)), return_dtype=pl.Utf8).alias('player_name')
).sort('recentered_pressure_deviation', descending=True)

print("\n=== SMAT UNEXCLUDED BOWLERS (>=15b, >=2m) ===")
for r in w_unexcl.head(15).iter_rows(named=True):
    print(f"{r['player_id']}: {r['player_name']} | balls={r['pressure_balls']} | matches={r['matches_with_pressure_balls']} | raw_p={r['raw_pressure_value_per_ball']:.3f} | shrunk_p={r['shrunk_pressure_value_per_ball']:.3f} | p_dev={r['recentered_pressure_deviation']:.3f}")
