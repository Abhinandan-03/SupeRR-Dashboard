import polars as pl

retired = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237'}

b_df = pl.read_csv('pipeline_outputs/tables/batter_all_player_recentered_eb_diagnostics.csv')
b_sel = b_df.filter(
    (pl.col('pressure_balls') >= 100) &
    (pl.col('matches_with_pressure_balls') >= 12) &
    (~pl.col('excluded_by_ipl2026_or_pakistan')) &
    (~pl.col('player_id').is_in(retired))
).sort('rating', descending=True)

print(f"Eligible Batters count: {len(b_sel)}")
print("New Batter Top 10:")
for r in b_sel.head(10).iter_rows(named=True):
    print(f"  {r['player_id']}: {r['player_name']} | rating={r['rating']:.3f} | balls={r['pressure_balls']}")

w_df = pl.read_csv('pipeline_outputs/tables/bowler_all_player_recentered_eb_diagnostics.csv')
w_sel = w_df.filter(
    (pl.col('pressure_balls') >= 100) &
    (pl.col('matches_with_pressure_balls') >= 12) &
    (~pl.col('excluded_by_ipl2026_or_pakistan')) &
    (~pl.col('player_id').is_in(retired))
).sort('rating', descending=True)

print(f"\nEligible Bowlers count: {len(w_sel)}")
print("New Bowler Top 10:")
for r in w_sel.head(10).iter_rows(named=True):
    print(f"  {r['player_id']}: {r['player_name']} | rating={r['rating']:.3f} | balls={r['pressure_balls']}")
