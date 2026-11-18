import polars as pl

base = pl.read_parquet('pipeline_outputs/modeling/combined_delivery_base.parquet')

# Test tracking batting order by innings
test_m = base.filter((pl.col('match_id') == base['match_id'][0]) & (pl.col('innings') == 1))
print("Test match rows:", len(test_m))

batters_seen = []
orders = []
for r in test_m.iter_rows(named=True):
    s = r['striker_player_id']
    ns = r['non_striker_player_id']
    for p in [s, ns]:
        if p and p not in batters_seen:
            batters_seen.append(p)
    orders.append(batters_seen.index(s) + 1)

print("Unique batters in order:", batters_seen)
print("Orders assigned sample:", orders[:15])
