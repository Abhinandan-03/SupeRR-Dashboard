import polars as pl
import time

t0 = time.time()
base = pl.read_parquet('pipeline_outputs/modeling/combined_delivery_base.parquet')

# Fast delivery-level batting order
# In each match and innings, order of appearance of striker
# We can find the min delivery row index for each striker in (match_id, innings)
first_seen = base.filter(pl.col('batter_faced') == True).group_by(['match_id', 'innings', 'striker_player_id']).agg([
    pl.col('innings_ball_num').min().alias('first_ball')
]).sort(['match_id', 'innings', 'first_ball'])

first_seen = first_seen.with_columns([
    pl.int_range(1, pl.len() + 1).over(['match_id', 'innings']).alias('inferred_batting_order')
])

print(f"Computed inferred batting order in {time.time() - t0:.2f}s. Rows: {len(first_seen)}")
print(first_seen.head(10))
