import polars as pl

base = pl.read_parquet('pipeline_outputs/modeling/combined_delivery_base.parquet')

# First seen ball in innings
first_seen = base.filter(pl.col('batter_faced') == True).group_by(['match_id', 'innings', 'striker_player_id']).agg([
    pl.col('innings_ball_num').min().alias('first_ball')
]).sort(['match_id', 'innings', 'first_ball'])

first_seen = first_seen.with_columns([
    pl.int_range(1, pl.len() + 1).over(['match_id', 'innings']).alias('inferred_batting_order')
])

# Join back to deliveries where batter_faced is True
deliv_order = base.filter(pl.col('batter_faced') == True).join(
    first_seen.select(['match_id', 'innings', 'striker_player_id', 'inferred_batting_order']),
    on=['match_id', 'innings', 'striker_player_id'],
    how='left'
)

# Aggregate position profile by batter
batter_positions = deliv_order.group_by('striker_player_id').agg([
    pl.len().alias('total_balls_faced'),
    (pl.col('inferred_batting_order').is_in([1, 2, 3])).mean().alias('pos_1_3_share'),
    (pl.col('inferred_batting_order') == 4).mean().alias('pos_4_share'),
    (pl.col('inferred_batting_order').is_in([5, 6, 7])).mean().alias('pos_5_7_share')
])

print("Batter positions sample:")
print(batter_positions.head(10))
