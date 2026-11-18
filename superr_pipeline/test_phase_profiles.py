import polars as pl
import numpy as np

base = pl.read_parquet('pipeline_outputs/modeling/combined_delivery_base.parquet')
print(f"Loaded {len(base):,} deliveries.")

# 1. Bowling phase profile
bowler_phases = base.filter(pl.col('legal_ball') == True).group_by('bowler_player_id').agg([
    pl.len().alias('total_legal_balls_bowled'),
    (pl.col('over') < 6).mean().alias('powerplay_overs_1_6_share'),
    ((pl.col('over') >= 6) & (pl.col('over') < 16)).mean().alias('middle_overs_7_16_share'),
    (pl.col('over') >= 16).mean().alias('death_overs_17_20_share')
])
print("Bowler phase profile sample:")
print(bowler_phases.head(5))

# 2. Batter power profile
batter_power = base.filter(pl.col('batter_faced') == True).group_by('striker_player_id').agg([
    pl.len().alias('total_balls_faced'),
    (pl.col('over') >= 16).mean().alias('death_balls_faced_share'),
    (pl.col('batter_runs') >= 4.0).mean().alias('boundary_rate')
])
print("\nBatter power profile sample:")
print(batter_power.head(5))
