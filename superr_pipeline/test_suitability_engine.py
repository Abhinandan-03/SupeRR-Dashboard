import polars as pl
import numpy as np

# Load b_sel and w_sel from previous script
import test_new_rankings as tnr
b_top30 = tnr.b_sel.head(30)
w_top30 = tnr.w_sel.head(30)

base = pl.read_parquet('pipeline_outputs/modeling/combined_delivery_base.parquet')
bbb = pl.read_csv('inputs/data_pack/Data Pack/ball_by_ball.csv', columns=['striker_id', 'bowler_id', 'batting_type', 'bowling_type', 'batting_order'])

# Compute delivery-level inferred batting order
first_seen = base.filter(pl.col('batter_faced') == True).group_by(['match_id', 'innings', 'striker_player_id']).agg([
    pl.col('innings_ball_num').min().alias('first_ball')
]).sort(['match_id', 'innings', 'first_ball'])

first_seen = first_seen.with_columns([
    pl.int_range(1, pl.len() + 1).over(['match_id', 'innings']).alias('inferred_batting_order')
])

deliv_order = base.filter(pl.col('batter_faced') == True).join(
    first_seen.select(['match_id', 'innings', 'striker_player_id', 'inferred_batting_order']),
    on=['match_id', 'innings', 'striker_player_id'],
    how='left'
)

# Batter position & power profiles
batter_prof = deliv_order.group_by('striker_player_id').agg([
    pl.len().alias('balls_faced'),
    (pl.col('inferred_batting_order').is_in([1, 2, 3])).mean().alias('pos_1_3_share'),
    (pl.col('inferred_batting_order') == 4).mean().alias('pos_4_share'),
    (pl.col('inferred_batting_order').is_in([5, 6, 7])).mean().alias('pos_5_7_share'),
    (pl.col('over') >= 16).mean().alias('death_balls_faced_share'),
    (pl.col('batter_runs') >= 4.0).mean().alias('boundary_rate')
]).with_columns([
    (pl.col('pos_1_3_share') + pl.col('pos_4_share')).alias('pos_1_4_share')
])
batter_prof_dict = {r['striker_player_id']: r for r in batter_prof.to_dicts()}

# Bowler phase profiles
bowler_prof = base.filter(pl.col('legal_ball') == True).group_by('bowler_player_id').agg([
    pl.len().alias('legal_balls_bowled'),
    (pl.col('over') < 6).mean().alias('powerplay_overs_1_6_share'),
    ((pl.col('over') >= 6) & (pl.col('over') < 16)).mean().alias('middle_overs_7_16_share'),
    (pl.col('over') >= 16).mean().alias('death_overs_17_20_share'),
    pl.col('pace_or_spin').mode().first().alias('base_pace_or_spin')
])
bowler_prof_dict = {r['bowler_player_id']: r for r in bowler_prof.to_dicts()}

# Batter hand from bbb
pid_bat_hand = {}
for r in bbb.filter(pl.col('striker_id').is_not_null() & pl.col('batting_type').is_not_null()).iter_rows(named=True):
    pid = f"PID_{r['striker_id']}"
    if pid not in pid_bat_hand:
        pid_bat_hand[pid] = r['batting_type']

# Bowler type from bbb
pid_bowl_type = {}
for r in bbb.filter(pl.col('bowler_id').is_not_null() & pl.col('bowling_type').is_not_null()).iter_rows(named=True):
    pid = f"PID_{r['bowler_id']}"
    if pid not in pid_bowl_type:
        pid_bowl_type[pid] = r['bowling_type']

# Nationality mapping
squads = pl.read_csv('inputs/squads.csv')
squads_nat = dict(zip(squads['player'].str.to_lowercase().to_list(), squads['nationality'].to_list()))
mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
player_nat_map = {}
for r in mp.iter_rows(named=True):
    pid = r['player_id']
    tm = r['team_name']
    if pid not in player_nat_map:
        if tm in ['Pakistan', 'Pakistan A']: player_nat_map[pid] = 'Pakistan'
        elif tm in ['India', 'India A']: player_nat_map[pid] = 'Indian'
        elif tm in ['Australia', 'England', 'South Africa', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan', 'Ireland', 'Zimbabwe', 'Netherlands', 'Scotland', 'Namibia', 'Nepal', 'Oman', 'UAE', 'USA']:
            player_nat_map[pid] = 'Overseas'

for r in mp.filter(pl.col('comp_name') == 'IPL 2026').iter_rows(named=True):
    pid = r['player_id']
    pname = r['player_name'].lower()
    if pname in squads_nat:
        player_nat_map[pid] = squads_nat[pname]

def get_nat(pid_str):
    if pid_str.startswith('PID_'):
        pid_num = int(pid_str.replace('PID_', ''))
        return player_nat_map.get(pid_num, 'Indian / Domestic')
    return 'Unknown'

def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    sd = np.std(arr, ddof=0)
    return (arr - np.mean(arr)) / sd if sd > 1e-12 else np.zeros_like(arr)

# 1. EVALUATE BATTERS
print("=== EVALUATING RR SUITABILITY FOR TOP 30 BATTERS ===")
b_rows = []
for r in b_top30.iter_rows(named=True):
    pid = r['player_id']
    b_prof = batter_prof_dict.get(pid, {})
    w_prof = bowler_prof_dict.get(pid, {})
    
    is_cs = pid.startswith('CS_')
    hand = pid_bat_hand.get(pid, 'Unknown' if is_cs else 'RHB')
    bowl_style = pid_bowl_type.get(pid, w_prof.get('base_pace_or_spin', 'Unknown' if is_cs else 'UNK'))
    is_pace = 'pace' in bowl_style.lower() or 'fast' in bowl_style.lower() or 'medium' in bowl_style.lower()
    
    pos_1_3 = b_prof.get('pos_1_3_share', 0.0)
    pos_4 = b_prof.get('pos_4_share', 0.0)
    pos_5_7 = b_prof.get('pos_5_7_share', 0.0)
    pos_1_4 = b_prof.get('pos_1_4_share', 0.0)
    death_share = b_prof.get('death_balls_faced_share', 0.0)
    b_rate = b_prof.get('boundary_rate', 0.0)
    balls_faced = b_prof.get('balls_faced', 0)
    balls_bowled = w_prof.get('legal_balls_bowled', 0)
    
    satisfied_needs = []
    verify_flags = []
    
    # Need 4: RH top-order anchor/number 3 at 1-4 (weight 1.0)
    if pos_1_4 >= 0.50:
        if hand == 'RHB':
            satisfied_needs.append((1.0, 4, "Need 4: RH top-order anchor/No.3 (1-4)"))
        elif is_cs:
            satisfied_needs.append((0.5 * 1.0, 4, "Need 4: RH top-order anchor/No.3 (1-4)"))
            verify_flags.append("verify manually: batting hand")
            
    # Need 3: Power finisher at positions 5-7 (weight 1.0)
    if pos_5_7 >= 0.25 and (death_share >= 0.20 or b_rate >= 0.15):
        satisfied_needs.append((1.0, 3, "Need 3: Power finisher (5-7)"))
        
    # Need 5: Seam-bowling all-rounder at 6-7 (weight 0.8)
    if balls_faced >= 30 and pos_5_7 >= 0.25 and balls_bowled >= 30:
        if is_pace:
            satisfied_needs.append((0.8, 5, "Need 5: Seam-bowling all-rounder (6-7)"))
        elif is_cs:
            satisfied_needs.append((0.5 * 0.8, 5, "Need 5: Seam-bowling all-rounder (6-7)"))
            verify_flags.append("verify manually: bowling type")
            
    # Need 7: Opener backup, positions 1-3 (weight 0.6)
    if pos_1_3 >= 0.50:
        satisfied_needs.append((0.6, 7, "Need 7: Opener backup (1-3)"))
        
    if satisfied_needs:
        satisfied_needs.sort(key=lambda x: x[0], reverse=True)
        best_w, best_nid, best_desc = satisfied_needs[0]
    else:
        best_w, best_nid, best_desc = 0.0, 0, "None"
        
    # Label
    if best_nid in [1, 3, 4]: label = "XI Contender"
    elif best_nid in [2, 5]: label = "XI Contender / Rotation"
    elif best_nid in [6, 7]: label = "Backup"
    else: label = "Backup / Depth"
    
    b_rows.append({
        'player_id': pid,
        'player_name': r['player_name'],
        'performance_rank': r['final_rank'] if 'final_rank' in r else 0,
        'rating': r['rating'],
        'need_fit_score': best_w,
        'primary_need_filled': best_desc,
        'role_label': label,
        'nationality': get_nat(pid),
        'verify_manually': "; ".join(verify_flags) if verify_flags else "None",
        'pressure_balls': r['pressure_balls'],
        'matches_with_pressure_balls': r['matches_with_pressure_balls']
    })

df_b_suit = pl.DataFrame(b_rows)
# Compute z-scores within top 30
z_perf_b = zscore(df_b_suit['rating'].to_numpy())
z_fit_b = zscore(df_b_suit['need_fit_score'].to_numpy())
score_75_25_b = 0.75 * z_perf_b + 0.25 * z_fit_b
score_100_0_b = z_perf_b
score_50_50_b = 0.50 * z_perf_b + 0.50 * z_fit_b

df_b_suit = df_b_suit.with_columns([
    pl.Series('rr_suitability_score_75_25', score_75_25_b),
    pl.Series('score_100_0', score_100_0_b),
    pl.Series('score_50_50', score_50_50_b)
]).sort('rr_suitability_score_75_25', descending=True)

for i, r in enumerate(df_b_suit.head(10).iter_rows(named=True), 1):
    print(f"  #{i:<2} {r['player_name']:<20} | RR Suit: {r['rr_suitability_score_75_25']:+.3f} | Perf Rk: {r['performance_rank']:<2} (r={r['rating']:+.2f}) | Fit: {r['need_fit_score']:.2f} | Need: {r['primary_need_filled']} | {r['role_label']} ({r['nationality']})")

# 2. EVALUATE BOWLERS
print("\n=== EVALUATING RR SUITABILITY FOR TOP 30 BOWLERS ===")
w_rows = []
for r in w_top30.iter_rows(named=True):
    pid = r['player_id']
    w_prof = bowler_prof_dict.get(pid, {})
    b_prof = batter_prof_dict.get(pid, {})
    
    is_cs = pid.startswith('CS_')
    bowl_style = pid_bowl_type.get(pid, w_prof.get('base_pace_or_spin', 'Unknown' if is_cs else 'UNK'))
    is_pace = 'pace' in bowl_style.lower() or 'fast' in bowl_style.lower() or 'medium' in bowl_style.lower()
    is_spin = 'spin' in bowl_style.lower() or 'orthodox' in bowl_style.lower() or 'chinaman' in bowl_style.lower() or 'offbreak' in bowl_style.lower() or 'leg-break' in bowl_style.lower()
    
    pp_share = w_prof.get('powerplay_overs_1_6_share', 0.0)
    mid_share = w_prof.get('middle_overs_7_16_share', 0.0)
    death_share = w_prof.get('death_overs_17_20_share', 0.0)
    balls_bowled = w_prof.get('legal_balls_bowled', 0)
    balls_faced = b_prof.get('balls_faced', 0)
    pos_5_7 = b_prof.get('pos_5_7_share', 0.0)
    
    satisfied_needs = []
    verify_flags = []
    
    # Need 1: Pace death-overs specialist (weight 1.0)
    if death_share >= 0.20:
        if is_pace:
            satisfied_needs.append((1.0, 1, "Need 1: Pace death specialist"))
        elif is_cs:
            satisfied_needs.append((0.5 * 1.0, 1, "Need 1: Pace death specialist"))
            verify_flags.append("verify manually: bowling type")
            
    # Need 2: Pace new-ball/middle seam (weight 0.8)
    if pp_share >= 0.30 or mid_share >= 0.40:
        if is_pace:
            satisfied_needs.append((0.8, 2, "Need 2: Pace powerplay/middle seam"))
        elif is_cs:
            satisfied_needs.append((0.5 * 0.8, 2, "Need 2: Pace powerplay/middle seam"))
            verify_flags.append("verify manually: bowling type")
            
    # Need 5: Seam-bowling all-rounder at 6-7 (weight 0.8)
    if balls_bowled >= 30 and balls_faced >= 30 and pos_5_7 >= 0.25:
        if is_pace:
            satisfied_needs.append((0.8, 5, "Need 5: Seam-bowling all-rounder (6-7)"))
        elif is_cs:
            satisfied_needs.append((0.5 * 0.8, 5, "Need 5: Seam-bowling all-rounder (6-7)"))
            verify_flags.append("verify manually: bowling type")
            
    # Need 6: Spin backup (weight 0.6)
    if is_spin:
        satisfied_needs.append((0.6, 6, "Need 6: Spin backup"))
    elif is_cs:
        satisfied_needs.append((0.5 * 0.6, 6, "Need 6: Spin backup"))
        verify_flags.append("verify manually: bowling type")
        
    if satisfied_needs:
        satisfied_needs.sort(key=lambda x: x[0], reverse=True)
        best_w, best_nid, best_desc = satisfied_needs[0]
    else:
        best_w, best_nid, best_desc = 0.0, 0, "None"
        
    # Label
    if best_nid in [1, 3, 4]: label = "XI Contender"
    elif best_nid in [2, 5]: label = "XI Contender / Rotation"
    elif best_nid in [6, 7]: label = "Backup"
    else: label = "Backup / Depth"
    
    w_rows.append({
        'player_id': pid,
        'player_name': r['player_name'],
        'performance_rank': r['final_rank'] if 'final_rank' in r else 0,
        'rating': r['rating'],
        'need_fit_score': best_w,
        'primary_need_filled': best_desc,
        'role_label': label,
        'nationality': get_nat(pid),
        'verify_manually': "; ".join(verify_flags) if verify_flags else "None",
        'pressure_balls': r['pressure_balls'],
        'matches_with_pressure_balls': r['matches_with_pressure_balls']
    })

df_w_suit = pl.DataFrame(w_rows)
z_perf_w = zscore(df_w_suit['rating'].to_numpy())
z_fit_w = zscore(df_w_suit['need_fit_score'].to_numpy())
score_75_25_w = 0.75 * z_perf_w + 0.25 * z_fit_w
score_100_0_w = z_perf_w
score_50_50_w = 0.50 * z_perf_w + 0.50 * z_fit_w

df_w_suit = df_w_suit.with_columns([
    pl.Series('rr_suitability_score_75_25', score_75_25_w),
    pl.Series('score_100_0', score_100_0_w),
    pl.Series('score_50_50', score_50_50_w)
]).sort('rr_suitability_score_75_25', descending=True)

for i, r in enumerate(df_w_suit.head(10).iter_rows(named=True), 1):
    print(f"  #{i:<2} {r['player_name']:<20} | RR Suit: {r['rr_suitability_score_75_25']:+.3f} | Perf Rk: {r['performance_rank']:<2} (r={r['rating']:+.2f}) | Fit: {r['need_fit_score']:.2f} | Need: {r['primary_need_filled']} | {r['role_label']} ({r['nationality']})")
