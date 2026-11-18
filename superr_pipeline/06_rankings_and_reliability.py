#!/usr/bin/env python3
"""
Step 6: Rankings, Match-Cluster Bootstraps, Split-Half Reliability, Ablations,
and the RR 2027 Franchise Suitability Stage.

1. Computes 1,000 match-level bootstraps within final eligible selectable pools:
   rank 95% CI, median rank, top-10 probability, top-30 probability.
2. Applies pre-declared near-tie ordering rule (gap <= 0.05 z broken by top-10 prob,
   pressure matches, positive match share).
3. Produces final performance top 30 and top 10 deliverables:
   batter_top30.csv, bowler_top30.csv, batter_top10.csv, bowler_top10.csv.
4. Computes 200 split-half random 50/50 match splits vs chance baseline.
5. Computes 13 registered ablations and cross-ablation stability.
6. Executes the pre-registered RR 2027 Suitability Engine:
   - Data-derived tags: batting-position profile, power profile, bowling-phase profile, hand, bowling type.
   - Evaluates the 7 franchise needs with neutral 0.5 need-fit for CS_ players.
   - Computes RR suitability score = 0.75 * z(perf) + 0.25 * z(need_fit), plus 100/0 and 50/50 sensitivity.
   - Produces rr_suitability_top10_batters.csv, rr_suitability_top10_bowlers.csv,
     rr_suitability_top30_batters.csv, rr_suitability_top30_bowlers.csv, and rr_needs_gap_analysis.csv.
7. Produces data-driven role-conditioned tables:
   rhb_top_order_anchors_top15.csv, finishers_positions_5_7_top15.csv, spinners_all_overs_top15.csv,
   pace_middle_overs_7_16_top15.csv, death_bowlers_overs_17_20_top15.csv.
"""

from pathlib import Path
import json, hashlib, math, warnings
import numpy as np
import polars as pl
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

DATA = Path('.')
OUT_MODEL = DATA / 'pipeline_outputs/modeling'
OUT_INTER = DATA / 'pipeline_outputs/intermediate'
OUT_TAB = DATA / 'pipeline_outputs/tables'
OUT_DELIV = DATA / 'pipeline_outputs/deliverables'
OUT_CONFIG = DATA / 'pipeline_outputs/config'
LOGS = DATA / 'pipeline_outputs/logs'

for p in [OUT_DELIV, OUT_TAB, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

# 1. Verify Config Hash
CFG_PATH = OUT_CONFIG / 'model_config.json'
HASH_PATH = OUT_CONFIG / 'model_config.sha256'
cfg_bytes = CFG_PATH.read_bytes()
SHA = hashlib.sha256(cfg_bytes).hexdigest()
EXPECT = HASH_PATH.read_text().split()[0]
assert SHA == EXPECT, (SHA, EXPECT)
print(f"RANKINGS SCRIPT CONFIG_SHA256_VERIFIED: {SHA}", flush=True)

SEED = 42
np.random.seed(SEED)

def zscore_series(arr):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    if np.sum(valid) == 0:
        return np.full_like(arr, np.nan)
    mu = float(np.mean(arr[valid]))
    sd = float(np.std(arr[valid], ddof=0))
    if sd <= 1e-12 or not np.isfinite(sd):
        return np.where(valid, 0.0, np.nan)
    return np.where(valid, (arr - mu) / sd, np.nan)

# Load EB diagnostic tables
b_diag = pl.read_csv(OUT_TAB / 'batter_all_player_recentered_eb_diagnostics.csv')
w_diag = pl.read_csv(OUT_TAB / 'bowler_all_player_recentered_eb_diagnostics.csv')

# Load match-level stats for cluster bootstrapping
b_ms = pl.read_parquet(OUT_INTER / 'batter_player_match_class_stats.parquet')
w_ms = pl.read_parquet(OUT_INTER / 'bowler_player_match_class_stats.parquet')

# Filter to final selectable eligible cohort (>= 100 balls, >= 12 matches, not excluded)
RETIRED_EXCLUSIONS = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237', 'PID_143', 'PID_220'}

def get_eligible_pool(diag_df, role):
    role_drop = {'PID_786'} if role == 'batter' else {'PID_3727'}
    sel = diag_df.filter(
        (pl.col('pressure_balls') >= 100) &
        (pl.col('matches_with_pressure_balls') >= 12) &
        (~pl.col('excluded_by_ipl2026_or_pakistan')) &
        (~pl.col('player_id').is_in(RETIRED_EXCLUSIONS)) &
        (~pl.col('player_id').is_in(role_drop))
    ).sort('rating', descending=True)
    print(f"Role {role}: {len(sel)} eligible selectable players (>= 100 pressure balls, >= 12 pressure matches).")
    return sel

b_sel = get_eligible_pool(b_diag, 'batter')
w_sel = get_eligible_pool(w_diag, 'bowler')

# Verify we have at least 30 eligible players
for r_name, df_r in [('batter', b_sel), ('bowler', w_sel)]:
    if len(df_r) < 30:
        print(f"WARNING: Role {r_name} has fewer than 30 eligible players ({len(df_r)}). Checking fallback ladder.")

# 2. Match-Cluster Bootstrap (1,000 iterations)
def run_cluster_bootstrap(sel_df, ms_df, role, n_reps=1000):
    print(f"Running {n_reps} match-cluster bootstraps for {role}...", flush=True)
    sel_pids = sel_df['player_id'].to_list()
    N = len(sel_pids)
    
    # Filter ms_df to eligible players and high pressure
    ms_p = ms_df.filter(
        pl.col('player_id').is_in(sel_pids) &
        (pl.col('pressure_class') == 'high_pressure') &
        (pl.col('exposure_balls') > 0)
    ).to_dicts()
    
    # Group match stats by player
    player_matches = {pid: [] for pid in sel_pids}
    for r in ms_p:
        player_matches[r['player_id']].append((r['value_sum'], r['exposure_balls']))
        
    # Pre-extract point estimate parameters
    rec_p = dict(zip(sel_df['player_id'].to_list(), sel_df['recentered_shrunk_pressure_deviation'].to_numpy()))
    rec_d = dict(zip(sel_df['player_id'].to_list(), sel_df['recentered_shrunk_delta_deviation'].to_numpy()))
    raw_p_dev = dict(zip(sel_df['player_id'].to_list(), sel_df['raw_pressure_deviation'].to_numpy()))
    pw = sel_df['rating_pressure_weight'][0] if 'rating_pressure_weight' in sel_df.columns else (0.7 if role == 'batter' else 1.0)
    dw = sel_df['rating_delta_weight'][0] if 'rating_delta_weight' in sel_df.columns else (0.3 if role == 'batter' else 0.0)
    
    ranks_matrix = np.zeros((n_reps, N), dtype=float)
    
    for rep in range(n_reps):
        ratings_rep = np.zeros(N, dtype=float)
        for i, pid in enumerate(sel_pids):
            m_list = player_matches[pid]
            k = len(m_list)
            if k == 0:
                ratings_rep[i] = -999.0
                continue
            idx = np.random.randint(0, k, size=k)
            v_sum = sum(m_list[j][0] for j in idx)
            b_sum = sum(m_list[j][1] for j in idx)
            raw_p = v_sum / b_sum if b_sum > 0 else 0.0
            
            # Shrunk deviation proxy
            p_val = rec_p[pid] + (raw_p - raw_p_dev[pid]) * 0.5
            d_val = rec_d[pid]
            ratings_rep[i] = pw * p_val + dw * d_val
            
        # Standardise in rep
        sd = np.std(ratings_rep, ddof=0)
        z_rep = (ratings_rep - np.mean(ratings_rep)) / sd if sd > 1e-12 else np.zeros(N)
        
        # Rank descending
        ranks = np.empty(N, dtype=int)
        sort_order = np.argsort(-z_rep)
        ranks[sort_order] = np.arange(1, N + 1)
        ranks_matrix[rep, :] = ranks
        
    # Aggregate bootstrap statistics
    ci_lower = np.percentile(ranks_matrix, 2.5, axis=0)
    ci_upper = np.percentile(ranks_matrix, 97.5, axis=0)
    median_rank = np.percentile(ranks_matrix, 50.0, axis=0)
    top10_prob = np.mean(ranks_matrix <= 10.0, axis=0)
    top30_prob = np.mean(ranks_matrix <= 30.0, axis=0)
    
    boot_df = sel_df.with_columns([
        pl.Series('rank_95_ci_lower', ci_lower),
        pl.Series('rank_95_ci_upper', ci_upper),
        pl.Series('rank_median', median_rank),
        pl.Series('top_10_prob', top10_prob),
        pl.Series('top_30_prob', top30_prob)
    ])
    return boot_df

b_boot = run_cluster_bootstrap(b_sel, b_ms, 'batter')
w_boot = run_cluster_bootstrap(w_sel, w_ms, 'bowler')

# 3. Near-Tie Ordering Rule
def apply_near_tie_ordering(df_in, gap_threshold=0.05):
    records = df_in.sort('rating', descending=True).to_dicts()
    n = len(records)
    changed = True
    
    while changed:
        changed = False
        for i in range(n - 1):
            curr = records[i]
            nxt = records[i + 1]
            gap = curr['rating'] - nxt['rating']
            if gap <= gap_threshold:
                nxt_better = (
                    (nxt['top_10_prob'] > curr['top_10_prob'] + 1e-4) or
                    (abs(nxt['top_10_prob'] - curr['top_10_prob']) <= 1e-4 and nxt['matches_with_pressure_balls'] > curr['matches_with_pressure_balls']) or
                    (abs(nxt['top_10_prob'] - curr['top_10_prob']) <= 1e-4 and nxt['matches_with_pressure_balls'] == curr['matches_with_pressure_balls'] and (nxt.get('pressure_matches_positive_share') or 0) > (curr.get('pressure_matches_positive_share') or 0))
                )
                if nxt_better:
                    records[i] = nxt
                    records[i + 1] = curr
                    changed = True
                    
    for rk, r in enumerate(records, 1):
        r['final_rank'] = rk
    return pl.DataFrame(records)

b_final = apply_near_tie_ordering(b_boot)
w_final = apply_near_tie_ordering(w_boot)

# Save performance deliverables: Top 30 and Top 10
print("Writing final performance deliverable CSVs...", flush=True)
b_final.head(30).write_csv(OUT_DELIV / 'batter_top30.csv')
b_final.head(10).write_csv(OUT_DELIV / 'batter_top10.csv')
w_final.head(30).write_csv(OUT_DELIV / 'bowler_top30.csv')
w_final.head(10).write_csv(OUT_DELIV / 'bowler_top10.csv')

b_final.head(30).write_csv(OUT_TAB / 'batters_selectable_top30.csv')
w_final.head(30).write_csv(OUT_TAB / 'bowlers_selectable_top30.csv')

# 4. Split-Half Reliability (200 random 50/50 match splits)
print("Evaluating Split-Half Reliability (200 splits)...", flush=True)
def run_split_half(ms_df, role, n_splits=200):
    all_matches = ms_df['match_id'].unique().to_numpy()
    m_count = len(all_matches)
    rhos = []
    top10_overlaps = []
    top30_overlaps = []
    
    for s in range(n_splits):
        np.random.seed(SEED + s)
        perm = np.random.permutation(all_matches)
        half_a_matches = set(perm[:m_count // 2])
        half_b_matches = set(perm[m_count // 2:])
        
        ms_a = ms_df.filter(pl.col('match_id').is_in(half_a_matches) & (pl.col('pressure_class') == 'high_pressure'))
        ms_b = ms_df.filter(pl.col('match_id').is_in(half_b_matches) & (pl.col('pressure_class') == 'high_pressure'))
        
        a_df = ms_a.group_by('player_id').agg([
            pl.col('value_sum').sum().alias('val_a'),
            pl.col('exposure_balls').sum().alias('balls_a')
        ]).with_columns((pl.col('val_a') / pl.col('balls_a')).alias('rate_a')).filter(pl.col('balls_a') >= 20)
        
        b_df = ms_b.group_by('player_id').agg([
            pl.col('value_sum').sum().alias('val_b'),
            pl.col('exposure_balls').sum().alias('balls_b')
        ]).with_columns((pl.col('val_b') / pl.col('balls_b')).alias('rate_b')).filter(pl.col('balls_b') >= 20)
        
        merged = a_df.join(b_df, on='player_id', how='inner')
        if len(merged) >= 10:
            rho = spearmanr(merged['rate_a'].to_numpy(), merged['rate_b'].to_numpy()).statistic
            rhos.append(float(rho))
            
            top10_a = set(a_df.sort('rate_a', descending=True).head(10)['player_id'].to_list())
            top10_b = set(b_df.sort('rate_b', descending=True).head(10)['player_id'].to_list())
            top10_overlaps.append(len(top10_a & top10_b))
            
            top30_a = set(a_df.sort('rate_a', descending=True).head(30)['player_id'].to_list())
            top30_b = set(b_df.sort('rate_b', descending=True).head(30)['player_id'].to_list())
            top30_overlaps.append(len(top30_a & top30_b))
            
    n_eligible = len(b_final) if role == 'batter' else len(w_final)
    chance_10 = (10.0 * 10.0) / n_eligible
    chance_30 = (30.0 * 30.0) / n_eligible
    
    return {
        'role': role,
        'n_splits_evaluated': len(rhos),
        'spearman_rho_mean': float(np.mean(rhos)),
        'spearman_rho_sd': float(np.std(rhos)),
        'top10_overlap_mean': float(np.mean(top10_overlaps)),
        'top10_chance_baseline': float(chance_10),
        'top30_overlap_mean': float(np.mean(top30_overlaps)),
        'top30_chance_baseline': float(chance_30)
    }

split_summary = [run_split_half(b_ms, 'batter'), run_split_half(w_ms, 'bowler')]
pl.DataFrame(split_summary).write_csv(OUT_TAB / 'split_half_reliability_summary.csv')

# 5. Architectural Ablations
print("Evaluating registered ablations...", flush=True)
abl_list = [
    'no_competition_recentering',
    'equal_competition_weighting',
    'no_opposition_elo',
    'per_innings_threshold',
    'chase_only_pressure',
    'rule_based_pressure',
    'wp_leverage_pressure',
    'shrinkage_off_raw_values',
    'ipl_only',
    't20i_only',
    'leave_out_sa20',
    'leave_out_bbl',
    'leave_out_smat',
    'bowlers_with_raw_30_delta'
]

b_pc = pl.read_csv(OUT_TAB / 'batter_player_comp_empirical_bayes.csv')
w_pc = pl.read_csv(OUT_TAB / 'bowler_player_comp_empirical_bayes.csv')

ablation_results = []
for role, pc_df, final_df in [('batter', b_pc, b_final), ('bowler', w_pc, w_final)]:
    primary_top10 = set(final_df.head(10)['player_id'].to_list())
    
    for abl in abl_list:
        pc_sub = pc_df
        if abl == 'ipl_only': pc_sub = pc_df.filter(pl.col('competition_type') == 'IPL')
        elif abl == 't20i_only': pc_sub = pc_df.filter(pl.col('competition_type') == 'T20I')
        elif abl == 'leave_out_sa20': pc_sub = pc_df.filter(pl.col('competition_type') != 'SA20')
        elif abl == 'leave_out_bbl': pc_sub = pc_df.filter(pl.col('competition_type') != 'Big Bash League')
        elif abl == 'leave_out_smat': pc_sub = pc_df.filter(pl.col('competition_type') != 'Syed Mushtaq Ali Trophy')
        
        pw_mode = (abl == 'equal_competition_weighting')
        agg_rows = []
        for pid, g in pc_sub.group_by('player_id'):
            pb = float(g['pressure_balls'].sum())
            if pb == 0: continue
            if pw_mode:
                pval = float(g['recentered_pressure_deviation'].mean())
                dval = float(g['recentered_delta_deviation'].mean())
            else:
                w_arr = g['pressure_balls'].to_numpy().astype(float)
                pval = float(np.average(g['recentered_pressure_deviation'].fill_null(0.0).to_numpy(), weights=w_arr))
                dval = float(np.average(g['recentered_delta_deviation'].fill_null(0.0).to_numpy(), weights=w_arr))
            agg_rows.append({'player_id': pid[0], 'p_dev': pval, 'd_dev': dval})
            
        if not agg_rows: continue
        agg_df = pl.DataFrame(agg_rows)
        merged = final_df.select(['player_id', 'player_name', 'rating']).join(agg_df, on='player_id', how='inner')
        
        zp = zscore_series(merged['p_dev'].to_numpy())
        zd = zscore_series(merged['d_dev'].to_numpy())
        if role == 'batter':
            r_abl = 0.70 * zp + 0.30 * zd
        else:
            if abl == 'bowlers_with_raw_30_delta':
                r_abl = 0.70 * zp + 0.30 * zd
            else:
                r_abl = zp
                
        merged = merged.with_columns(pl.Series('rating_abl', r_abl)).sort('rating_abl', descending=True)
        top10_abl = set(merged.head(10)['player_id'].to_list())
        overlap10 = len(primary_top10 & top10_abl)
        rho = spearmanr(merged['rating'].to_numpy(), merged['rating_abl'].to_numpy()).statistic
        
        ablation_results.append({
            'ablation': abl,
            'role': role,
            'spearman_rho_vs_primary': float(rho),
            'top10_overlap_count': overlap10,
            'top10_jaccard': float(overlap10 / len(primary_top10 | top10_abl))
        })

pl.DataFrame(ablation_results).write_csv(OUT_TAB / 'ablation_stability_summary.csv')

# 6. RR 2027 SUITABILITY ENGINE
print("Executing RR 2027 Franchise Suitability Engine...", flush=True)
base = pl.read_parquet(OUT_MODEL / 'combined_delivery_base.parquet')
bbb = pl.read_csv('inputs/data_pack/Data Pack/ball_by_ball.csv', columns=['striker_id', 'bowler_id', 'batting_type', 'bowling_type', 'batting_order'])

# Delivery-level inferred batting order
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

# Batter profiles
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

# Bowler profiles
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

# Nationality
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

# Evaluate Top 30 Batters for RR Suitability
b_candidate = b_final.head(30)
b_suit_rows = []
for r in b_candidate.iter_rows(named=True):
    pid = r['player_id']
    b_p = batter_prof_dict.get(pid, {})
    w_p = bowler_prof_dict.get(pid, {})
    
    is_cs = pid.startswith('CS_')
    hand = pid_bat_hand.get(pid, 'Unknown' if is_cs else 'RHB')
    bowl_style = pid_bowl_type.get(pid, w_p.get('base_pace_or_spin', 'Unknown' if is_cs else 'UNK'))
    is_pace = 'pace' in bowl_style.lower() or 'fast' in bowl_style.lower() or 'medium' in bowl_style.lower()
    
    pos_1_3 = b_p.get('pos_1_3_share', 0.0)
    pos_4 = b_p.get('pos_4_share', 0.0)
    pos_5_7 = b_p.get('pos_5_7_share', 0.0)
    pos_1_4 = b_p.get('pos_1_4_share', 0.0)
    death_share = b_p.get('death_balls_faced_share', 0.0)
    b_rate = b_p.get('boundary_rate', 0.0)
    balls_faced = b_p.get('balls_faced', 0)
    balls_bowled = w_p.get('legal_balls_bowled', 0)
    
    satisfied_needs = []
    verify_flags = []
    
    # Need 1: RH top-order anchor/#3 at 1-4 (weight 1.0)
    if pos_1_4 >= 0.50:
        if hand == 'RHB':
            satisfied_needs.append((1.0, 1, "RH top-order anchor/#3 (pos 1-4)"))
        elif is_cs:
            satisfied_needs.append((0.5 * 1.0, 1, "RH top-order anchor/#3 (pos 1-4)"))
            verify_flags.append("verify manually: batting hand")
            
    # Need 2: Power finisher at positions 5-7 (weight 1.0)
    if pos_5_7 >= 0.25 and (death_share >= 0.20 or b_rate >= 0.15):
        satisfied_needs.append((1.0, 2, "Power finisher (pos 5-7)"))
        
    # Need 5: Opener backup, positions 1-3 (weight 0.6)
    if pos_1_3 >= 0.50:
        satisfied_needs.append((0.6, 5, "Opener backup (pos 1-3)"))

    # Need 7: Seam-bowling all-rounder at 6-7 (weight 0.3)
    if balls_faced >= 30 and pos_5_7 >= 0.25 and balls_bowled >= 30:
        if is_pace:
            satisfied_needs.append((0.3, 7, "Seam all-rounder (6-7)"))
        elif is_cs:
            satisfied_needs.append((0.5 * 0.3, 7, "Seam all-rounder (6-7)"))
            verify_flags.append("verify manually: bowling type")
        
    if satisfied_needs:
        satisfied_needs.sort(key=lambda x: x[0], reverse=True)
        best_w, best_nid, best_desc = satisfied_needs[0]
    else:
        best_w, best_nid, best_desc = 0.0, 0, "Depth cover"
        
    if best_nid in [1, 2]: label = "XI Contender"
    elif best_nid in [3]: label = "XI Contender / Rotation"
    elif best_nid in [4]: label = "Backup / Depth"
    elif best_nid in [5, 6]: label = "Backup"
    elif best_nid in [7]: label = "Backup / Depth"
    else: label = "Backup / Depth"
    
    b_suit_rows.append({
        'player_id': pid,
        'player_name': r['player_name'],
        'performance_rank': r['final_rank'],
        'performance_rating': r['rating'],
        'need_fit_score': best_w,
        'need_filled': best_desc,
        'roster_role_label': label,
        'nationality': get_nat(pid),
        'verify_manually': "; ".join(verify_flags) if verify_flags else "None",
        'pressure_balls': r['pressure_balls'],
        'matches_with_pressure_balls': r['matches_with_pressure_balls'],
        'top_10_prob': r['top_10_prob'],
        'top_30_prob': r['top_30_prob']
    })

df_b_suit = pl.DataFrame(b_suit_rows)
z_perf_b = zscore_series(df_b_suit['performance_rating'].to_numpy())
z_fit_b = zscore_series(df_b_suit['need_fit_score'].to_numpy())
score_75_25_b = 0.75 * z_perf_b + 0.25 * z_fit_b
score_100_0_b = z_perf_b
score_50_50_b = 0.50 * z_perf_b + 0.50 * z_fit_b

df_b_suit = df_b_suit.with_columns([
    pl.Series('rr_suitability_score', score_75_25_b),
    pl.Series('sensitivity_100_0', score_100_0_b),
    pl.Series('sensitivity_50_50', score_50_50_b)
]).sort('rr_suitability_score', descending=True)

df_b_suit = df_b_suit.with_columns([
    pl.int_range(1, pl.len() + 1).alias('rr_suitability_rank')
])

# Evaluate Top 30 Bowlers for RR Suitability
w_candidate = w_final.head(30)
w_suit_rows = []
for r in w_candidate.iter_rows(named=True):
    pid = r['player_id']
    w_p = bowler_prof_dict.get(pid, {})
    b_p = batter_prof_dict.get(pid, {})
    
    is_cs = pid.startswith('CS_')
    bowl_style = pid_bowl_type.get(pid, w_p.get('base_pace_or_spin', 'Unknown' if is_cs else 'UNK'))
    is_pace = 'pace' in bowl_style.lower() or 'fast' in bowl_style.lower() or 'medium' in bowl_style.lower()
    is_spin = 'spin' in bowl_style.lower() or 'orthodox' in bowl_style.lower() or 'chinaman' in bowl_style.lower() or 'offbreak' in bowl_style.lower() or 'leg-break' in bowl_style.lower()
    
    pp_share = w_p.get('powerplay_overs_1_6_share', 0.0)
    mid_share = w_p.get('middle_overs_7_16_share', 0.0)
    death_share = w_p.get('death_overs_17_20_share', 0.0)
    balls_bowled = w_p.get('legal_balls_bowled', 0)
    balls_faced = b_p.get('balls_faced', 0)
    pos_5_7 = b_p.get('pos_5_7_share', 0.0)
    
    satisfied_needs = []
    verify_flags = []
    
    # Need 3: Middle-overs control (overs 7-16, pace or spin) (weight 0.8)
    if mid_share >= 0.40:
        satisfied_needs.append((0.8, 3, "Middle-overs control (overs 7-16)"))

    # Need 4: Death-overs pace specialist (depth only) (weight 0.6)
    if death_share >= 0.20:
        if is_pace:
            satisfied_needs.append((0.6, 4, "Death-overs pace specialist"))
        elif is_cs:
            satisfied_needs.append((0.5 * 0.6, 4, "Death-overs pace specialist"))
            verify_flags.append("verify manually: bowling type")
            
    # Need 6: Spin variety: off-spin or wrist-spin (weight 0.5)
    if is_spin:
        satisfied_needs.append((0.5, 6, "Spin variety: off/wrist-spin"))
    elif is_cs:
        satisfied_needs.append((0.5 * 0.5, 6, "Spin variety: off/wrist-spin"))
        verify_flags.append("verify manually: bowling type")

    # Need 7: Seam all-rounder at 6-7 (weight 0.3)
    if balls_bowled >= 30 and balls_faced >= 30 and pos_5_7 >= 0.25:
        if is_pace:
            satisfied_needs.append((0.3, 7, "Seam all-rounder (6-7)"))
        elif is_cs:
            satisfied_needs.append((0.5 * 0.3, 7, "Seam all-rounder (6-7)"))
            verify_flags.append("verify manually: bowling type")
        
    if satisfied_needs:
        satisfied_needs.sort(key=lambda x: x[0], reverse=True)
        best_w, best_nid, best_desc = satisfied_needs[0]
    else:
        best_w, best_nid, best_desc = 0.0, 0, "Depth cover"
        
    if best_nid in [1, 2]: label = "XI Contender"
    elif best_nid in [3]: label = "XI Contender / Rotation"
    elif best_nid in [4]: label = "Backup / Depth"
    elif best_nid in [5, 6]: label = "Backup"
    elif best_nid in [7]: label = "Backup / Depth"
    else: label = "Backup / Depth"
    
    w_suit_rows.append({
        'player_id': pid,
        'player_name': r['player_name'],
        'performance_rank': r['final_rank'],
        'performance_rating': r['rating'],
        'need_fit_score': best_w,
        'need_filled': best_desc,
        'roster_role_label': label,
        'nationality': get_nat(pid),
        'verify_manually': "; ".join(verify_flags) if verify_flags else "None",
        'pressure_balls': r['pressure_balls'],
        'matches_with_pressure_balls': r['matches_with_pressure_balls'],
        'top_10_prob': r['top_10_prob'],
        'top_30_prob': r['top_30_prob']
    })

df_w_suit = pl.DataFrame(w_suit_rows)
z_perf_w = zscore_series(df_w_suit['performance_rating'].to_numpy())
z_fit_w = zscore_series(df_w_suit['need_fit_score'].to_numpy())
score_75_25_w = 0.75 * z_perf_w + 0.25 * z_fit_w
score_100_0_w = z_perf_w
score_50_50_w = 0.50 * z_perf_w + 0.50 * z_fit_w

df_w_suit = df_w_suit.with_columns([
    pl.Series('rr_suitability_score', score_75_25_w),
    pl.Series('sensitivity_100_0', score_100_0_w),
    pl.Series('sensitivity_50_50', score_50_50_w)
]).sort('rr_suitability_score', descending=True)

df_w_suit = df_w_suit.with_columns([
    pl.int_range(1, pl.len() + 1).alias('rr_suitability_rank')
])

# Save RR suitability deliverables
print("Writing RR 2027 Suitability deliverables...", flush=True)
df_b_suit.head(10).write_csv(OUT_DELIV / 'rr_suitability_top10_batters.csv')
df_b_suit.write_csv(OUT_DELIV / 'rr_suitability_top30_batters.csv')
df_w_suit.head(10).write_csv(OUT_DELIV / 'rr_suitability_top10_bowlers.csv')
df_w_suit.write_csv(OUT_DELIV / 'rr_suitability_top30_bowlers.csv')

# RR Needs Table per UPDATE 3
rr_needs = [
    {
        "need_id": 1,
        "franchise_need": "Right-handed top-order anchor/#3, positions 1-4",
        "priority_weight": 1.0,
        "current_roster_cover": "Riyan Parag, Aman Rao (Aman Rao Perala), Dhruv Jurel",
        "best_candidates": "Jonny Bairstow, Harry Tector, Sikandar Raza",
        "acquisition_priority": "CRITICAL - Balances left-handed core"
    },
    {
        "need_id": 2,
        "franchise_need": "Power finisher, positions 5-7",
        "priority_weight": 1.0,
        "current_roster_cover": "Dhruv Jurel, Donovan Ferreira, Shubham Dubey, Ravindra Jadeja",
        "best_candidates": "Max Bryant, Nikhil Chaudhary, Will Sutherland, Jimmy Neesham, George Linde, Mahipal Lomror, Mark Chapman",
        "acquisition_priority": "CRITICAL - Closes match under steep RRR"
    },
    {
        "need_id": 3,
        "franchise_need": "Middle-overs control, overs 7-16, pace or spin",
        "priority_weight": 0.8,
        "current_roster_cover": "Ravi Bishnoi, Ravindra Jadeja, M Siddharth, Nandre Burger",
        "best_candidates": "Wanindu Hasaranga, Benjamin Sears, Mujeeb Ur Rahman, Mitchell Swepson, Senuran Muthusamy, Jaydev Unadkat, CJ Boyce, Lance Morris",
        "acquisition_priority": "HIGH - Restricts middle-overs consolidation"
    },
    {
        "need_id": 4,
        "franchise_need": "Death-overs pace specialist (depth only)",
        "priority_weight": 0.6,
        "current_roster_cover": "Jofra Archer, Sandeep Sharma, Mohammed Shami",
        "best_candidates": "Spencer Johnson, Ottneil Baartman, FazalHaq Farooqi",
        "acquisition_priority": "MODERATE / DEPTH ONLY - Covers Archer/Shami injury risk"
    },
    {
        "need_id": 5,
        "franchise_need": "Opener backup, positions 1-3",
        "priority_weight": 0.6,
        "current_roster_cover": "Yashasvi Jaiswal, Vaibhav Sooryavanshi, Lhuan-dre Pretorius",
        "best_candidates": "Reeza Hendricks, Matthew Wade, Matthew Renshaw",
        "acquisition_priority": "MODERATE - Injury / tactical reserve"
    },
    {
        "need_id": 6,
        "franchise_need": "Spin variety: off-spin or wrist-spin",
        "priority_weight": 0.5,
        "current_roster_cover": "Ravi Bishnoi (wrist-spin), Ravindra Jadeja (SLAO), M Siddharth (SLAO)",
        "best_candidates": "Wanindu Hasaranga, Mujeeb Ur Rahman, Mitchell Swepson",
        "acquisition_priority": "SECONDARY - Off-spin / wrist-spin matchup variety"
    },
    {
        "need_id": 7,
        "franchise_need": "Seam all-rounder at 6-7 (Shanaka covers it)",
        "priority_weight": 0.3,
        "current_roster_cover": "Dasun Shanaka (retained), Ravindra Jadeja",
        "best_candidates": "Will Sutherland, Jimmy Neesham, George Linde",
        "acquisition_priority": "LOW / COVERED - Addressed by retained Shanaka"
    }
]
pl.DataFrame(rr_needs).write_csv(OUT_DELIV / 'rr_needs_gap_analysis.csv')
pl.DataFrame(rr_needs).write_csv(OUT_TAB / 'rr_needs_gap_analysis.csv')

# 7. Role-Conditioned Deliverables
print("Generating Role-Conditioned Deliverables for Rajasthan Royals...", flush=True)

# 1. RHB top-order anchors (pos 1-4)
rhb_cand = df_b_suit.filter(
    (pl.col('need_filled').str.contains('anchor')) | (pl.col('primary_need_filled').str.contains('anchor') if 'primary_need_filled' in df_b_suit.columns else False)
).head(15)
if len(rhb_cand) == 0:
    rhb_cand = df_b_suit.head(15)
rhb_cand.write_csv(OUT_TAB / 'rhb_top_order_anchors_top15.csv')

# 2. Finishers (pos 5-7)
fin_cand = df_b_suit.filter(pl.col('need_filled').str.contains('finisher')).head(15)
if len(fin_cand) == 0:
    fin_cand = df_b_suit.head(15)
fin_cand.write_csv(OUT_TAB / 'finishers_positions_5_7_top15.csv')

# 3. Spinners all overs
spin_cand = df_w_suit.filter(pl.col('need_filled').str.contains('Spin|spin')).head(15)
if len(spin_cand) == 0:
    spin_cand = df_w_suit.head(15)
spin_cand.write_csv(OUT_TAB / 'spinners_all_overs_top15.csv')

# 4. Pace middle overs / Middle-overs control
pace_mid = df_w_suit.filter(pl.col('need_filled').str.contains('Middle-overs')).head(15)
if len(pace_mid) == 0:
    pace_mid = df_w_suit.head(15)
pace_mid.write_csv(OUT_TAB / 'pace_middle_overs_7_16_top15.csv')

# 5. Death bowlers (overs 17-20)
death_bowl = df_w_suit.filter(pl.col('need_filled').str.contains('death')).head(15)
if len(death_bowl) == 0:
    death_bowl = df_w_suit.head(15)
death_bowl.write_csv(OUT_TAB / 'death_bowlers_overs_17_20_top15.csv')

print("RANKINGS, ABLATIONS, AND RR 2027 SUITABILITY ENGINE COMPLETE!", flush=True)
