#!/usr/bin/env python3
"""
Step 3: Win Probability Model & Composite Pressure Index.
Fits regularised Logistic Regression (C=0.3) for Innings 1 and 2 with GroupKFold (5 splits),
match-balanced weights, Platt calibration, computes composite pressure index P = C * B * W * D,
applies global 25/50/25 thresholding across all 264,506 deliveries, and outputs diagnostics.
"""

from pathlib import Path
import json, hashlib, warnings
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

threadpool_limits(1)
warnings.filterwarnings('ignore')

DATA = Path('.')
OUT_MODEL = DATA / 'pipeline_outputs/modeling'
OUT_INTER = DATA / 'pipeline_outputs/intermediate'
OUT_TAB = DATA / 'pipeline_outputs/tables'
OUT_CONFIG = DATA / 'pipeline_outputs/config'
LOGS = DATA / 'pipeline_outputs/logs'

for p in [OUT_INTER, OUT_TAB, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

# 1. Verify Config Hash
CFG_PATH = OUT_CONFIG / 'model_config.json'
HASH_PATH = OUT_CONFIG / 'model_config.sha256'
cfg_bytes = CFG_PATH.read_bytes()
SHA = hashlib.sha256(cfg_bytes).hexdigest()
EXPECT = HASH_PATH.read_text().split()[0]
assert SHA == EXPECT, (SHA, EXPECT)
print(f"PRESSURE MODEL CONFIG_SHA256_VERIFIED: {SHA}", flush=True)

# 2. Load Combined Delivery Base
df_pl = pl.read_parquet(OUT_MODEL / 'combined_delivery_base.parquet')
N = len(df_pl)
print(f"Loaded combined delivery base: {N:,} rows across {df_pl['match_group_id'].n_unique()} matches.", flush=True)

# Extract numpy arrays for modeling
MATCH = df_pl['match_group_id'].to_numpy()
Y = df_pl['win_label'].to_numpy().astype(int)
INN = df_pl['innings'].to_numpy().astype(int)
COMP = df_pl['competition'].to_numpy()
VENUE = df_pl['venue'].to_numpy()
BALLS_LEFT = df_pl['balls_left_before'].to_numpy().astype(float)
LIMIT_BALLS = df_pl['innings_limit_balls'].to_numpy().astype(float)
WICKETS_BEFORE = df_pl['wickets_before'].to_numpy().astype(float)
SCORE_BEFORE = df_pl['score_before'].to_numpy().astype(float)
ELAPSED = df_pl['legal_elapsed_before'].to_numpy().astype(float)
TARGET = df_pl['target'].to_numpy().astype(float)
BALL_NUM = df_pl['innings_ball_num'].to_numpy().astype(int)
FIRST_TOTAL = df_pl['first_innings_total'].to_numpy().astype(float)
MODEL_OBS = df_pl['model_observation'].to_numpy().astype(bool)

def get_match_weights(indices):
    m_sub = MATCH[indices]
    unique, counts = np.unique(m_sub, return_counts=True)
    cnt_map = dict(zip(unique, counts))
    w = np.array([1.0 / cnt_map[m] for m in m_sub], dtype=float)
    return w * (len(w) / np.sum(w))

# 3. Build Resource Reference Table from Training Folds
def build_reference(train_indices):
    idx = np.asarray(train_indices, dtype=int)
    mask = (INN[idx] == 1) & MODEL_OBS[idx]
    train_inn1 = idx[mask]
    
    stage_bins = np.clip(np.floor(10.0 * ELAPSED[train_inn1] / np.maximum(LIMIT_BALLS[train_inn1], 1.0)), 0, 9).astype(int)
    wk_ref = np.clip(WICKETS_BEFORE[train_inn1], 0, 9).astype(int)
    rem_runs = FIRST_TOTAL[train_inn1] - SCORE_BEFORE[train_inn1]
    
    ref_df = pl.DataFrame({
        'match_id': MATCH[train_inn1],
        'competition': COMP[train_inn1],
        'venue': VENUE[train_inn1],
        'stage_bin': stage_bins,
        'wickets_lost_ref': wk_ref,
        'remaining_runs': rem_runs,
        'first_total': FIRST_TOTAL[train_inn1]
    })
    
    mc = ref_df.group_by(['match_id', 'competition', 'stage_bin', 'wickets_lost_ref']).agg(pl.col('remaining_runs').mean())
    
    levels = [
        ('exact', ['competition', 'stage_bin', 'wickets_lost_ref']),
        ('stage_wicket', ['stage_bin', 'wickets_lost_ref']),
        ('comp_stage', ['competition', 'stage_bin']),
        ('stage', ['stage_bin'])
    ]
    
    stats = {}
    owns = {}
    for nm, keys in levels:
        per_match = mc.group_by(['match_id'] + keys).agg(pl.col('remaining_runs').mean())
        per = per_match.group_by(keys).agg([
            pl.len().alias('count'),
            pl.col('remaining_runs').sum().alias('sum'),
            (pl.col('remaining_runs') ** 2).sum().alias('sum_sq')
        ])
        stats[nm] = per
        owns[nm] = per_match
        
    mt = ref_df.select(['match_id', 'venue', 'competition', 'first_total']).unique('match_id')
    vstats = mt.group_by('venue').agg([pl.len().alias('count'), pl.col('first_total').sum().alias('sum')])
    ownv = mt.select(['match_id', 'venue', pl.col('first_total').alias('own_total')])
    cstats = mt.group_by('competition').agg([pl.len().alias('c_count'), pl.col('first_total').sum().alias('c_sum')])
    
    return {
        'stats': stats, 'owns': owns, 'levels': levels,
        'vstats': vstats, 'ownv': ownv, 'cstats': cstats,
        'global_mean': float(mt['first_total'].mean() or 160.0),
        'stage_global_mean': float(mc['remaining_runs'].mean() or 80.0),
        'stage_global_sd': float(mc['remaining_runs'].std() or 20.0)
    }

def lookup_resource_and_venue(indices, ref, train_matches_set):
    idx = np.asarray(indices, dtype=int)
    n = len(idx)
    stage_bins = np.clip(np.floor(10.0 * ELAPSED[idx] / np.maximum(LIMIT_BALLS[idx], 1.0)), 0, 9).astype(int)
    wk_ref = np.clip(WICKETS_BEFORE[idx], 0, 9).astype(int)
    
    q_df = pl.DataFrame({
        'orig_idx': np.arange(n),
        'match_id': MATCH[idx],
        'competition': COMP[idx],
        'venue': VENUE[idx],
        'stage_bin': stage_bins,
        'wickets_lost_ref': wk_ref
    })
    
    om = np.full(n, np.nan)
    osd = np.full(n, np.nan)
    loo_mask = np.isin(MATCH[idx], list(train_matches_set))
    
    for nm, keys in ref['levels']:
        stat_df = ref['stats'][nm]
        own_df = ref['owns'][nm].rename({'remaining_runs': 'own_remaining'})
        
        merged = q_df.join(stat_df, on=keys, how='left').join(own_df, on=['match_id'] + keys, how='left')
        
        cnt = merged['count'].fill_null(0).to_numpy().astype(float)
        s_val = merged['sum'].fill_null(0).to_numpy().astype(float)
        sq_val = merged['sum_sq'].fill_null(0).to_numpy().astype(float)
        own_val = merged['own_remaining'].fill_null(0).to_numpy().astype(float)
        has_own = loo_mask & merged['own_remaining'].is_not_null().to_numpy()
        
        nn = cnt - has_own.astype(float)
        ss = s_val - np.where(has_own, own_val, 0.0)
        sq = sq_val - np.where(has_own, own_val ** 2, 0.0)
        
        valid = (nn >= 8) & np.isnan(om)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = np.where(nn > 0, ss / nn, np.nan)
            var = np.where(nn > 1, (sq - (ss * ss / np.maximum(nn, 1.0))) / (nn - 1.0), np.nan)
            sd = np.sqrt(np.maximum(var, 0.0))
            
        om[valid] = mu[valid]
        osd[valid] = sd[valid]
        
    bad = ~np.isfinite(om)
    om[bad] = ref['stage_global_mean']
    osd[bad] = ref['stage_global_sd']
    osd = np.maximum(np.where(np.isfinite(osd), osd, 20.0), 5.0)
    
    v_merged = q_df.join(ref['vstats'], on='venue', how='left').join(ref['ownv'], on=['match_id', 'venue'], how='left')
    v_cnt = v_merged['count'].fill_null(0).to_numpy().astype(float)
    v_sum = v_merged['sum'].fill_null(0).to_numpy().astype(float)
    v_own = v_merged['own_total'].fill_null(0).to_numpy().astype(float)
    v_has = loo_mask & v_merged['own_total'].is_not_null().to_numpy()
    
    v_nn = v_cnt - v_has.astype(float)
    v_ss = v_sum - np.where(v_has, v_own, 0.0)
    v_mu = np.where(v_nn > 0, v_ss / v_nn, np.nan)
    
    c_merged = q_df.join(ref['cstats'], on='competition', how='left')
    c_mu = np.where(c_merged['c_count'].fill_null(0).to_numpy() > 0,
                    c_merged['c_sum'].fill_null(0).to_numpy() / c_merged['c_count'].fill_null(1).to_numpy(),
                    ref['global_mean'])
                    
    venue_expected = np.where(np.isfinite(v_mu), v_mu, np.where(np.isfinite(c_mu), c_mu, ref['global_mean']))
    
    return om, osd, venue_expected

NUMS = {
    1: ['legal_balls_remaining', 'legal_over', 'wickets_lost', 'wickets_in_hand',
        'team_score', 'current_run_rate', 'log_balls_remaining', 'balls_left_x_wickets_in_hand',
        'expected_remaining_runs', 'stage_uncertainty_runs', 'resource_margin_z',
        'venue_expected_first_innings_score'],
    2: ['legal_balls_remaining', 'legal_over', 'wickets_lost', 'wickets_in_hand',
        'team_score', 'current_run_rate', 'log_balls_remaining', 'balls_left_x_wickets_in_hand',
        'expected_remaining_runs', 'stage_uncertainty_runs', 'resource_margin_z',
        'venue_expected_first_innings_score', 'runs_required', 'required_run_rate',
        'required_minus_current_run_rate', 'rrr_x_balls_left']
}
CATS = ['competition_type', 'venue_id']

def make_features_df(indices, ref, train_matches_set):
    idx = np.asarray(indices, dtype=int)
    inn = INN[idx]
    score = SCORE_BEFORE[idx]
    wk = WICKETS_BEFORE[idx]
    elapsed = ELAPSED[idx]
    limit = LIMIT_BALLS[idx]
    target = TARGET[idx]
    
    left = np.maximum(limit - elapsed, 0.0)
    over = np.floor(elapsed / 6.0).astype(float)
    wih = np.maximum(10.0 - wk, 0.0)
    crr = np.where(elapsed > 0, 6.0 * score / elapsed, 0.0)
    
    req = np.where(inn == 2, target - score, 0.0)
    rrr = np.where((inn == 2) & (left > 0), 6.0 * req / np.maximum(left, 1.0), 0.0)
    rrr_minus = np.where(inn == 2, rrr - crr, 0.0)
    
    er, sd, venue_exp = lookup_resource_and_venue(idx, ref, train_matches_set)
    margin = np.where(inn == 1, score + er - venue_exp, er - req)
    margin_z = margin / sd
    
    cols = {
        'legal_balls_remaining': left,
        'legal_over': over,
        'wickets_lost': wk,
        'wickets_in_hand': wih,
        'team_score': score,
        'current_run_rate': crr,
        'log_balls_remaining': np.log1p(left),
        'balls_left_x_wickets_in_hand': left * wih,
        'expected_remaining_runs': er,
        'stage_uncertainty_runs': sd,
        'resource_margin_z': margin_z,
        'venue_expected_first_innings_score': venue_exp,
        'runs_required': req,
        'required_run_rate': rrr,
        'required_minus_current_run_rate': rrr_minus,
        'rrr_x_balls_left': rrr * left,
        'competition_type': COMP[idx],
        'venue_id': VENUE[idx]
    }
    return pl.DataFrame(cols), margin_z

def make_lr_pipeline(inn):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, NUMS[inn]),
        ('cat', cat_pipe, CATS)
    ])
    return Pipeline([
        ('pre', pre),
        ('lr', LogisticRegression(C=0.3, solver='liblinear', max_iter=1500, random_state=42))
    ])

def fit_raw_inner_oof(indices, tag):
    idx = np.asarray(indices, dtype=int)
    groups = MATCH[idx]
    y_sub = Y[idx]
    out = np.full(len(idx), np.nan)
    loc = {int(orig): j for j, orig in enumerate(idx)}
    
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (trl, tel) in enumerate(cv.split(np.zeros(len(idx)), y_sub, groups=groups), 1):
        tr = idx[trl]
        te = idx[tel]
        tm = set(MATCH[tr])
        ref = build_reference(tr)
        xtr_df, _ = make_features_df(tr, ref, tm)
        xte_df, _ = make_features_df(te, ref, tm)
        
        for inn in [1, 2]:
            tri = tr[INN[tr] == inn]
            tei = te[INN[te] == inn]
            if len(tei) == 0:
                continue
                
            m = make_lr_pipeline(inn)
            cols = NUMS[inn] + CATS
            
            # Subselect rows
            sub_tr = xtr_df.filter(INN[tr] == inn).select(cols)
            sub_te = xte_df.filter(INN[te] == inn).select(cols)
            
            w_tri = get_match_weights(tri)
            m.fit(sub_tr, Y[tri], lr__sample_weight=w_tri)
            z = m.decision_function(sub_te)
            
            for original_pos, val in zip(tei, z):
                out[loc[int(original_pos)]] = val
                
    assert np.isfinite(out).all(), f"{tag} incomplete inner OOF"
    return out

# 4. Outer 5-fold GroupKFold CV
print("Beginning Outer 5-fold GroupKFold cross-validation...", flush=True)
outer = GroupKFold(n_splits=5, shuffle=True, random_state=42)
wp = np.full(N, np.nan)
margin_oof = np.full(N, np.nan)
folds_diag = []

for f, (tr, te) in enumerate(outer.split(np.zeros(N), Y, groups=MATCH), 1):
    tm = set(MATCH[tr])
    print(f"PRESSURE outer fold {f}/5: train_rows={len(tr):,} valid_rows={len(te):,}", flush=True)
    inner_raw = fit_raw_inner_oof(tr, f"outer{f}")
    
    # Platt calibration
    cal = LogisticRegression(C=1e6, solver='liblinear', max_iter=5000, random_state=42)
    cal.fit(inner_raw.reshape(-1, 1), Y[tr], sample_weight=get_match_weights(tr))
    
    ref = build_reference(tr)
    xtr_df, _ = make_features_df(tr, ref, tm)
    xte_df, mte = make_features_df(te, ref, tm)
    
    for inn in [1, 2]:
        tri = tr[INN[tr] == inn]
        tei = te[INN[te] == inn]
        if len(tei) == 0:
            continue
            
        m = make_lr_pipeline(inn)
        cols = NUMS[inn] + CATS
        
        sub_tr = xtr_df.filter(INN[tr] == inn).select(cols)
        sub_te = xte_df.filter(INN[te] == inn).select(cols)
        
        w_tri = get_match_weights(tri)
        m.fit(sub_tr, Y[tri], lr__sample_weight=w_tri)
        rawp = m.decision_function(sub_te)
        
        wp[tei] = cal.predict_proba(rawp.reshape(-1, 1))[:, 1]
        margin_oof[tei] = mte[INN[te] == inn]
        
    folds_diag.append({
        'fold': f, 'train_matches': len(tm), 'valid_matches': len(set(MATCH[te])),
        'train_rows': len(tr), 'valid_rows': len(te)
    })

assert np.isfinite(wp).all(), "Missing WP predictions"
assert np.isfinite(margin_oof).all(), "Missing resource margin values"

# 5. Composite Pressure Index Formula
closeness = np.clip(1.0 - 2.0 * np.abs(wp - 0.5), 0.0, 1.0)
B = 1.0 - BALLS_LEFT / LIMIT_BALLS
W = 1.0 - np.maximum(10.0 - WICKETS_BEFORE, 0.0) / 11.0
Dmd = 1.0 / (1.0 + np.exp(np.clip(margin_oof, -30.0, 30.0)))
score = closeness * B * W * Dmd

# 6. Global 25/50/25 Thresholding
ord_df = pl.DataFrame({
    'i': np.arange(N),
    'score': score,
    'match': MATCH,
    'innings': INN,
    'ball': BALL_NUM
}).sort(['score', 'match', 'innings', 'ball'])

sorted_i = ord_df['i'].to_numpy()
rank = np.empty(N, dtype=int)
rank[sorted_i] = np.arange(1, N + 1)
pct = rank / float(N)

pc = np.full(N, 'middle_excluded', dtype=object)
pc[pct <= 0.50] = 'ordinary'
pc[pct > 0.75] = 'high_pressure'

# 7. Create Pressure Labels Output DataFrame
pressure_df = pl.DataFrame({
    'match_group_id': MATCH,
    'innings': INN,
    'innings_ball_num': BALL_NUM,
    'competition': COMP,
    'wp_oof_before': wp,
    'resource_margin_z': margin_oof,
    'closeness': closeness,
    'ball_scarcity': B,
    'wicket_scarcity': W,
    'demand': Dmd,
    'composite_pressure_index': score,
    'global_rank': rank,
    'global_percentile': pct,
    'pressure_class': pc
})

# Save intermediate labels
pressure_df.write_parquet(OUT_INTER / 'combined_oof_pressure_labels.parquet')
pressure_df.write_csv(OUT_INTER / 'combined_oof_pressure_labels.csv.gz', compression='gzip')
pl.DataFrame(folds_diag).write_csv(OUT_TAB / 'pressure_outer_fold_diagnostics.csv')

# 8. Evaluation Metrics
metrics = []
for comp_name in ['ALL'] + sorted(np.unique(COMP)):
    if comp_name == 'ALL':
        idx = np.arange(N)
    else:
        idx = np.flatnonzero(COMP == comp_name)
    w_sub = get_match_weights(idx)
    brier = float(brier_score_loss(Y[idx], wp[idx], sample_weight=w_sub))
    ll = float(log_loss(Y[idx], np.clip(wp[idx], 1e-7, 1.0 - 1e-7), labels=[0, 1], sample_weight=w_sub))
    metrics.append({
        'competition': comp_name,
        'rows': len(idx),
        'matches': int(len(np.unique(MATCH[idx]))),
        'brier': brier,
        'log_loss': ll
    })
pl.DataFrame(metrics).write_csv(OUT_TAB / 'pressure_wp_oof_metrics.csv')

# Counts by competition and innings
counts_comp = (
    pressure_df.group_by(['competition', 'pressure_class'])
    .agg(pl.len().alias('deliveries'))
    .sort(['competition', 'pressure_class'])
)
counts_comp.write_csv(OUT_TAB / 'pressure_label_counts_by_competition.csv')

counts_inn = (
    pressure_df.group_by(['innings', 'pressure_class'])
    .agg(pl.len().alias('deliveries'))
    .sort(['innings', 'pressure_class'])
)
counts_inn.write_csv(OUT_TAB / 'pressure_label_counts_by_innings.csv')

# 9. Shift on Legacy IPL/T20I Rows
legacy_mask = (df_pl['source_type'] == 'legacy').to_numpy()
leg_pc = pc[legacy_mask]
legacy_hp = int(np.sum(leg_pc == 'high_pressure'))
legacy_ord = int(np.sum(leg_pc == 'ordinary'))
legacy_mid = int(np.sum(leg_pc == 'middle_excluded'))
print(f"Legacy 190,763 rows label distribution under 5-competition global threshold:")
print(f"  High Pressure: {legacy_hp:,} (vs original 47,691)")
print(f"  Ordinary: {legacy_ord:,} (vs original 95,381)")
print(f"  Middle Excluded: {legacy_mid:,} (vs original 47,691)")

summary = {
    'config_sha256': SHA,
    'total_rows': N,
    'total_matches': int(len(np.unique(MATCH))),
    'high_pressure': int(np.sum(pc == 'high_pressure')),
    'ordinary': int(np.sum(pc == 'ordinary')),
    'middle_excluded': int(np.sum(pc == 'middle_excluded')),
    'high_pressure_cutoff_min_score': float(np.min(score[pc == 'high_pressure'])),
    'ordinary_cutoff_max_score': float(np.max(score[pc == 'ordinary'])),
    'wp_oof_mean': float(np.mean(wp)),
    'wp_oof_std': float(np.std(wp)),
    'resource_margin_z_mean': float(np.mean(margin_oof)),
    'resource_margin_z_sd': float(np.std(margin_oof)),
    'legacy_shift': {
        'high_pressure': legacy_hp,
        'ordinary': legacy_ord,
        'middle_excluded': legacy_mid
    }
}
with open(OUT_INTER / 'pressure_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("PRESSURE MODEL COMPLETE!")
print(json.dumps(summary, indent=2))
