#!/usr/bin/env python3
"""
Step 4: Expected Value & Player Value Models.
Fits 4 LightGBM models (expected batter runs, batter dismissal prob, expected bowler runs,
bowler credited wicket prob) plus state-specific remaining runs (wicket cost lambda) using
5-fold GroupKFold by match_group_id on the common 8 features.
Runs paired before/after feature comparison on legacy IPL/T20I.
Assigns ball-level BV (batting) and BoV (bowling) value contributions.
"""

from pathlib import Path
import json, hashlib, warnings
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, brier_score_loss, log_loss, roc_auc_score
from lightgbm import LGBMRegressor, LGBMClassifier
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
print(f"VALUE MODELS CONFIG_SHA256_VERIFIED: {SHA}", flush=True)

# 2. Load Combined Delivery Base & Pressure Labels
d = pl.read_parquet(OUT_MODEL / 'combined_delivery_base.parquet')
p = pl.read_parquet(OUT_INTER / 'combined_oof_pressure_labels.parquet')

assert len(d) == len(p) == 264506, f"Length mismatch: {len(d)} vs {len(p)}"
assert (d['match_group_id'] == p['match_group_id']).all()
assert (d['innings'] == p['innings']).all()
assert (d['innings_ball_num'] == p['innings_ball_num']).all()
print(f"Loaded {len(d):,} deliveries with verified pressure labels.", flush=True)

# Attach pressure columns
d = d.with_columns([
    p['pressure_class'].alias('pressure_class'),
    p['composite_pressure_index'].alias('composite_pressure_index'),
    p['wp_oof_before'].alias('wp_oof_before')
])

# Ensure correct data types
d = d.with_columns([
    pl.col('match_group_id').cast(pl.Utf8).alias('match_id'),
    pl.col('competition').cast(pl.Utf8).alias('competition_type'),
    pl.col('venue').cast(pl.Utf8),
    pl.col('innings').cast(pl.Int32),
    pl.col('over').cast(pl.Float64),
    pl.col('wickets_lost').cast(pl.Float64),
    pl.col('balls_left').cast(pl.Float64),
    pl.col('required_rate').cast(pl.Float64),
    pl.col('opposition_elo_batter').fill_null(1500.0).cast(pl.Float64),
    pl.col('opposition_elo_bowler').fill_null(1500.0).cast(pl.Float64),
    pl.col('batter_faced').cast(pl.Boolean),
    pl.col('legal_ball').cast(pl.Boolean),
    pl.col('batter_dismissal_target').cast(pl.Int32),
    pl.col('bowler_credited_wicket_target').cast(pl.Int32),
    pl.col('is_wicket_bool').cast(pl.Boolean),
    pl.col('batter_runs').cast(pl.Float64),
    pl.col('bowler_charged_runs').cast(pl.Float64),
    pl.col('remaining_innings_runs_target').cast(pl.Float64),
    pl.col('source_type').cast(pl.Utf8)
])

# LightGBM parameters from config
PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': 1,
    'verbose': -1
}

NUM = ['innings', 'over', 'wickets_lost', 'balls_left', 'required_rate', 'opposition_elo']
CAT = ['competition_type', 'venue']
CONTEXT_NUM = ['innings', 'over', 'wickets_lost', 'balls_left', 'required_rate']

def make_model_X(frame, elo_col):
    df_sub = frame.select(CONTEXT_NUM + CAT)
    elo = frame[elo_col].fill_null(1500.0).alias('opposition_elo')
    return df_sub.with_columns(elo).select(NUM + CAT)

def build_pipe(kind, num_cols, cat_cols):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    pre = ColumnTransformer([('num', numeric, num_cols), ('cat', categorical, cat_cols)], remainder='drop')
    mdl = LGBMRegressor(objective='regression_l2', **PARAMS) if kind == 'regression' else LGBMClassifier(objective='binary', **PARAMS)
    return Pipeline([('pre', pre), ('model', mdl)])

def get_match_weights(groups_arr):
    unique, counts = np.unique(groups_arr, return_counts=True)
    cnt_map = dict(zip(unique, counts))
    w = np.array([1.0 / cnt_map[g] for g in groups_arr], dtype=float)
    return w * (len(w) / np.sum(w))

def metric_row(name, kind, y, pred, w, comp):
    if kind == 'regression':
        return {
            'model': name, 'competition': comp, 'n': len(y),
            'mae': float(mean_absolute_error(y, pred, sample_weight=w)),
            'rmse': float(mean_squared_error(y, pred, sample_weight=w) ** 0.5),
            'r2': float(r2_score(y, pred, sample_weight=w)),
            'brier': np.nan, 'log_loss': np.nan, 'roc_auc': np.nan
        }
    pp = np.clip(pred, 1e-7, 1.0 - 1e-7)
    auc = roc_auc_score(y, pp, sample_weight=w) if len(np.unique(y)) == 2 else np.nan
    return {
        'model': name, 'competition': comp, 'n': len(y),
        'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
        'brier': float(brier_score_loss(y, pp, sample_weight=w)),
        'log_loss': float(log_loss(y, pp, labels=[0, 1], sample_weight=w)),
        'roc_auc': float(auc) if np.isfinite(auc) else np.nan
    }

specs = {
    'expected_batter_runs': ('regression', 'batter_runs', 'batter_faced', 'opposition_elo_batter'),
    'batter_dismissal_probability': ('classification', 'batter_dismissal_target', None, 'opposition_elo_batter'),
    'expected_bowler_charged_runs': ('regression', 'bowler_charged_runs', None, 'opposition_elo_bowler'),
    'bowler_credited_wicket_probability': ('classification', 'bowler_credited_wicket_target', None, 'opposition_elo_bowler')
}

groups = d['match_id'].to_numpy()
gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(d), -1, dtype=int)
for f, (_, va) in enumerate(gkf.split(np.zeros(len(d)), groups=groups)):
    folds[va] = f
assert (folds >= 0).all()

preds = {}
production_metrics = []
foldrows = []

# Fit 4 production models
for name, (kind, target_col, maskcol, elo_col) in specs.items():
    print(f"PRODUCTION MODEL START: {name}", flush=True)
    X = make_model_X(d, elo_col)
    y = d[target_col].to_numpy().astype(float)
    mask = np.ones(len(d), bool) if maskcol is None else d[maskcol].to_numpy().astype(bool)
    pr = np.full(len(d), np.nan)
    
    for f in range(5):
        tr = np.flatnonzero((folds != f) & mask & np.isfinite(y))
        va = np.flatnonzero((folds == f) & mask & np.isfinite(y))
        
        pipe = build_pipe(kind, NUM, CAT)
        w = get_match_weights(groups[tr])
        
        # Subselect rows using Polars filter
        X_tr = X[tr]
        X_va = X[va]
        
        pipe.fit(X_tr, y[tr], model__sample_weight=w)
        if kind == 'regression':
            pr[va] = pipe.predict(X_va)
        else:
            pr[va] = pipe.predict_proba(X_va)[:, 1]
            
        foldrows.append({
            'model': name, 'fold': f, 'train_rows': len(tr), 'valid_rows': len(va),
            'train_matches': len(np.unique(groups[tr])), 'valid_matches': len(np.unique(groups[va]))
        })
        print(f"  {name} fold {f+1}/5 complete", flush=True)
        
    assert np.isfinite(pr[mask]).all(), f"NaN in predictions for {name}"
    if kind == 'classification':
        pr = np.clip(pr, 1e-7, 1.0 - 1e-7)
    preds[name] = pr
    
    # Compute metrics
    production_metrics.append(metric_row(name, kind, y[mask], pr[mask], get_match_weights(groups[mask]), 'ALL'))
    comp_arr = d['competition_type'].to_numpy()
    for comp in sorted(np.unique(comp_arr)):
        ix = mask & (comp_arr == comp)
        if np.sum(ix) > 0:
            production_metrics.append(metric_row(name, kind, y[ix], pr[ix], get_match_weights(groups[ix]), comp))

# Fit Wicket-cost (lambda) model
print("WICKET-COST MODEL START", flush=True)
XL = make_model_X(d, 'opposition_elo_batter')
yl = d['remaining_innings_runs_target'].to_numpy().astype(float)
lambda_pred = np.full(len(d), np.nan)
lambda_more = np.full(len(d), np.nan)

for f in range(5):
    tr = np.flatnonzero(folds != f)
    va = np.flatnonzero(folds == f)
    pipe = build_pipe('regression', NUM, CAT)
    w = get_match_weights(groups[tr])
    
    pipe.fit(XL[tr], yl[tr], model__sample_weight=w)
    lambda_pred[va] = pipe.predict(XL[va])
    
    # Predict with wickets_lost + 1
    XL_more = XL[va].with_columns(
        (pl.col('wickets_lost') + 1.0).clip(0.0, 10.0).alias('wickets_lost')
    )
    lambda_more[va] = pipe.predict(XL_more)
    print(f"  Wicket cost fold {f+1}/5 complete", flush=True)

wk_lost = d['wickets_lost'].to_numpy()
lambda_state = np.maximum(0.0, lambda_pred - lambda_more)
lambda_state[wk_lost >= 10.0] = 0.0

# Paired before/after comparison on legacy IPL/T20I deliveries
print("PAIRED BEFORE/AFTER FEATURE COMPARISON START", flush=True)
OLD_NUM = ['innings', 'legal_over_before', 'balls_left_before', 'wickets_before', 'wickets_in_hand', 'required_run_rate']
OLD_CAT = ['batter_hand', 'pace_or_spin', 'bowling_type', 'batting_type', 'venue', 'competition_type']

legacy_mask = (d['source_type'] == 'legacy') & d['competition_type'].is_in(['IPL', 'T20I'])
legacy = d.filter(legacy_mask)
legacy_groups = legacy['match_id'].to_numpy()

lgkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
lfold = np.full(len(legacy), -1, dtype=int)
for f, (_, va) in enumerate(lgkf.split(np.zeros(len(legacy)), groups=legacy_groups)):
    lfold[va] = f

# Prepare old features
bbb = pl.read_csv('inputs/data_pack/Data Pack/ball_by_ball.csv', columns=['match_id', 'innings', 'overid', 'over_ball_num', 'bowling_type']).with_columns([
    pl.col('match_id').cast(pl.Utf8).alias('source_match_id'),
    pl.col('innings').cast(pl.Int32),
    pl.col('overid').cast(pl.Int32),
    pl.col('over_ball_num').cast(pl.Int32)
])
legacy = legacy.join(bbb, on=['source_match_id', 'innings', 'overid', 'over_ball_num'], how='left')
legacy = legacy.with_columns([
    (10.0 - pl.col('wickets_before')).clip(0.0, 10.0).alias('wickets_in_hand'),
    pl.col('batter_hand').fill_null('UNK').cast(pl.Utf8),
    pl.col('pace_or_spin').fill_null('UNK').cast(pl.Utf8),
    pl.col('bowling_type').fill_null('UNK').cast(pl.Utf8),
    pl.col('batter_hand').fill_null('UNK').cast(pl.Utf8).alias('batting_type')
])

comparison_rows = []
for name, (kind, target_col, maskcol, elo_col) in specs.items():
    Xnew = make_model_X(legacy, elo_col)
    Xold = legacy.select(OLD_NUM + OLD_CAT)
    y_leg = legacy[target_col].to_numpy().astype(float)
    mask_leg = np.ones(len(legacy), bool) if maskcol is None else legacy[maskcol].to_numpy().astype(bool)
    
    for version, X_feat, num_c, cat_c in [('previous_richer', Xold, OLD_NUM, OLD_CAT), ('reduced_common', Xnew, NUM, CAT)]:
        pr_leg = np.full(len(legacy), np.nan)
        print(f"  Before/after: {name} ({version})", flush=True)
        for f in range(5):
            tr = np.flatnonzero((lfold != f) & mask_leg & np.isfinite(y_leg))
            va = np.flatnonzero((lfold == f) & mask_leg & np.isfinite(y_leg))
            pipe = build_pipe(kind, num_c, cat_c)
            w = get_match_weights(legacy_groups[tr])
            pipe.fit(X_feat[tr], y_leg[tr], model__sample_weight=w)
            if kind == 'regression':
                pr_leg[va] = pipe.predict(X_feat[va])
            else:
                pr_leg[va] = pipe.predict_proba(X_feat[va])[:, 1]
                
        if kind == 'classification':
            pr_leg = np.clip(pr_leg, 1e-7, 1.0 - 1e-7)
            
        leg_comps = legacy['competition_type'].to_numpy()
        for comp in ['ALL', 'IPL', 'T20I']:
            ix = mask_leg if comp == 'ALL' else (mask_leg & (leg_comps == comp))
            r = metric_row(name, kind, y_leg[ix], pr_leg[ix], get_match_weights(legacy_groups[ix]), comp)
            r['feature_set'] = version
            comparison_rows.append(r)

pl.DataFrame(comparison_rows).write_csv(OUT_TAB / 'ipl_t20i_feature_before_after_metrics.csv')

# Compute paired deltas: reduced_common minus previous_richer
cmp_df = pl.DataFrame(comparison_rows)
delta_rows = []
metric_cols = ['mae', 'rmse', 'r2', 'brier', 'log_loss', 'roc_auc']

for (m, c), g in cmp_df.group_by(['model', 'competition']):
    r = {'model': m, 'competition': c}
    old_row = g.filter(pl.col('feature_set') == 'previous_richer')
    new_row = g.filter(pl.col('feature_set') == 'reduced_common')
    if len(old_row) > 0 and len(new_row) > 0:
        for col in metric_cols:
            old_v = old_row[col][0]
            new_v = new_row[col][0]
            r[col + '_previous_richer'] = old_v
            r[col + '_reduced_common'] = new_v
            r[col + '_delta_reduced_minus_previous'] = (new_v - old_v) if (old_v is not None and new_v is not None) else None
        delta_rows.append(r)
pl.DataFrame(delta_rows).write_csv(OUT_TAB / 'ipl_t20i_feature_before_after_deltas.csv')

# Assign Player Value Contributions
print("ASSIGNING BATTER & BOWLER VALUE CONTRIBUTIONS", flush=True)
p_dismissal = np.clip(preds['batter_dismissal_probability'], 1e-7, 1.0 - 1e-7)
p_wicket = np.clip(preds['bowler_credited_wicket_probability'], 1e-7, 1.0 - 1e-7)

batter_runs = d['batter_runs'].to_numpy()
bowler_runs = d['bowler_charged_runs'].to_numpy()
faced = d['batter_faced'].to_numpy()
is_dismissal = d['batter_dismissal_target'].to_numpy().astype(bool)
bowler_wicket = d['bowler_credited_wicket_target'].to_numpy()
has_dismissed_id = d['dismissed_player_id_model'].is_not_null().to_numpy()

batter_run_residual = np.where(faced, batter_runs - preds['expected_batter_runs'], 0.0)
striker_bv = batter_run_residual + np.where((~is_dismissal) & faced, lambda_state * p_dismissal, 0.0)
dismissed_bv = np.where(is_dismissal & has_dismissed_id, -lambda_state * (1.0 - p_dismissal), 0.0)
bowler_bov = (preds['expected_bowler_charged_runs'] - bowler_runs) + lambda_state * (bowler_wicket - p_wicket)

# Create Output Predictions DataFrame
prd = pl.DataFrame({
    'match_id': d['match_id'],
    'match_group_id': d['match_group_id'],
    'competition': d['competition'],
    'innings': d['innings'],
    'innings_ball_num': d['innings_ball_num'],
    'striker_player_id': d['striker_player_id'],
    'bowler_player_id': d['bowler_player_id'],
    'dismissed_player_id_model': d['dismissed_player_id_model'],
    'pressure_class': d['pressure_class'],
    'batter_faced': d['batter_faced'],
    'legal_ball': d['legal_ball'],
    'expected_batter_runs': preds['expected_batter_runs'],
    'batter_dismissal_probability': preds['batter_dismissal_probability'],
    'expected_bowler_charged_runs': preds['expected_bowler_charged_runs'],
    'bowler_credited_wicket_probability': preds['bowler_credited_wicket_probability'],
    'lambda_state': lambda_state,
    'striker_bv_contribution': striker_bv,
    'dismissed_bv_contribution': dismissed_bv,
    'bowler_bov_contribution': bowler_bov
})

prd.write_parquet(OUT_INTER / 'combined_oof_expected_value_predictions.parquet')
prd.write_csv(OUT_INTER / 'combined_oof_expected_value_predictions.csv.gz', compression='gzip')

pl.DataFrame(production_metrics).write_csv(OUT_TAB / 'combined_oof_expected_value_metrics.csv')
pl.DataFrame(foldrows).write_csv(OUT_TAB / 'combined_oof_expected_value_fold_diagnostics.csv')

lambda_summary = {
    'rows': len(d),
    'matches': int(d['match_id'].n_unique()),
    'mae': float(mean_absolute_error(yl, lambda_pred)),
    'rmse': float(mean_squared_error(yl, lambda_pred) ** 0.5),
    'lambda_mean': float(np.mean(lambda_state)),
    'lambda_median': float(np.median(lambda_state)),
    'lambda_p25': float(np.quantile(lambda_state, 0.25)),
    'lambda_p75': float(np.quantile(lambda_state, 0.75)),
    'lambda_p90': float(np.quantile(lambda_state, 0.90)),
    'lambda_max': float(np.max(lambda_state)),
    'zero_lambda_share': float(np.mean(lambda_state == 0.0))
}
with open(OUT_INTER / 'combined_lambda_summary.json', 'w') as f:
    json.dump(lambda_summary, f, indent=2)

print("VALUE MODELS COMPLETE!", flush=True)
print(json.dumps(lambda_summary, indent=2))
