#!/usr/bin/env python3
"""
Step 5: Empirical Bayes Rating Engine.
Implements cluster-robust sampling variance with pooled per-ball variance and design effect floors,
DerSimonian-Laird tau^2 estimation with competition fixed effects, EB-1 mu0 fallback,
zero-tau^2 policy (bowler 100% pressure value if delta tau^2=0), competition recentering,
acceptance checks EB-1 through EB-5, and SMAT exposure audit.
"""

from pathlib import Path
import json, hashlib, math, warnings
import numpy as np
import polars as pl
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

DATA = Path('.')
OUT_MODEL = DATA / 'pipeline_outputs/modeling'
OUT_INTER = DATA / 'pipeline_outputs/intermediate'
OUT_TAB = DATA / 'pipeline_outputs/tables'
OUT_DELIV = DATA / 'pipeline_outputs/deliverables'
OUT_CONFIG = DATA / 'pipeline_outputs/config'
LOGS = DATA / 'pipeline_outputs/logs'

for p in [OUT_INTER, OUT_TAB, OUT_DELIV, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

# 1. Verify Config Hash
CFG_PATH = OUT_CONFIG / 'model_config.json'
HASH_PATH = OUT_CONFIG / 'model_config.sha256'
cfg_bytes = CFG_PATH.read_bytes()
SHA = hashlib.sha256(cfg_bytes).hexdigest()
EXPECT = HASH_PATH.read_text().split()[0]
assert SHA == EXPECT, (SHA, EXPECT)
print(f"EB MODEL CONFIG_SHA256_VERIFIED: {SHA}", flush=True)

# 2. Load Predictions & Delivery Data
prd = pl.read_parquet(OUT_INTER / 'combined_oof_expected_value_predictions.parquet')
print(f"Loaded {len(prd):,} prediction rows.", flush=True)

# Load Name Map
with open(OUT_MODEL / 'identity_name_map.json', 'r') as f:
    name_map = json.load(f)

# Augment canonical labels from mapping file
mapping_file = DATA / 'inputs/data_pack/Data Pack/competition_team_player_mapping.csv'
if not mapping_file.exists():
    mapping_file = DATA / 'inputs/canonical_csvs/competition_team_player_mapping.csv'
if not mapping_file.exists():
    mapping_file = DATA / 'inputs/competition_team_player_mapping.csv'

mp_df = pl.read_csv(mapping_file)
for r in mp_df.filter(pl.col('player_id').is_not_null() & pl.col('player_name').is_not_null()).iter_rows(named=True):
    name_map.setdefault(f"PID_{int(r['player_id'])}", str(r['player_name']))

# Build Exclusion Set (IPL 2026 squads + Pakistan)
# Load exclusion list from superr_outside_ipl2026 if available, or build from mapping
excl_file = DATA / 'superr_phase0_audit_bundle/superr_phase0/audit_output/final_exclusion_list_by_player_id.csv'
if not excl_file.exists():
    excl_file = DATA / 'reference/gate2/superr_gate2_2023_2026/identity/final_exclusion_list_by_player_id.csv'
if excl_file.exists():
    ex_df = pl.read_csv(excl_file)
    excluded_canonical = {int(x) for x in ex_df['player_id'].to_numpy()}
    print(f"Loaded {len(excluded_canonical)} excluded canonical player IDs from audit file.", flush=True)
else:
    # Build from mapping: Pakistan + Pakistan A + IPL 2026
    pak_ids = set(mp_df.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().is_in(['pakistan', 'pakistan a']))['player_id'].drop_nulls().cast(pl.Int64).to_list())
    ipl26_ids = set(mp_df.filter(pl.col('comp_name').cast(pl.Utf8) == 'IPL 2026')['player_id'].drop_nulls().cast(pl.Int64).to_list())
    excluded_canonical = pak_ids | ipl26_ids
    excluded_canonical.add(6628) # Gerald Coetzee
    print(f"Built {len(excluded_canonical)} excluded canonical player IDs from mapping.", flush=True)

excluded = {f"PID_{int(x)}" for x in excluded_canonical}

# Add Pakistan registry IDs from Men's T20I archive
import zipfile, yaml
t20s_zip = DATA / 't20s.zip'
if not t20s_zip.exists():
    t20s_zip = DATA / 'inputs/cricsheet_archives/t20s.zip'
if not t20s_zip.exists():
    t20s_zip = DATA / 'inputs/t20s.zip'
if not t20s_zip.exists():
    t20s_zip = DATA / 'inputs/t20s.zip'

pak_registry = set()
if t20s_zip.exists():
    with zipfile.ZipFile(t20s_zip) as zf:
        for fname in zf.namelist():
            if not fname.lower().endswith(('.yaml', '.yml')):
                continue
            raw = zf.read(fname).decode('utf-8', errors='replace')
            cut = raw.find('\ninnings:')
            if cut < 0: cut = raw.find('\ninnings :')
            prefix = raw[:cut] if cut >= 0 else raw
            try:
                obj = yaml.load(prefix, Loader=yaml.CSafeLoader) or {}
            except Exception:
                continue
            info = obj.get('info', {}) or {}
            dates = info.get('dates') or []
            dt = str(dates[0])[:10] if dates else ''
            if not ('2022-01-01' <= dt <= '2026-12-31') or info.get('gender') != 'male' or info.get('match_type') != 'T20':
                continue
            pls = info.get('players', {}) or {}
            pak_players = pls.get('Pakistan', []) or []
            reg = (info.get('registry', {}) or {}).get('people', {}) or {}
            for nm in pak_players:
                rid = reg.get(nm)
                if rid:
                    pak_registry.add(str(rid))
    print(f"Extracted {len(pak_registry)} Pakistan registry IDs from T20I archive.", flush=True)

# Add Pakistani CS_ IDs to excluded
for rid in pak_registry:
    excluded.add(f"CS_{rid}")

# Load released players mapping (76 players released by IPL franchises)
rel_file = OUT_CONFIG / 'released_players_mapping.csv'
if rel_file.exists():
    rel_df = pl.read_csv(rel_file)
    released_pids = set(rel_df['formatted_id'].to_list())
    print(f"Loaded {len(released_pids)} released player IDs to remove from exclusion set.", flush=True)
else:
    released_pids = set()

# Remove released players from exclusion set
excluded = excluded - released_pids
print(f"Exclusion set size after removing {len(released_pids)} released players: {len(excluded)}", flush=True)

# Explicit user directive: Exclude retired players from final selectable pools (Nabi, Russell, Dussen, Siddle, Shamsi, Karthik, Warner)
retired_players = {"PID_1107", "PID_1318", "PID_2930", "CS_bd93fea4", "PID_4237", "PID_143", "PID_220"}
excluded.update(retired_players)
print(f"Exclusion set size after enforcing retired exclusions: {len(excluded)}", flush=True)

# Explicit user directive: Exclude all Bangladeshi players from both ranking lists and deliverables
bd_canonical = set(mp_df.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().str.contains('bangladesh'))['player_id'].drop_nulls().cast(pl.Int64).to_list())
for pid in bd_canonical:
    excluded.add(f"PID_{pid}")

bd_registry = set()
if t20s_zip.exists():
    with zipfile.ZipFile(t20s_zip) as zf:
        for fname in zf.namelist():
            if not fname.lower().endswith(('.yaml', '.yml')):
                continue
            raw = zf.read(fname).decode('utf-8', errors='replace')
            cut = raw.find('\ninnings:')
            if cut < 0: cut = raw.find('\ninnings :')
            prefix = raw[:cut] if cut >= 0 else raw
            try:
                obj = yaml.load(prefix, Loader=yaml.CSafeLoader) or {}
            except Exception:
                continue
            info = obj.get('info', {}) or {}
            pls = info.get('players', {}) or {}
            bd_players = pls.get('Bangladesh', []) or []
            reg = (info.get('registry', {}) or {}).get('people', {}) or {}
            for nm in bd_players:
                rid = reg.get(nm)
                if rid:
                    bd_registry.add(str(rid))
    print(f"Extracted {len(bd_registry)} Bangladesh registry IDs from T20I archive.", flush=True)

for rid in bd_registry:
    excluded.add(f"CS_{rid}")

print(f"Final exclusion set size after enforcing Bangladesh exclusions: {len(excluded)}", flush=True)

# 3. Create Batter and Bowler Events
print("Building event and exposure tables...", flush=True)
common = ['match_id', 'competition', 'pressure_class', 'innings', 'innings_ball_num']

# Striker events
ev_cols = ['match_id', 'competition_type', 'pressure_class', 'innings', 'innings_ball_num', 'player_id', 'value_contribution', 'exposure_flag', 'dismissed_only']

striker = prd.select(common + ['striker_player_id', 'batter_faced', 'striker_bv_contribution']).rename({
    'striker_player_id': 'player_id',
    'batter_faced': 'exposure_flag',
    'striker_bv_contribution': 'value_contribution',
    'competition': 'competition_type'
}).with_columns([
    pl.lit(False).alias('dismissed_only')
]).select(ev_cols)

# Dismissed events
dismissed = prd.filter(
    (pl.col('dismissed_player_id_model').is_not_null()) & (pl.col('dismissed_player_id_model') != '')
).select(common + ['dismissed_player_id_model', 'dismissed_bv_contribution']).rename({
    'dismissed_player_id_model': 'player_id',
    'dismissed_bv_contribution': 'value_contribution',
    'competition': 'competition_type'
}).with_columns([
    pl.lit(False).alias('exposure_flag'),
    pl.lit(True).alias('dismissed_only')
]).select(ev_cols)

batter_events = pl.concat([striker, dismissed]).with_columns(
    pl.col('value_contribution').fill_null(0.0)
)
batter_exposures = striker.select(['match_id', 'competition_type', 'pressure_class', 'player_id', 'exposure_flag']).with_columns(
    pl.col('exposure_flag').cast(pl.Int32).alias('exposure_balls')
).drop('exposure_flag')

# Bowler events
bowler_events = prd.select(common + ['bowler_player_id', 'legal_ball', 'bowler_bov_contribution']).rename({
    'bowler_player_id': 'player_id',
    'bowler_bov_contribution': 'value_contribution',
    'competition': 'competition_type'
}).with_columns([
    pl.col('legal_ball').cast(pl.Int32).alias('exposure_balls'),
    pl.lit(False).alias('dismissed_only'),
    pl.col('legal_ball').alias('exposure_flag')
])
bowler_exposures = bowler_events.select(['match_id', 'competition_type', 'pressure_class', 'player_id', 'exposure_balls'])

# 4. Make Player-Competition Statistics
def make_pc(events_pl, exposures_pl, role):
    classes = ['high_pressure', 'ordinary']
    ev = events_pl.filter(pl.col('pressure_class').is_in(classes))
    ex = exposures_pl.filter(pl.col('pressure_class').is_in(classes))
    
    # Sums by match
    sums = ev.group_by(['player_id', 'match_id', 'competition_type', 'pressure_class']).agg(
        pl.col('value_contribution').sum().alias('value_sum')
    )
    exs = ex.group_by(['player_id', 'match_id', 'competition_type', 'pressure_class']).agg(
        pl.col('exposure_balls').sum().alias('exposure_balls')
    )
    
    ms = sums.join(exs, on=['player_id', 'match_id', 'competition_type', 'pressure_class'], how='full', coalesce=True).with_columns([
        pl.col('value_sum').fill_null(0.0),
        pl.col('exposure_balls').fill_null(0.0)
    ]).with_columns(
        pl.when(pl.col('exposure_balls') > 0)
        .then(pl.col('value_sum') / pl.col('exposure_balls'))
        .otherwise(None)
        .alias('match_mean')
    )
    
    # Aggregate to player x competition
    # We iterate over player x competition
    rows = []
    # Convert ms to pandas-like grouping or process with polars
    # Using python dict for speed
    ms_records = ms.to_dicts()
    p_comp_map = {}
    for r in ms_records:
        key = (r['player_id'], r['competition_type'])
        if key not in p_comp_map:
            p_comp_map[key] = []
        p_comp_map[key].append(r)
        
    for (pid, comp), items in p_comp_map.items():
        # Separate high_pressure and ordinary
        p_items = [it for it in items if it['pressure_class'] == 'high_pressure']
        o_items = [it for it in items if it['pressure_class'] == 'ordinary']
        
        Sp = sum(it['value_sum'] for it in p_items)
        Np = sum(it['exposure_balls'] for it in p_items)
        So = sum(it['value_sum'] for it in o_items)
        No = sum(it['exposure_balls'] for it in o_items)
        
        vp = Sp / Np if Np > 0 else np.nan
        vo = So / No if No > 0 else np.nan
        
        # Match arrays
        m_set = {it['match_id'] for it in items}
        p_dict = {it['match_id']: (it['value_sum'], it['exposure_balls']) for it in p_items}
        o_dict = {it['match_id']: (it['value_sum'], it['exposure_balls']) for it in o_items}
        
        npm = sum(1 for it in p_items if it['exposure_balls'] > 0)
        nom = sum(1 for it in o_items if it['exposure_balls'] > 0)
        ncm = sum(1 for m in m_set if p_dict.get(m, (0, 0))[1] > 0 and o_dict.get(m, (0, 0))[1] > 0)
        
        A_list = []
        B_list = []
        for m in m_set:
            ps, pn = p_dict.get(m, (0.0, 0.0))
            os, on = o_dict.get(m, (0.0, 0.0))
            A_list.append(ps - vp * pn if Np > 0 else 0.0)
            B_list.append(os - vo * on if No > 0 else 0.0)
            
        A_arr = np.array(A_list, dtype=float)
        B_arr = np.array(B_list, dtype=float)
        
        varp = float(np.sum(A_arr * A_arr) / (Np * Np)) if (Np > 0 and npm >= 2) else np.nan
        varo = float(np.sum(B_arr * B_arr) / (No * No)) if (No > 0 and nom >= 2) else np.nan
        vard = float(np.sum((A_arr / Np - B_arr / No) ** 2)) if (Np > 0 and No > 0 and ncm >= 2) else np.nan
        
        pm_means = [it['match_mean'] for it in p_items if it['exposure_balls'] > 0 and it['match_mean'] is not None]
        pos_share = float(np.mean([m > 0 for m in pm_means])) if len(pm_means) > 0 else np.nan
        sd_val = float(np.std(pm_means, ddof=1)) if len(pm_means) > 1 else (0.0 if len(pm_means) == 1 else np.nan)
        
        rows.append({
            'player_id': str(pid),
            'competition_type': comp,
            'pressure_value_sum': Sp,
            'pressure_balls': Np,
            'ordinary_value_sum': So,
            'ordinary_balls': No,
            'raw_pressure_value_per_ball': vp,
            'raw_ordinary_value_per_ball': vo,
            'raw_pressure_delta': (vp - vo) if (np.isfinite(vp) and np.isfinite(vo)) else np.nan,
            'n_pressure_matches': npm,
            'n_ordinary_matches': nom,
            'n_common_matches': ncm,
            'matches_with_pressure_balls': npm,
            'positive_pressure_match_share': pos_share,
            'pressure_match_value_sd': sd_val,
            'var_pressure_cluster': varp,
            'var_ordinary_cluster': varo,
            'var_delta_cluster': vard
        })
        
    pc_df = pl.DataFrame(rows)
    
    # Ball groups for pooled variance
    eb_filtered = events_pl.filter(
        (pl.col('exposure_flag') == True) | (pl.col('dismissed_only') == True)
    ) if role == 'batter' else events_pl
    
    ball_df = eb_filtered.group_by(
        ['player_id', 'competition_type', 'pressure_class', 'match_id', 'innings', 'innings_ball_num']
    ).agg(pl.col('value_contribution').sum())
    
    # Group balls into dict
    bg_dict = {}
    for r in ball_df.to_dicts():
        k = (str(r['player_id']), r['competition_type'], r['pressure_class'])
        if k not in bg_dict:
            bg_dict[k] = []
        bg_dict[k].append(r['value_contribution'])
    bg = {k: np.array(v, dtype=float) for k, v in bg_dict.items()}
    
    return pc_df, ms, bg

# Variance floor helpers
def pooled_var(pc_df, bg, eligible_df, cls):
    sse = 0.0
    dfree = 0
    for r in eligible_df.iter_rows(named=True):
        v = bg.get((str(r['player_id']), r['competition_type'], cls), np.array([], dtype=float))
        if len(v) >= 2:
            sse += float(np.sum((v - np.mean(v)) ** 2))
            dfree += len(v) - 1
    return (sse / dfree if dfree > 0 else np.nan), dfree

def deff(col, ncol, eligible_df, var_val):
    x = eligible_df.filter(pl.col(col).is_not_null() & (pl.col(ncol) > 0))
    if len(x) == 0 or not np.isfinite(var_val) or var_val <= 0:
        return 1.0, 0.0, 0.0, len(x)
    cs = float(x[col].sum())
    nv = float(np.sum(var_val / x[ncol].to_numpy().astype(float)))
    return (cs / nv if nv > 0 else 1.0), cs, nv, len(x)

def add_floors(pc_df, bg, role):
    ep = pc_df.filter(
        (pl.col('n_pressure_matches') >= 5) &
        (pl.col('pressure_balls') >= 60) &
        pl.col('raw_pressure_value_per_ball').is_not_null()
    )
    ed = pc_df.filter(
        (pl.col('n_common_matches') >= 5) &
        (pl.col('ordinary_balls') >= 40) &
        pl.col('raw_pressure_delta').is_not_null()
    )
    assert len(ep) >= 3 and len(ed) >= 3, f"{role}: inadequate EB variance floor cohort"
    
    sp, dfp = pooled_var(pc_df, bg, ep, 'high_pressure')
    dep, clp, navp, ndep = deff('var_pressure_cluster', 'pressure_balls', ep, sp)
    
    spd, dfpd = pooled_var(pc_df, bg, ed, 'high_pressure')
    sod, dfod = pooled_var(pc_df, bg, ed, 'ordinary')
    dedp, cldp, navdp, ndepd = deff('var_pressure_cluster', 'pressure_balls', ed, spd)
    dedo, cldo, navdo, ndedo = deff('var_ordinary_cluster', 'ordinary_balls', ed, sod)
    
    pb = pc_df['pressure_balls'].to_numpy().astype(float)
    ob = pc_df['ordinary_balls'].to_numpy().astype(float)
    
    vpc = pc_df['var_pressure_cluster'].fill_null(0.0).to_numpy()
    voc = pc_df['var_ordinary_cluster'].fill_null(0.0).to_numpy()
    vdc = pc_df['var_delta_cluster'].fill_null(0.0).to_numpy()
    
    vpf = np.where(pb > 0, sp * dep / pb, np.nan)
    vof = np.where(ob > 0, sod * dedo / ob, np.nan)
    vdf = np.where((pb > 0) & (ob > 0), spd * dedp / pb + sod * dedo / ob, np.nan)
    
    vp_final = np.where(np.isfinite(vpf), np.maximum(vpc, vpf), np.nan)
    vo_final = np.where(np.isfinite(vof), np.maximum(voc, vof), np.nan)
    vd_final = np.where(np.isfinite(vdf), np.maximum(vdc, vdf), np.nan)
    
    out_df = pc_df.with_columns([
        pl.Series('var_pressure_floor', vpf),
        pl.Series('var_ordinary_floor', vof),
        pl.Series('var_delta_floor', vdf),
        pl.Series('var_pressure_final', vp_final),
        pl.Series('var_ordinary_final', vo_final),
        pl.Series('var_delta_final', vd_final)
    ])
    
    diag = {
        'role': role,
        'pressure_estimation_player_comp_rows': len(ep),
        'pressure_estimation_unique_players': int(ep['player_id'].n_unique()),
        'pressure_pooled_per_ball_variance': sp,
        'pressure_pooled_variance_df': dfp,
        'pressure_design_effect': dep,
        'pressure_design_effect_n': ndep,
        'pressure_cluster_variance_sum': clp,
        'pressure_naive_variance_sum': navp,
        'delta_estimation_player_comp_rows': len(ed),
        'delta_estimation_unique_players': int(ed['player_id'].n_unique()),
        'delta_pressure_pooled_per_ball_variance': spd,
        'delta_pressure_pooled_variance_df': dfpd,
        'delta_pressure_design_effect': dedp,
        'delta_pressure_design_effect_n': ndepd,
        'delta_pressure_cluster_variance_sum': cldp,
        'delta_pressure_naive_variance_sum': navdp,
        'delta_ordinary_pooled_per_ball_variance': sod,
        'delta_ordinary_pooled_variance_df': dfod,
        'delta_ordinary_design_effect': dedo,
        'delta_ordinary_design_effect_n': ndedo,
        'delta_ordinary_cluster_variance_sum': cldo,
        'delta_ordinary_naive_variance_sum': navdo
    }
    return out_df, diag

def fit_dl(pc_df, metric, role):
    if metric == 'pressure':
        y_col = 'raw_pressure_value_per_ball'
        v_col = 'var_pressure_final'
        est = pc_df.filter(
            (pl.col('n_pressure_matches') >= 5) &
            (pl.col('pressure_balls') >= 60) &
            pl.col(y_col).is_not_null() &
            pl.col(v_col).is_not_null()
        )
        wt = est['pressure_balls'].to_numpy().astype(float)
    else:
        y_col = 'raw_pressure_delta'
        v_col = 'var_delta_final'
        est = pc_df.filter(
            (pl.col('n_common_matches') >= 5) &
            (pl.col('ordinary_balls') >= 40) &
            pl.col(y_col).is_not_null() &
            pl.col(v_col).is_not_null()
        )
        pb = est['pressure_balls'].to_numpy().astype(float)
        ob = est['ordinary_balls'].to_numpy().astype(float)
        wt = 1.0 / (1.0 / pb + 1.0 / ob)
        
    assert len(est) >= 3, f"{role}/{metric}: fewer than 3 EB estimation rows"
    
    v_arr = np.clip(est[v_col].to_numpy().astype(float), 1e-8, None)
    w_arr = 1.0 / v_arr
    y_arr = est[y_col].to_numpy().astype(float)
    comp_arr = est['competition_type'].to_numpy()
    
    mu_comp = {}
    Q = 0.0
    C = 0.0
    dfree = 0
    
    for comp in np.unique(comp_arr):
        idx = (comp_arr == comp)
        sw = float(np.sum(w_arr[idx]))
        sw2 = float(np.sum(w_arr[idx] ** 2))
        mu = float(np.sum(w_arr[idx] * y_arr[idx]) / sw)
        mu_comp[comp] = mu
        Q += float(np.sum(w_arr[idx] * (y_arr[idx] - mu) ** 2))
        C += sw - sw2 / sw
        dfree += int(np.sum(idx)) - 1
        
    swtot = float(np.sum(w_arr))
    muglobal = float(sum(np.sum(w_arr[comp_arr == c]) * mu for c, mu in mu_comp.items()) / swtot)
    ballmean = float(np.average(y_arr, weights=wt))
    diff = abs(muglobal - ballmean)
    fallback = bool(diff >= 0.05)
    
    shift = 0.0
    if fallback:
        shift = ballmean - muglobal
        mu_comp = {c: float(m + shift) for c, m in mu_comp.items()}
        muglobal = ballmean
        
    tau = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
    return {
        'role': role, 'metric': metric,
        'mu0_global_initial': float(muglobal - shift) if fallback else float(muglobal),
        'mu0_global_final': float(muglobal),
        'mu0_by_competition': mu_comp,
        'ball_weighted_mean': ballmean,
        'mu0_abs_diff_initial': diff,
        'mu0_fallback_applied': fallback,
        'mu0_abs_diff_final': abs(muglobal - ballmean),
        'tau2': float(tau),
        'between_player_sd': float(np.sqrt(tau)),
        'Q': float(Q), 'C': float(C), 'df': int(dfree),
        'estimation_rows': len(est),
        'estimation_players': int(est['player_id'].n_unique()),
        'estimation_threshold': 'n_pressure_matches>=5 & pressure_balls>=60' if metric == 'pressure' else 'n_common_matches>=5 & ordinary_balls>=40'
    }

def apply_eb(pc_df, pfit, dfit):
    # Pure Polars and NumPy implementation
    mu0_p = [pfit['mu0_by_competition'].get(c, pfit['mu0_global_final']) for c in pc_df['competition_type'].to_list()]
    mu0_d = [dfit['mu0_by_competition'].get(c, dfit['mu0_global_final']) for c in pc_df['competition_type'].to_list()]
    
    tp = pfit['tau2']
    td = dfit['tau2']
    
    vp = pc_df['var_pressure_final'].fill_null(np.inf).to_numpy()
    vd = pc_df['var_delta_final'].fill_null(np.inf).to_numpy()
    
    wp = np.where(np.isfinite(vp) & (tp > 0), tp / (tp + vp), 0.0)
    wd = np.where(np.isfinite(vd) & (td > 0), td / (td + vd), 0.0)
    
    raw_p = pc_df['raw_pressure_value_per_ball'].to_numpy()
    raw_d = pc_df['raw_pressure_delta'].to_numpy()
    
    shrunk_p = np.array(mu0_p) + wp * (raw_p - np.array(mu0_p))
    shrunk_d = np.array(mu0_d) + wd * (raw_d - np.array(mu0_d))
    
    if tp <= 1e-12:
        shrunk_p = np.array(mu0_p)
    else:
        shrunk_p[np.isnan(raw_p)] = np.nan
        
    if td <= 1e-12:
        shrunk_d = np.array(mu0_d)
    else:
        shrunk_d[np.isnan(raw_d)] = np.nan
        
    recentered_p = shrunk_p - np.array(mu0_p)
    recentered_d = shrunk_d - np.array(mu0_d)
    
    return pc_df.with_columns([
        pl.Series('mu0_pressure_comp', mu0_p),
        pl.Series('mu0_delta_comp', mu0_d),
        pl.Series('eb_pressure_weight', wp),
        pl.Series('eb_delta_weight', wd),
        pl.Series('shrunk_pressure_value_per_ball', shrunk_p),
        pl.Series('shrunk_pressure_delta', shrunk_d),
        pl.Series('recentered_pressure_deviation', recentered_p),
        pl.Series('recentered_delta_deviation', recentered_d)
    ])

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

def build_player_ratings(pc_df, role, excluded_set, dfit_role, min_balls=100, min_matches=12):
    records = pc_df.to_dicts()
    p_map = {}
    for r in records:
        pid = r['player_id']
        if pid not in p_map:
            p_map[pid] = []
        p_map[pid].append(r)
        
    rows = []
    for pid, g in p_map.items():
        pp = sum(r['pressure_balls'] for r in g)
        oo = sum(r['ordinary_balls'] for r in g)
        
        # Exposure weights
        w = np.array([r['pressure_balls'] for r in g], dtype=float)
        pdev_vals = np.array([r['recentered_pressure_deviation'] for r in g], dtype=float)
        ddev_vals = np.array([r['recentered_delta_deviation'] for r in g], dtype=float)
        
        p_valid = np.isfinite(pdev_vals) & (w > 0)
        pdev = float(np.average(pdev_vals[p_valid], weights=w[p_valid])) if np.sum(p_valid) > 0 else np.nan
        
        d_valid = np.isfinite(ddev_vals) & (w > 0)
        ddev = float(np.average(ddev_vals[d_valid], weights=w[d_valid])) if np.sum(d_valid) > 0 else np.nan
        
        rawp_vals = np.array([r['raw_pressure_value_per_ball'] - r['mu0_pressure_comp'] for r in g], dtype=float)
        rawp_valid = np.isfinite(rawp_vals) & (w > 0)
        rawp = float(np.average(rawp_vals[rawp_valid], weights=w[rawp_valid])) if np.sum(rawp_valid) > 0 else np.nan
        
        rawd_vals = np.array([r['raw_pressure_delta'] - r['mu0_delta_comp'] for r in g], dtype=float)
        rawd_valid = np.isfinite(rawd_vals) & (w > 0)
        rawd = float(np.average(rawd_vals[rawd_valid], weights=w[rawd_valid])) if np.sum(rawd_valid) > 0 else np.nan
        
        nmatch = sum(r['n_pressure_matches'] for r in g)
        t20 = sum(r['pressure_balls'] for r in g if r['competition_type'] == 'T20I')
        
        Sp = sum(r['pressure_value_sum'] for r in g)
        So = sum(r['ordinary_value_sum'] for r in g)
        vp = Sp / pp if pp > 0 else np.nan
        vo = So / oo if oo > 0 else np.nan
        
        rows.append({
            'player_id': str(pid),
            'player_name': name_map.get(str(pid), ''),
            'pressure_balls': int(pp),
            'ordinary_balls': int(oo),
            'matches_with_pressure_balls': int(nmatch),
            'matches_with_ordinary_balls': int(sum(r['n_ordinary_matches'] for r in g)),
            't20i_pressure_balls': int(t20),
            't20i_pressure_exposure_share': float(t20 / pp) if pp > 0 else np.nan,
            'raw_pressure_value_per_ball': vp,
            'raw_ordinary_value_per_ball': vo,
            'raw_pressure_delta': (vp - vo) if np.isfinite(vp) and np.isfinite(vo) else np.nan,
            'raw_pressure_deviation': rawp,
            'raw_delta_deviation': rawd,
            'recentered_shrunk_pressure_deviation': pdev,
            'recentered_shrunk_delta_deviation': ddev,
            'excluded_by_ipl2026_or_pakistan': bool(str(pid) in excluded_set)
        })
        
    out_df = pl.DataFrame(rows)
    
    # Eligibility filter
    sel_mask = (
        (out_df['pressure_balls'] >= min_balls) &
        (out_df['matches_with_pressure_balls'] >= min_matches) &
        (~out_df['excluded_by_ipl2026_or_pakistan'])
    )
    sel = out_df.filter(sel_mask)
    
    if len(sel) > 0:
        zp = zscore_series(sel['recentered_shrunk_pressure_deviation'].to_numpy())
        zd = zscore_series(sel['recentered_shrunk_delta_deviation'].to_numpy())
        
        if role == 'batter' and dfit_role['tau2'] > 1e-12:
            rating = 0.70 * zp + 0.30 * zd
            pw, dw = 0.70, 0.30
        else:
            zd = np.zeros_like(zp)
            rating = zp
            pw, dw = 1.0, 0.0
            
        sel = sel.with_columns([
            pl.Series('z_pressure', zp),
            pl.Series('z_delta', zd),
            pl.Series('rating', rating),
            pl.lit(pw).alias('rating_pressure_weight'),
            pl.lit(dw).alias('rating_delta_weight')
        ])
    else:
        sel = sel.with_columns([
            pl.Series('z_pressure', np.array([], dtype=float)),
            pl.Series('z_delta', np.array([], dtype=float)),
            pl.Series('rating', np.array([], dtype=float)),
            pl.Series('rating_pressure_weight', np.array([], dtype=float)),
            pl.Series('rating_delta_weight', np.array([], dtype=float))
        ])
        
    # Merge ratings back to out_df
    out_df = out_df.join(
        sel.select(['player_id', 'z_pressure', 'z_delta', 'rating', 'rating_pressure_weight', 'rating_delta_weight']),
        on='player_id', how='left'
    )
    return out_df, sel

# Execute EB pipeline
role_results = {}
eb_rows = []
variance_rows = []
dfit_by_role = {}

for role, evs, exs in [('batter', batter_events, batter_exposures), ('bowler', bowler_events, bowler_exposures)]:
    print(f"Running Empirical Bayes for {role}...", flush=True)
    pc_raw, ms_raw, bg_raw = make_pc(evs, exs, role)
    pc_floored, vdiag = add_floors(pc_raw, bg_raw, role)
    pfit = fit_dl(pc_floored, 'pressure', role)
    dfit = fit_dl(pc_floored, 'delta', role)
    dfit_by_role[role] = dfit
    shr = apply_eb(pc_floored, pfit, dfit)
    
    role_results[role] = {
        'pc': shr, 'ms': ms_raw, 'ball_groups': bg_raw,
        'pressure_fit': pfit, 'delta_fit': dfit, 'variance_diag': vdiag,
        'events': evs, 'exposures': exs
    }
    for fit in [pfit, dfit]:
        eb_rows.append({k: v for k, v in fit.items() if k != 'mu0_by_competition'} | {
            'mu0_by_competition_json': json.dumps(fit['mu0_by_competition'], sort_keys=True)
        })
    variance_rows.append(vdiag)
    
    shr.write_csv(OUT_TAB / f'{role}_player_comp_empirical_bayes.csv')
    ms_raw.write_parquet(OUT_INTER / f'{role}_player_match_class_stats.parquet')
    ms_raw.write_csv(OUT_INTER / f'{role}_player_match_class_stats.csv.gz', compression='gzip')

# Build Ratings for Main Eligible Pools
print("Building player ratings for main pools (floors: 100 balls / 12 matches)...", flush=True)
main_ratings = {}
main_eligible = {}

for role in ['batter', 'bowler']:
    role_excl = excluded | ({"PID_786"} if role == 'batter' else {"PID_3727"})
    allp, sel = build_player_ratings(role_results[role]['pc'], role, role_excl, dfit_by_role[role], 100, 12)
    main_ratings[role] = allp
    main_eligible[role] = sel
    allp.write_csv(OUT_TAB / f'{role}_all_player_recentered_eb_diagnostics.csv')
    sel.write_csv(OUT_TAB / f'{role}_eligible_recentered_rating_components_gate2.csv')
    sel.drop(['z_pressure', 'z_delta', 'rating']).write_csv(OUT_TAB / f'{role}_main_eligibility_exposure_audit.csv')

# Acceptance Checks EB-1 through EB-5
print("Evaluating Acceptance Checks EB-1 through EB-5...", flush=True)
acc = []
for role in ['batter', 'bowler']:
    for metric, fit in [('pressure', role_results[role]['pressure_fit']), ('delta', role_results[role]['delta_fit'])]:
        acc.append({
            'check': 'EB1_mu0_ball_weighted_mean', 'role': role, 'metric': metric,
            'n_estimation_rows': fit['estimation_rows'], 'mu0_final': fit['mu0_global_final'],
            'ball_weighted_raw_mean': fit['ball_weighted_mean'], 'absolute_difference': fit['mu0_abs_diff_final'],
            'threshold': '<0.05', 'pass': fit['mu0_abs_diff_final'] < 0.05,
            'fallback_applied': fit['mu0_fallback_applied']
        })

for role in ['batter', 'bowler']:
    pc = role_results[role]['pc']
    records = pc.to_dicts()
    p_map = {}
    for r in records:
        pid = r['player_id']
        if pid not in p_map: p_map[pid] = []
        p_map[pid].append(r)
        
    q_rows = []
    for pid, g in p_map.items():
        Np = sum(r['pressure_balls'] for r in g)
        if Np < 100: continue
        raw = sum(r['pressure_value_sum'] for r in g) / Np
        vsum = sum(((r['pressure_balls'] / Np) ** 2) * r['var_pressure_final'] for r in g if r['var_pressure_final'] is not None)
        q_rows.append({'player_id': pid, 'pressure_balls': Np, 'raw_pressure_value_per_ball': raw, 'final_sampling_variance': vsum})
        
    q_df = pl.DataFrame(q_rows) if q_rows else pl.DataFrame()
    meanv = float(q_df['final_sampling_variance'].mean()) if len(q_df) > 0 else np.nan
    obsvar = float(q_df['raw_pressure_value_per_ball'].var(ddof=1)) if len(q_df) > 1 else np.nan
    
    # In Gate 2 brief, EB-2 condition is noted
    acc.append({
        'check': 'EB2_sampling_variance_not_below_observed', 'role': role, 'metric': 'pressure',
        'n_players_ge_100_balls': len(q_df), 'mean_sampling_variance': meanv,
        'observed_raw_per_ball_variance': obsvar, 'threshold': '>=',
        'pass': bool(np.isfinite(meanv) and np.isfinite(obsvar) and meanv >= obsvar)
    })

for role in ['batter', 'bowler']:
    sel = main_eligible[role].filter(pl.col('rating').is_not_null())
    top = sel.sort('rating', descending=True).head(30)
    rating_arr = top['rating'].to_numpy()
    ties = int(np.sum(np.isin(rating_arr, rating_arr[np.unique(rating_arr, return_counts=True)[1] > 1])))
    acc.append({
        'check': 'EB3_no_exact_rating_ties_top30', 'role': role, 'metric': 'rating',
        'n_eligible': len(sel), 'top30_rows': len(top), 'tied_rows_top30': ties,
        'threshold': '0', 'pass': ties == 0
    })
    
    q200 = sel.filter(pl.col('pressure_balls') >= 200).filter(
        pl.col('recentered_shrunk_pressure_deviation').is_not_null() & pl.col('raw_pressure_deviation').is_not_null()
    )
    rho = float(spearmanr(q200['recentered_shrunk_pressure_deviation'].to_numpy(), q200['raw_pressure_deviation'].to_numpy(), nan_policy='omit').statistic) if len(q200) > 2 else np.nan
    acc.append({
        'check': 'EB4_spearman_shrunk_vs_raw_pressure', 'role': role, 'metric': 'pressure',
        'n_players_ge_200_balls': len(q200), 'spearman': rho, 'threshold': '>0.5',
        'pass': bool(np.isfinite(rho) and rho > 0.5)
    })
    
    t20_share = sel['t20i_pressure_exposure_share'].to_numpy()
    rating_sel = sel['rating'].to_numpy()
    valid_corr = np.isfinite(t20_share) & np.isfinite(rating_sel)
    corr = float(pearsonr(rating_sel[valid_corr], t20_share[valid_corr]).statistic) if np.sum(valid_corr) > 2 and len(np.unique(t20_share[valid_corr])) > 1 else np.nan
    acc.append({
        'check': 'EB5_rating_vs_T20I_exposure_share', 'role': role, 'metric': 'rating',
        'n_eligible': len(sel), 'pearson_correlation': corr, 'abs_correlation': abs(corr) if np.isfinite(corr) else np.nan,
        'threshold': 'abs<0.30', 'pass': bool(np.isfinite(corr) and abs(corr) < 0.30)
    })

# Add tau2 boundary notes
for role in ['batter', 'bowler']:
    for metric in ['pressure', 'delta']:
        f = role_results[role][f'{metric}_fit']
        acc.append({
            'check': 'tau2_boundary_note', 'role': role, 'metric': metric,
            'tau2': f['tau2'], 'between_player_sd': f['between_player_sd'],
            'boundary_zero': f['tau2'] <= 1e-12, 'pass': True,
            'policy': 'zero tau2 valid; posterior equals competition prior mean; component deviation zero'
        })

pl.DataFrame(acc).write_csv(OUT_TAB / 'empirical_bayes_acceptance_checks.csv')
pl.DataFrame(eb_rows).write_csv(OUT_TAB / 'empirical_bayes_hyperparameter_estimates.csv')
pl.DataFrame(variance_rows).write_csv(OUT_TAB / 'empirical_bayes_variance_floor_diagnostics.csv')

# SMAT Exposure Audit
print("Generating SMAT Exposure Audit & Watchlist...", flush=True)
sm = prd.filter(pl.col('competition') == 'Syed Mushtaq Ali Trophy')
sm_rows = []
for role, pidcol, expcol in [('batter', 'striker_player_id', 'batter_faced'), ('bowler', 'bowler_player_id', 'legal_ball')]:
    z = sm.filter(pl.col(expcol) == True).filter(pl.col('pressure_class') == 'high_pressure')
    agg = z.group_by(pidcol).agg([
        pl.len().alias('pressure_balls'),
        pl.col('match_id').n_unique().alias('pressure_matches')
    ]).rename({pidcol: 'player_id'})
    
    allapp = sm.filter(pl.col(expcol) == True).group_by(pidcol).agg(
        pl.col('match_id').n_unique().alias('smat_matches_played')
    ).rename({pidcol: 'player_id'})
    
    q = allapp.join(agg, on='player_id', how='left').with_columns([
        pl.col('pressure_balls').fill_null(0),
        pl.col('pressure_matches').fill_null(0),
        pl.lit(role).alias('role'),
        pl.col('player_id').map_elements(lambda pid: name_map.get(str(pid), ''), return_dtype=pl.Utf8).alias('player_name')
    ]).with_columns([
        ((pl.col('pressure_balls') >= 30) & (pl.col('pressure_matches') >= 4)).alias('watchlist_floor_met'),
        ((pl.col('pressure_balls') >= 20) & (pl.col('pressure_matches') >= 2)).alias('exploratory_floor_met'),
        pl.lit('LOW - SMAT maximum seven matches').alias('confidence')
    ])
    sm_rows.append(q)

smat_exp = pl.concat(sm_rows)
smat_exp.write_csv(OUT_TAB / 'smat_player_pressure_exposure_audit.csv')

# Watchlist table (expected to be empty under strict 30 balls / 4 matches floor)
smat_watch_cols = ['player_id', 'player_name', 'role', 'smat_matches_played', 'pressure_balls', 'pressure_matches', 'confidence']
smat_watch = smat_exp.filter(pl.col('watchlist_floor_met') == True).select(smat_watch_cols)
smat_watch.write_csv(OUT_TAB / 'smat_indian_domestic_watchlist_low_confidence.csv')

# Exploratory exposure table (>= 20 balls, >= 2 matches)
smat_exploratory = smat_exp.filter(pl.col('exploratory_floor_met') == True).select(smat_watch_cols + ['exploratory_floor_met'])
smat_exploratory.write_csv(OUT_TAB / 'smat_exploratory_exposure_table.csv')

# Dedicated SMAT Rankings under reduced floor (>= 15 pressure balls, >= 2 pressure matches)
# Filtered to unexcluded domestic talent (outside IPL 2026, Pakistan, and retired players)
print("Generating Dedicated SMAT Rankings (>= 15 pressure balls, >= 2 pressure matches)...", flush=True)
for role in ['batter', 'bowler']:
    pc = role_results[role]['pc']
    pc_smat = pc.filter(
        (pl.col('competition_type') == 'Syed Mushtaq Ali Trophy') &
        (pl.col('pressure_balls') >= 15) &
        (pl.col('matches_with_pressure_balls') >= 2) &
        (~pl.col('player_id').is_in(excluded))
    )
    
    # Get total SMAT matches played from smat_exp
    matches_lookup = dict(zip(
        smat_exp.filter(pl.col('role') == role)['player_id'].to_list(),
        smat_exp.filter(pl.col('role') == role)['smat_matches_played'].to_list()
    ))
    
    # Compute z-scores for rating
    p_devs = pc_smat['recentered_pressure_deviation'].fill_null(0.0).to_numpy()
    smat_z = zscore_series(p_devs)
    
    smat_ranked = pc_smat.with_columns([
        pl.Series('smat_rating', smat_z),
        pl.col('player_id').map_elements(lambda pid: name_map.get(str(pid), str(pid)), return_dtype=pl.Utf8).alias('player_name'),
        pl.col('player_id').map_elements(lambda pid: matches_lookup.get(str(pid), 0), return_dtype=pl.Int64).alias('smat_matches_played'),
        pl.lit(role).alias('role'),
        pl.lit('LOW - Limited SMAT coverage (2-4 pressure matches); indicative domestic talent signal').alias('confidence')
    ]).sort('smat_rating', descending=True)
    
    # Assign 1-based smat_rank
    smat_records = smat_ranked.to_dicts()
    for rk, r in enumerate(smat_records, 1):
        r['smat_rank'] = rk
        
    smat_out_cols = [
        'smat_rank', 'player_id', 'player_name', 'role', 'smat_matches_played',
        'pressure_balls', 'matches_with_pressure_balls', 'raw_pressure_value_per_ball',
        'raw_ordinary_value_per_ball', 'raw_pressure_delta',
        'shrunk_pressure_value_per_ball', 'recentered_pressure_deviation',
        'smat_rating', 'confidence'
    ]
    
    smat_out_df = pl.DataFrame(smat_records).select(smat_out_cols)
    smat_out_df.write_csv(OUT_DELIV / f'smat_{role}_rankings.csv')
    smat_out_df.write_csv(OUT_TAB / f'smat_{role}_rankings.csv')
    print(f"Wrote SMAT {role} rankings ({len(smat_out_df)} players) to {OUT_DELIV}/smat_{role}_rankings.csv", flush=True)

print(f"SMAT Exposure Audit complete: {len(smat_exp)} player-role rows. Watchlist count: {len(smat_watch)}. Exploratory count: {len(smat_exploratory)}.")

print("EMPIRICAL BAYES MODEL COMPLETE!", flush=True)
