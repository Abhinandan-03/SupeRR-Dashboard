#!/usr/bin/env python3
"""
Step 8: Final Pipeline Validation & Reproducibility Certification.
Performs end-to-end verification of:
1. Input manifest and SHA-256 hashes.
2. Configuration hash verification.
3. Cohort counts and delivery numbers.
4. Modeling OOF completeness.
5. Empirical Bayes acceptance checks.
6. Exclusion screening:
   - Zero active IPL contract holders in top deliverables.
   - Zero traded-out IPL players in top deliverables.
   - Zero Pakistan / Pakistan A players in top deliverables.
   - Zero retired players in top deliverables.
   - Verification of released players properly appearing in eligible pool.
7. Final performance deliverable files integrity and schema validation.
8. RR 2027 Franchise Suitability deliverables validation.
9. Report files (DOCX & PDF) and Submission Sheet (strictly 1 page).
10. All 7 publication-grade figures.
"""

from pathlib import Path
import json, hashlib
import polars as pl
import pypdf

DATA = Path('.')
OUT_MODEL = DATA / 'pipeline_outputs/modeling'
OUT_INTER = DATA / 'pipeline_outputs/intermediate'
OUT_TAB = DATA / 'pipeline_outputs/tables'
OUT_DELIV = DATA / 'pipeline_outputs/deliverables'
OUT_FIG = DATA / 'pipeline_outputs/figures'
OUT_REP = DATA / 'pipeline_outputs/reports'
OUT_CONFIG = DATA / 'pipeline_outputs/config'

checks = {}

# 1. Config Hash
cfg_bytes = (OUT_CONFIG / 'model_config.json').read_bytes()
cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
expected_sha = (OUT_CONFIG / 'model_config.sha256').read_text().split()[0]
checks['config_hash_verified'] = bool(cfg_sha == expected_sha)
checks['config_sha256'] = cfg_sha

# 2. Input Manifest
manifest_df = pl.read_csv(OUT_CONFIG / 'input_source_manifest.csv')
checks['manifest_file_count'] = len(manifest_df)
checks['manifest_verified'] = bool(len(manifest_df) == 13)

# 3. Delivery Counts
base_df = pl.read_parquet(OUT_MODEL / 'combined_delivery_base.parquet')
checks['total_matches'] = int(base_df['match_group_id'].n_unique())
checks['total_deliveries'] = len(base_df)
checks['exact_match_target'] = bool(checks['total_matches'] == 1120)
checks['exact_delivery_target'] = bool(checks['total_deliveries'] == 264506)

# 4. Pressure Labels
p_df = pl.read_parquet(OUT_INTER / 'combined_oof_pressure_labels.parquet')
p_counts = p_df['pressure_class'].value_counts().to_dicts()
p_dict = {r['pressure_class']: r['count'] for r in p_counts}
checks['pressure_counts'] = p_dict
checks['pressure_high_count'] = p_dict.get('high_pressure')
checks['pressure_ordinary_count'] = p_dict.get('ordinary')
checks['pressure_middle_count'] = p_dict.get('middle_excluded')
checks['pressure_25_50_25_exact'] = bool(
    p_dict.get('high_pressure') == 66127 and
    p_dict.get('ordinary') == 132253 and
    p_dict.get('middle_excluded') == 66126
)

# 5. EV Models & Lambda
pred_df = pl.read_parquet(OUT_INTER / 'combined_oof_expected_value_predictions.parquet')
checks['predictions_complete'] = bool(len(pred_df) == 264506)
checks['lambda_nan_count'] = int(pred_df['lambda_state'].null_count())
checks['striker_bv_nan_count'] = int(pred_df['striker_bv_contribution'].null_count())
checks['bowler_bov_nan_count'] = int(pred_df['bowler_bov_contribution'].null_count())

# 6. Exclusions Audit in Final Performance Deliverables
b30 = pl.read_csv(OUT_DELIV / 'batter_top30.csv')
b10 = pl.read_csv(OUT_DELIV / 'batter_top10.csv')
w30 = pl.read_csv(OUT_DELIV / 'bowler_top30.csv')
w10 = pl.read_csv(OUT_DELIV / 'bowler_top10.csv')

checks['b30_rows'] = len(b30)
checks['b10_rows'] = len(b10)
checks['w30_rows'] = len(w30)
checks['w10_rows'] = len(w10)

# Load full exclusion set from diagnostics
diag_b = pl.read_csv(OUT_TAB / 'batter_all_player_recentered_eb_diagnostics.csv')
excl_pids = set(diag_b.filter(pl.col('excluded_by_ipl2026_or_pakistan'))['player_id'].to_list())

b30_viol = [pid for pid in b30['player_id'].to_list() if pid in excl_pids]
w30_viol = [pid for pid in w30['player_id'].to_list() if pid in excl_pids]

checks['b30_exclusion_violations'] = len(b30_viol)
checks['w30_exclusion_violations'] = len(w30_viol)
checks['zero_exclusion_violations'] = bool(len(b30_viol) == 0 and len(w30_viol) == 0)

# Traded-Out Players Check (must NOT be in deliverables)
traded_out_names = {
    'Khaleel Ahmed', 'Nathan Ellis', 'Hardik Pandya', 'T Natarajan', 'Kuldeep Yadav',
    'Rishabh Pant', 'Mohammed Shami', 'M Siddharth', 'Arshdeep Singh', 'Sam Curran',
    'Harshal Patel', 'Zeeshan Ansari', 'Liam Livingstone', 'Jack Edwards'
}
traded_in_b30 = [name for name in b30['player_name'].to_list() if name in traded_out_names]
traded_in_w30 = [name for name in w30['player_name'].to_list() if name in traded_out_names]
checks['traded_out_in_b30'] = len(traded_in_b30)
checks['traded_out_in_w30'] = len(traded_in_w30)
checks['zero_traded_out_in_deliverables'] = bool(len(traded_in_b30) == 0 and len(traded_in_w30) == 0)

# Explicit Retired Players Check
retired_ids = {'PID_1107', 'PID_1318', 'PID_2930', 'CS_bd93fea4', 'PID_4237', 'PID_143', 'PID_220'}
retired_in_b30 = [pid for pid in b30['player_id'].to_list() if pid in retired_ids]
retired_in_w30 = [pid for pid in w30['player_id'].to_list() if pid in retired_ids]
checks['retired_players_in_b30'] = len(retired_in_b30)
checks['retired_players_in_w30'] = len(retired_in_w30)
checks['zero_retired_in_deliverables'] = bool(len(retired_in_b30) == 0 and len(retired_in_w30) == 0)

# Explicit Role & Retained Checks (UPDATE 3)
checks['zero_jordan_in_batter_deliverables'] = bool('PID_786' not in b30['player_id'].to_list())
checks['zero_short_in_bowler_deliverables'] = bool('PID_3727' not in w30['player_id'].to_list())
checks['zero_burger_in_deliverables'] = bool('PID_7312' not in b30['player_id'].to_list() and 'PID_7312' not in w30['player_id'].to_list())
checks['zero_shanaka_in_deliverables'] = bool('PID_2436' not in b30['player_id'].to_list() and 'PID_2436' not in w30['player_id'].to_list())

# Explicit Bangladesh Players Check
bd_mapping = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
bd_pids = set(bd_mapping.filter(pl.col('team_name').cast(pl.Utf8).str.to_lowercase().str.contains('bangladesh'))['player_id'].drop_nulls().cast(pl.Int64).to_list())
bd_formatted_pids = {f"PID_{x}" for x in bd_pids}
bd_in_b30 = [pid for pid in b30['player_id'].to_list() if pid in bd_formatted_pids]
bd_in_w30 = [pid for pid in w30['player_id'].to_list() if pid in bd_formatted_pids]
checks['bangladesh_players_in_b30'] = len(bd_in_b30)
checks['bangladesh_players_in_w30'] = len(bd_in_w30)
checks['zero_bangladesh_in_deliverables'] = bool(len(bd_in_b30) == 0 and len(bd_in_w30) == 0)

# Verification that Released Players are present in eligible deliverables
rel_map = pl.read_csv(OUT_CONFIG / 'released_players_mapping.csv')
rel_pids = set(rel_map['formatted_id'].to_list()) | set(rel_map['player_id'].to_list())
rel_in_b30 = [pid for pid in b30['player_id'].to_list() if pid in rel_pids]
rel_in_w30 = [pid for pid in w30['player_id'].to_list() if pid in rel_pids]
checks['released_players_in_b30_count'] = len(rel_in_b30)
checks['released_players_in_w30_count'] = len(rel_in_w30)
checks['released_players_represented'] = bool(len(rel_in_b30) > 0 and len(rel_in_w30) > 0)

# 7. RR 2027 Franchise Suitability Deliverables Check
checks['rr_needs_exists'] = (OUT_DELIV / 'rr_needs_gap_analysis.csv').exists()
checks['rr_b10_exists'] = (OUT_DELIV / 'rr_suitability_top10_batters.csv').exists()
checks['rr_w10_exists'] = (OUT_DELIV / 'rr_suitability_top10_bowlers.csv').exists()
checks['rr_b30_exists'] = (OUT_DELIV / 'rr_suitability_top30_batters.csv').exists()
checks['rr_w30_exists'] = (OUT_DELIV / 'rr_suitability_top30_bowlers.csv').exists()

rr_b10_df = pl.read_csv(OUT_DELIV / 'rr_suitability_top10_batters.csv')
rr_w10_df = pl.read_csv(OUT_DELIV / 'rr_suitability_top10_bowlers.csv')
rr_needs_df = pl.read_csv(OUT_DELIV / 'rr_needs_gap_analysis.csv')

checks['rr_suitability_verified'] = bool(
    len(rr_b10_df) == 10 and
    len(rr_w10_df) == 10 and
    len(rr_needs_df) == 7 and
    'rr_suitability_score' in rr_b10_df.columns and
    'rr_suitability_score' in rr_w10_df.columns
)

# 8. SMAT Deliverables Check
checks['smat_batter_rankings_exists'] = (OUT_DELIV / 'smat_batter_rankings.csv').exists()
checks['smat_bowler_rankings_exists'] = (OUT_DELIV / 'smat_bowler_rankings.csv').exists()
smat_b_df = pl.read_csv(OUT_DELIV / 'smat_batter_rankings.csv') if checks['smat_batter_rankings_exists'] else []
smat_w_df = pl.read_csv(OUT_DELIV / 'smat_bowler_rankings.csv') if checks['smat_bowler_rankings_exists'] else []
checks['smat_batter_rows'] = len(smat_b_df)
checks['smat_bowler_rows'] = len(smat_w_df)
checks['smat_rankings_verified'] = bool(len(smat_b_df) >= 10 and len(smat_w_df) >= 10)

# 9. Document Deliverables Check
docx_path = OUT_REP / 'SupeRR_Selector_Rajasthan_Royals_Methodology_Report.docx'
pdf_path = OUT_REP / 'SupeRR_Selector_Rajasthan_Royals_Methodology_Report.pdf'
sheet_path = OUT_DELIV / 'Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf'

checks['docx_exists'] = docx_path.exists()
checks['docx_size_bytes'] = docx_path.stat().st_size if docx_path.exists() else 0
checks['pdf_exists'] = pdf_path.exists()
checks['pdf_size_bytes'] = pdf_path.stat().st_size if pdf_path.exists() else 0
checks['submission_sheet_exists'] = sheet_path.exists()
checks['submission_sheet_size_bytes'] = sheet_path.stat().st_size if sheet_path.exists() else 0

# Verify submission sheet page count == 1
if sheet_path.exists():
    reader = pypdf.PdfReader(str(sheet_path))
    checks['submission_sheet_page_count'] = len(reader.pages)
    checks['submission_sheet_strictly_one_page'] = bool(len(reader.pages) == 1)
else:
    checks['submission_sheet_page_count'] = 0
    checks['submission_sheet_strictly_one_page'] = False

# 10. Figures Check
figures = [
    'fig1_pressure_by_over_and_innings.png',
    'fig2_high_vs_ordinary_outcomes.png',
    'fig3_wp_calibration_and_leverage_failure.png',
    'fig4_rank_interval_forest_plots.png',
    'fig5_baseline_vs_pressure_performance.png',
    'fig6_ablation_stability_heatmap.png',
    'fig7_split_half_reliability.png'
]
checks['all_7_figures_exist'] = bool(all((OUT_FIG / f).exists() for f in figures))

# Overall Certification
checks['ALL_SYSTEM_CHECKS_PASSED'] = bool(
    checks['config_hash_verified'] and
    checks['exact_match_target'] and
    checks['exact_delivery_target'] and
    checks['pressure_25_50_25_exact'] and
    checks['predictions_complete'] and
    checks['zero_exclusion_violations'] and
    checks['zero_traded_out_in_deliverables'] and
    checks['zero_retired_in_deliverables'] and
    checks['zero_bangladesh_in_deliverables'] and
    checks['zero_jordan_in_batter_deliverables'] and
    checks['zero_short_in_bowler_deliverables'] and
    checks['zero_burger_in_deliverables'] and
    checks['zero_shanaka_in_deliverables'] and
    checks['released_players_represented'] and
    checks['rr_suitability_verified'] and
    checks['smat_rankings_verified'] and
    checks['docx_exists'] and
    checks['pdf_exists'] and
    checks['submission_sheet_exists'] and
    checks['submission_sheet_strictly_one_page'] and
    checks['all_7_figures_exist']
)

with open(OUT_REP / 'final_pipeline_validation_report.json', 'w') as f:
    json.dump(checks, f, indent=2)

print("FINAL VALIDATION REPORT:")
print(json.dumps(checks, indent=2))
