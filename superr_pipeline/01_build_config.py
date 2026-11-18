import json, hashlib, os
from pathlib import Path

OUT_DIR = Path('pipeline_outputs/config')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALIASES = [
  {"alias": "Royal Challengers Bangalore", "canonical": "Royal Challengers Bengaluru", "reason": "IPL franchise renaming"},
  {"alias": "Royal Challengers Bengaluru", "canonical": "Royal Challengers Bengaluru", "reason": "canonical IPL team name"},
  {"alias": "Delhi Daredevils", "canonical": "Delhi Capitals", "reason": "IPL franchise renaming"},
  {"alias": "Delhi Capitals", "canonical": "Delhi Capitals", "reason": "canonical IPL team name"},
  {"alias": "Kings XI Punjab", "canonical": "Punjab Kings", "reason": "IPL franchise renaming"},
  {"alias": "Punjab Kings", "canonical": "Punjab Kings", "reason": "canonical IPL team name"},
  {"alias": "USA", "canonical": "United States of America", "reason": "international team short name"},
  {"alias": "U.S.A.", "canonical": "United States of America", "reason": "international team abbreviation"},
  {"alias": "United States", "canonical": "United States of America", "reason": "international team short name"},
  {"alias": "United States Of America", "canonical": "United States of America", "reason": "capitalisation variant"},
  {"alias": "United States of America", "canonical": "United States of America", "reason": "canonical international team name"},
  {"alias": "UAE", "canonical": "United Arab Emirates", "reason": "international team abbreviation"},
  {"alias": "U.A.E.", "canonical": "United Arab Emirates", "reason": "international team abbreviation"},
  {"alias": "United Arab Emirates", "canonical": "United Arab Emirates", "reason": "canonical international team name"},
  {"alias": "Czech Republic", "canonical": "Czechia", "reason": "country-name variant"},
  {"alias": "Czechia", "canonical": "Czechia", "reason": "canonical international team name"},
  {"alias": "The Netherlands", "canonical": "Netherlands", "reason": "international team-name variant"},
  {"alias": "Netherlands", "canonical": "Netherlands", "reason": "canonical international team name"},
  {"alias": "Hong Kong, China", "canonical": "Hong Kong", "reason": "international team-name variant"},
  {"alias": "Hong Kong", "canonical": "Hong Kong", "reason": "canonical international team name"},
  {"alias": "P.N.G.", "canonical": "Papua New Guinea", "reason": "international team abbreviation"},
  {"alias": "PNG", "canonical": "Papua New Guinea", "reason": "international team abbreviation"},
  {"alias": "Papua New Guinea", "canonical": "Papua New Guinea", "reason": "canonical international team name"},
  {"alias": "Swaziland", "canonical": "Eswatini", "reason": "country-name variant"},
  {"alias": "Eswatini", "canonical": "Eswatini", "reason": "canonical international team name"},
  {"alias": "Türkiye", "canonical": "Turkey", "reason": "country-name variant"},
  {"alias": "Turkey", "canonical": "Turkey", "reason": "canonical international team name"},
  {"alias": "West Indies", "canonical": "West Indies", "reason": "canonical international team name"}
]

CONFIG = {
  "project": {
    "name": "SupeRR Selector Season 2, Round 1 (Rajasthan Royals)",
    "objective": "Identify the Most Reliable Players Under Pressure",
    "version": "season2_round1_five_competitions_revision_update3_rr2027_roster_needs_retained",
    "reproducible_seed": 42
  },
  "scope": {
    "legacy_source": "Original canonical CSV dataset only for IPL and men's T20I modelling, filtered to match dates 2022-01-01 through 2026-12-31 inclusive. Exclude 30 tournament matches (Mens T20 Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025), 16 matches without ball data, 25 matches without win_team_id, and 7 unreliable DLS/revised matches (16753, 19959, 20182, 23674, 24822, 28334, 48610) -> 805 modelling matches, 190,763 regulation delivery rows.",
    "legacy_competitions": ["IPL", "T20I"],
    "legacy_date_range": ["2022-01-01", "2026-12-31"],
    "legacy_retained_matches": 805,
    "legacy_regulation_deliveries": 190763,
    "new_competition_sources": {
      "SA20": "sat.zip",
      "Big Bash League": "bbl.zip",
      "Syed Mushtaq Ali Trophy": "sma.zip"
    },
    "new_competition_date_range": ["2023-01-01", "2026-12-31"],
    "new_competitions_gender": "male",
    "new_competitions_format": "six-ball-over T20",
    "excluded_sources": {
      "T20 Blast": "ntb.zip excluded per user brief; not in scope.",
      "The Hundred": "hnd.zip excluded per user brief; not in scope.",
      "archive (1).zip": "Excluded per user brief; aggregate SMAT tables are not modelling source data.",
      "archive.zip": "Squad reference only (squads.csv for IPL 2026 exclusion checks); its deliveries and aggregate statistics are not modelling data.",
      "ipl.zip": "Cricsheet IPL YAML archive used ONLY for registry-ID to canonical player_id alignment; never as modelling deliveries.",
      "t20s.zip": "Cricsheet T20 YAML archive filtered to men's international T20 matches 2022-2026 used ONLY for registry-ID to canonical player_id alignment; never as modelling deliveries."
    },
    "modelling_competitions": ["IPL", "T20I", "SA20", "BBL", "SMAT"],
    "overall_cohort": "Original canonical CSV IPL/T20I match observations dated 2022-01-01 through 2026-12-31 (805 matches), plus Cricsheet SA20, BBL, and SMAT match observations dated 2023-01-01 through 2026-12-31 after match exclusions (D/L, VJD revised targets, no-result, tied, and Super Over innings 3/4). Cricsheet IPL and T20I archives are identity crosswalk sources only."
  },
  "random_seeds": {
    "global": 42,
    "python_random": 42,
    "numpy": 42,
    "scikit_learn": 42,
    "lightgbm": 42,
    "bootstrap_rng": 42
  },
  "identity": {
    "cricsheet_registry_ids_global": True,
    "new_archive_merge_key": "Cricsheet registry ID only; no name-based joining across archives.",
    "alignment_method": {
      "match_key": ["match_date", "unordered_pair_of_normalised_team_names"],
      "alias_table": ALIASES,
      "date_window": ["2022-01-01", "2026-12-31"],
      "required_ball_agreement": 0.95,
      "candidate_collision_resolution": "compare first-innings total, retain exact-total candidates, then sequence compare (>=95% agreement required)",
      "mapping_rule": "map registry IDs to player_ids using STRIKER, BOWLER and DISMISSED-PLAYER evidence only: leading candidate must carry >=95% of that evidence. NON-STRIKER evidence may support a mapping but must never veto or create a conflict (resolving 6a26221c = Aiden Markram = 3605).",
      "unresolved_registry_player_policy": "assign CS_<registry_id>; never merge by name.",
      "safety_exclusion_rule": "any registry ID remaining unresolved with ANY candidate canonical ID in the exclusion set (IPL 2026 roster, squads.csv, Pakistan/Pakistan A) is excluded from selectable pool."
    },
    "venue_model_feature": {
      "enabled": True,
      "canonical_ipl_t20i": "LEGACY_VENUE_ID::<venue_id>",
      "new_cricsheet_competitions": "CRICSHEET_VENUE::<info.venue>",
      "cross_source_venue_crosswalk": False,
      "limitation": "Ground overlaps across sources are not merged; treated as distinct categorical levels."
    }
  },
  "validation": {
    "group_column": "match_group_id",
    "group_cv": "GroupKFold",
    "group_cv_splits": 5,
    "cross_fitting": {
      "unit": "match_group_id; unique across all legacy and Cricsheet feeds",
      "same_fold_assignment_for_all_four_expected_value_models": True,
      "out_of_fold_predictions": True,
      "no_player_id_features": True,
      "match_balanced_weights": "Each match contributes equal total weight; row weights are inverse training-row count within match, rescaled to sum to N_train"
    },
    "holdout_cohort_2026": {
      "ipl_t20i_holdout_matches": 140,
      "rule": "Used once for out-of-sample validation; not leaked into development training"
    }
  },
  "pressure": {
    "primary": {
      "definition": "composite_pressure_index",
      "formula": "P = C * B * W * D",
      "components": {
        "closeness_C": "clip(1 - 2 * |WP - 0.5|, 0, 1)",
        "ball_scarcity_B": "1 - balls_remaining / innings_limit_balls",
        "wicket_scarcity_W": "1 - wickets_in_hand / 11.0",
        "demand_D": "sigmoid(-resource_margin_z) = 1 / (1 + exp(clip(resource_margin_z, -30, 30)))"
      },
      "resource_margin_definition": "expected_remaining_runs(balls_left, wickets_lost, competition, stage) - runs_required, standardised by stage uncertainty (sd)",
      "global_thresholds": {
        "high_pressure_top_pct": 25,
        "ordinary_bottom_pct": 50,
        "middle_excluded_pct": 25,
        "computed_globally_across_all_modelling_deliveries": True
      },
      "reproduction_target_original_only": {
        "total_deliveries": 190763,
        "high_pressure": 47691,
        "ordinary": 95381,
        "middle_excluded": 47691
      },
      "wp_leverage_failure_and_fallback": {
        "wp_leverage_definition": "WP(after six) - WP(after wicket)",
        "failure_reason": "Failed pre-registered checks: over-1 high-pressure rate was 98.6% vs maximum 30% requirement, and chase pressure failed to rise late in the innings.",
        "fallback_action": "Fallback to pre-registered Composite Pressure Index P = C * B * W * D."
      }
    },
    "robustness_configurations": [
      "wp_leverage_pressure",
      "rule_based_pressure_index",
      "per_innings_thresholds",
      "chase_only_pressure"
    ]
  },
  "win_probability_model": {
    "model_family": "separate regularised logistic regressions by innings with match-balanced weights and Platt calibration",
    "innings_1_model": {
      "model": "sklearn LogisticRegression",
      "solver": "liblinear",
      "C": 0.3,
      "max_iter": 1500,
      "random_state": 42
    },
    "innings_2_model": {
      "model": "sklearn LogisticRegression",
      "solver": "liblinear",
      "C": 0.3,
      "max_iter": 1500,
      "random_state": 42
    },
    "features_innings_1": [
      "legal_balls_remaining", "legal_over", "wickets_lost", "wickets_in_hand",
      "team_score", "current_run_rate", "log_balls_remaining", "balls_left_x_wickets_in_hand",
      "expected_remaining_runs", "stage_uncertainty_runs", "resource_margin_z",
      "venue_expected_first_innings_score", "competition_type", "venue_id"
    ],
    "features_innings_2": [
      "legal_balls_remaining", "legal_over", "wickets_lost", "wickets_in_hand",
      "team_score", "current_run_rate", "log_balls_remaining", "balls_left_x_wickets_in_hand",
      "expected_remaining_runs", "stage_uncertainty_runs", "resource_margin_z",
      "venue_expected_first_innings_score", "runs_required", "required_run_rate",
      "required_minus_current_run_rate", "rrr_x_balls_left", "competition_type", "venue_id"
    ],
    "terminal_states": {
      "tenth_wicket_chase": "WP = 0 unless target already reached",
      "tenth_wicket_first_innings": "converts to opposing chase-start WP",
      "chase_six_cap": "six is capped at runs required to win"
    },
    "calibration": "pooled Platt scaling via LogisticRegression(C=1e6) on inner-OOF log-odds"
  },
  "performance_models": {
    "model_family": "LightGBM out-of-fold by match_id (5-fold GroupKFold)",
    "lightgbm_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.05,
      "num_leaves": 31,
      "min_child_samples": 20,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "random_state": 42,
      "n_jobs": 1,
      "verbose": -1
    },
    "common_features_all_sources": [
      "innings", "over", "wickets_lost", "balls_left", "required_rate", "competition", "opposition_elo", "venue"
    ],
    "excluded_features": [
      "batter_hand", "pace_or_spin", "bowling_type", "player_id", "pressure_class"
    ],
    "four_production_targets": {
      "expected_batter_runs": {
        "task": "regression",
        "target": "batter_runs",
        "filter": "batter_faced == True",
        "loss": "regression_l2",
        "opposition_elo": "opposition_elo_batter"
      },
      "batter_dismissal_probability": {
        "task": "classification",
        "target": "batter_dismissal_target",
        "filter": "all regulation deliveries",
        "loss": "binary",
        "opposition_elo": "opposition_elo_batter"
      },
      "expected_bowler_charged_runs": {
        "task": "regression",
        "target": "bowler_charged_runs",
        "filter": "all regulation deliveries",
        "loss": "regression_l2",
        "opposition_elo": "opposition_elo_bowler"
      },
      "bowler_credited_wicket_probability": {
        "task": "classification",
        "target": "bowler_credited_wicket_target",
        "filter": "all regulation deliveries",
        "loss": "binary",
        "opposition_elo": "opposition_elo_bowler"
      }
    },
    "state_specific_wicket_cost": {
      "model": "remaining innings runs regression on same 8 features",
      "formula": "lambda_state = max(0, E[remaining_runs | wickets_lost] - E[remaining_runs | wickets_lost + 1])",
      "terminal_wickets_lost_ge_10": "lambda_state = 0.0"
    }
  },
  "player_metrics": {
    "primary_batter_metric": "BV",
    "primary_bowler_metric": "BoV",
    "batting_value_formula": "BV_i = (batter_runs_i - E[batter_runs_i|context]) - lambda_state * (dismissed_i - P(batter_dismissal_i|context))",
    "bowling_value_formula": "BoV_i = (E[bowler_charged_runs_i|context] - bowler_charged_runs_i) + lambda_state * (bowler_credited_wicket_i - P(bowler_wicket_i|context))",
    "dismissed_player_attribution": "dismissed_player_id receives -lambda_state * (1 - P(dismissal)); striker receives residual batter runs + lambda_state * P(dismissal) on survival",
    "per_player_competition_aggregates": [
      "pressure_value_per_ball", "ordinary_value_per_ball", "pressure_delta",
      "pressure_balls", "ordinary_balls", "matches_with_pressure_balls",
      "positive_pressure_match_share", "pressure_match_value_sd"
    ]
  },
  "empirical_bayes": {
    "sampling_variance": "max(cluster_robust_variance_by_match, pooled_per_ball_variance * estimated_design_effect / n_balls)",
    "delta_sampling_variance": "includes cluster-robust covariance between pressure and ordinary matches",
    "hyperparameter_estimation_cohort": {
      "pressure_metric": "n_pressure_matches >= 5 and pressure_balls >= 60",
      "delta_metric": "n_common_matches >= 5 and ordinary_balls >= 40"
    },
    "between_player_variance": "DerSimonian-Laird tau^2 jointly across competitions with competition fixed effects",
    "prior_mean_mu0": {
      "method": "precision_weighted_mean across competitions",
      "fallback_check": "if |mu0_global - ball_weighted_mean| >= 0.05, use ball_weighted_mean (declared fallback)"
    },
    "zero_tau2_policy": {
      "rule": "tau^2 = 0 is a valid empirical result representing complete homogeneity in that component. Do NOT discard or fall back to raw values. Set that component shrunk deviation z-score = 0; derive rating from remaining component(s).",
      "bowlers_primary": "If delta tau^2 == 0, bowler rating is 100% z(pressure_value_per_ball)."
    },
    "competition_recentering": {
      "rule": "shrunk_deviation = shrunk_posterior - competition_prior_mean; exposure_weighted_average across competitions; standardise (z-score) within final eligible pool"
    },
    "final_rating_formula": {
      "batters": "0.7 * z(pressure_value) + 0.3 * z(delta)",
      "bowlers": "1.0 * z(pressure_value) if delta_tau2 == 0 else 0.7 * z(pressure_value) + 0.3 * z(delta)",
      "rank_order": "descending final rating"
    },
    "acceptance_checks": {
      "EB-1": "|mu0 - ball_weighted_mean| < 0.05",
      "EB-2": "STRICT: mean sampling variance < observed variance of raw pressure value per ball on player-competition rows with >= 100 pressure balls (failure recorded and carried to limitations, no changes)",
      "EB-3": "no exact numerical ties in top 30",
      "EB-4": "Spearman(shrunk, raw) > 0.5 for players with >= 200 pressure balls",
      "EB-5": "|corr(rating, competition_exposure_share)| < 0.3 for T20I share and all competition shares"
    }
  },
  "eligibility_and_exclusions": {
    "modelling_pool_rule": "Excluded players REMAIN in all modelling, baselines, Elo, and EB hyperparameter estimation; they are filtered only at final ranking.",
    "retired_players_exclusion": {
      "rationale": "Explicit user directive to exclude retired players from final ranking deliverables and top 10/top 30 selection pools.",
      "excluded_players": [
        {"player_id": "PID_1107", "player_name": "Mohammad Nabi", "role": "All-Rounder / Batter", "status": "Retired"},
        {"player_id": "PID_1318", "player_name": "Andre Russell", "role": "All-Rounder / Batter", "status": "Retired"},
        {"player_id": "PID_2930", "player_name": "Rassie van der Dussen", "role": "Batter", "status": "Retired"},
        {"player_id": "CS_bd93fea4", "player_name": "Peter Siddle", "role": "Bowler", "status": "Retired"},
        {"player_id": "PID_4237", "player_name": "Tabraiz Shamsi", "role": "Bowler", "status": "Retired"},
        {"player_id": "PID_143", "player_name": "Dinesh Karthik", "role": "Wicketkeeper / Batter", "status": "Retired"},
        {"player_id": "PID_220", "player_name": "David Warner", "role": "Batter", "status": "Retired"}
      ]
    },
    "role_specific_exclusions": {
      "rationale": "Explicit user directive to remove Chris Jordan from batters deliverables and Matthew Short from bowlers deliverables.",
      "batters_excluded": [
        {"player_id": "PID_786", "player_name": "Chris Jordan", "reason": "Explicit user instruction to remove Jordan from batters"}
      ],
      "bowlers_excluded": [
        {"player_id": "PID_3727", "player_name": "Matthew Short", "reason": "Explicit user instruction to remove Matthew Short from bowlers"}
      ]
    },
    "ipl_2026_contract_retention_rule": {
      "rationale": "Exclusion applies strictly to players who remain under active IPL contract. 76 players released by their franchises are removed from the exclusion set and made eligible for selection. 14 players traded out to another IPL team remain under contract and stay excluded. Per UPDATE 3, RR authoritative roster of 19 players are under contract (Burger and Shanaka are retained and excluded; Hetmyer, Milne, Maphaka, Deshpande, Kuldeep Sen, Yudhvir Singh, Vignesh Puthur, Ravi Singh, Emanjot Chahal are released).",
      "released_players_count": 76,
      "traded_out_stay_excluded_count": 14,
      "released_players_mapping_file": "pipeline_outputs/config/released_players_mapping.csv",
      "careful_distinctions": [
        "Kuldip Yadav (released, CSK, player_id 20693) is NOT Kuldeep Yadav (traded out, stays excluded, player_id 4287)",
        "Abishek Porel = Abhishek Porel (released, DC, player_id 19714)",
        "AM Ghazanfar = Allah Ghazanfar (released, MI, player_id 32862)",
        "Zak Foulkes = Zakary Foulkes (released, CSK, player_id 29066)",
        "Shahrukh Khan = M Shahrukh Khan (released, GT, player_id 3782)",
        "Mohammed Izhar (released, MI, player_id 71311) is NOT Mohammed Shami (traded out, stays excluded)",
        "Dishan Madushanka = Dilshan Madushanka (released, SRH, player_id 7689)",
        "Onkar Tarmale = Omkar Tarmale (released, SRH, player_id 66513)",
        "Nandre Burger (PID_7312) is RETAINED by RR per UPDATE 3 (stays excluded)",
        "Dasun Shanaka (PID_2436) is RETAINED by RR per UPDATE 3 (stays excluded)",
        "Emanjot Chahal (PID_43965) is RELEASED by RR per UPDATE 3 (eligible for pool)",
        "Sam Curran is TRADED OUT to another IPL team (stays excluded)"
      ]
    },
    "ipl_2026_squad_exclusion": {
      "sources": [
        "competition_team_player_mapping.csv for comp_name 'IPL 2026' (275 canonical IDs)",
        "squads.csv (257 players) matched by name and franchise",
        "Gerald Coetzee (player_id 6628) union",
        "known aliases: KL Rahul = Lokesh Rahul, Tilak Varma = Namboori Tilak Varma, Pat Cummins = Patrick Cummins, Shahrukh Khan = M Shahrukh Khan, Shivam Dubey = Shivam Dube, T Natrajan = T Natarajan, MS Dhoni = Mahendra Singh Dhoni, Will Jacks = William Jacks",
        "distinction: Shubham Dubey (RR) is separate from Shivam Dube (CSK)",
        "CS_ identities matched by initial + surname and flagged for review"
      ]
    },
    "pakistan_exclusion": {
      "sources": [
        "competition_team_player_mapping.csv where team_name is 'Pakistan' or 'Pakistan A' (68 canonical IDs)",
        "Cricsheet men's T20I matches 2022-2026 Pakistan team roster registry IDs"
      ]
    },
    "bangladesh_exclusion": {
      "rationale": "Explicit user directive to remove Bangladeshi players from both ranking lists and top 10 / top 30 deliverables.",
      "sources": [
        "competition_team_player_mapping.csv where team_name contains 'Bangladesh' or 'Bangladesh A'",
        "Cricsheet men's T20I matches 2022-2026 Bangladesh team roster registry IDs (CS_<registry_id>)"
      ]
    },
    "floors_main_list": {
      "primary_floor": ">= 100 pressure balls AND >= 12 pressure matches total across competitions",
      "fallback_ladder": "if role has < 30 eligible players: try 80 balls / 10 matches, then 60 balls / 8 matches",
      "frozen_appendix_floors": "batters >= 150 balls / 20 matches; bowlers >= 120 balls / 15 matches"
    },
    "smat_watchlist": {
      "floors": ">= 30 pressure balls AND >= 4 pressure matches in SMAT (strict floor yields empty list per data limitations)",
      "reduced_floors": ">= 15 pressure balls AND >= 2 pressure matches in SMAT (per user directive for dedicated domestic ranking)",
      "unrated_exploratory_table": ">= 20 pressure balls AND >= 2 pressure matches in SMAT"
    }
  },
  "reliability_and_robustness": {
    "bootstrap": {
      "iterations": 1000,
      "seed": 42,
      "resampling_unit": "match-level cluster bootstrap within final eligible pool",
      "metrics": ["rank_95_ci_lower", "rank_95_ci_upper", "rank_median", "top_10_prob", "top_30_prob"]
    },
    "split_half": {
      "iterations": 200,
      "resampling_unit": "random 50/50 disjoint match splits",
      "metrics": ["spearman_rho", "top_10_overlap", "top_30_overlap"],
      "chance_baseline": "k * k / N (e.g. 10*10/N for top 10)"
    },
    "ablations": [
      "no_competition_recentering",
      "equal_competition_weighting",
      "no_opposition_elo",
      "per_innings_threshold",
      "chase_only_pressure",
      "rule_based_pressure",
      "wp_leverage_pressure",
      "shrinkage_off_raw_values",
      "ipl_only",
      "t20i_only",
      "leave_out_sa20",
      "leave_out_bbl",
      "leave_out_smat",
      "bowlers_with_raw_30_delta"
    ],
    "near_tie_ordering_rule": {
      "threshold": "consecutive rating gap <= 0.05 z",
      "tie_breakers": [
        "bootstrap top-10 probability (descending)",
        "pressure matches count (descending)",
        "positive pressure match share (descending)"
      ],
      "prohibition": "Never reorder materially separated players (> 0.05 z); no reputation adjustments."
    }
  },
  "rr_2027_suitability_stage": {
    "rationale": "Evaluates Rajasthan Royals 2027 franchise tactical need-fit alongside pure pressure reliability per UPDATE 3 authoritative roster and needs hierarchy.",
    "assumed_roster": {
      "wicket_keepers": ["Dhruv Jurel", "Donovan Ferreira", "Lhuan-dre Pretorius"],
      "batters": ["Yashasvi Jaiswal", "Riyan Parag", "Vaibhav Sooryavanshi", "Shubham Dubey", "Aman Rao (Aman Rao Perala)"],
      "all_rounders": ["Ravindra Jadeja", "Dasun Shanaka"],
      "bowlers": ["Jofra Archer", "Nandre Burger", "Ravi Bishnoi", "Sandeep Sharma", "Sushant Mishra", "Brijesh Sharma", "Yash Raj Punja", "Mohammed Shami", "M Siddharth"],
      "total_roster_count": 19,
      "current_overseas_count": 5,
      "overseas_players": ["Donovan Ferreira", "Lhuan-dre Pretorius", "Dasun Shanaka", "Jofra Archer", "Nandre Burger"],
      "open_overseas_slots": 3,
      "open_squad_slots": 6
    },
    "needs_hierarchy": [
      {"id": 1, "description": "right-handed top-order anchor/number 3, positions 1-4", "priority_weight": 1.0, "type_dependent": True, "required_hand": "RHB", "position_condition": "pos_1_4_share >= 0.50"},
      {"id": 2, "description": "power finisher, positions 5-7", "priority_weight": 1.0, "type_dependent": False, "position_condition": "pos_5_7_share >= 0.25", "power_condition": "death_balls_faced_share >= 0.20 or boundary_rate >= 0.15"},
      {"id": 3, "description": "middle-overs control, bowlers with a large share of legal balls in overs 7-16, pace or spin", "priority_weight": 0.8, "type_dependent": False, "phase_condition": "middle_overs_7_16_share >= 0.40"},
      {"id": 4, "description": "death-overs pace specialist (depth only)", "priority_weight": 0.6, "type_dependent": True, "required_type": "Pace", "phase_condition": "death_overs_17_20_share >= 0.20"},
      {"id": 5, "description": "opener backup, positions 1-3", "priority_weight": 0.6, "type_dependent": False, "position_condition": "pos_1_3_share >= 0.50"},
      {"id": 6, "description": "spin variety: off-spin or wrist-spin", "priority_weight": 0.5, "type_dependent": True, "required_type": "Spin", "condition": "off-spin or wrist-spin"},
      {"id": 7, "description": "seam all-rounder at 6-7 (Shanaka covers it)", "priority_weight": 0.3, "type_dependent": True, "required_type": "Pace", "condition": "balls_faced >= 30 and pos_5_7_share >= 0.25 and legal_balls_bowled >= 30"}
    ],
    "candidate_pool": "Performance Top 30 of each role",
    "cs_neutral_rule": "For CS_ players, hand/type is unknown; need-fit for hand- or type-dependent needs is set to neutral 0.5 with verify manually flag; no external knowledge or reputation.",
    "need_fit_formula": "max priority weight among satisfied needs (0 if none), rescaled to [0, 1]",
    "rr_suitability_score_formula": "0.75 * z(performance_rating) + 0.25 * z(need_fit), z-scored within candidate set",
    "sensitivity_weights": [
      {"name": "pure_performance", "w_perf": 1.00, "w_fit": 0.00},
      {"name": "primary_suitability", "w_perf": 0.75, "w_fit": 0.25},
      {"name": "high_need_weight", "w_perf": 0.50, "w_fit": 0.50}
    ]
  }
}

config_json = json.dumps(CONFIG, indent=2)
config_path = OUT_DIR / 'model_config.json'
config_path.write_text(config_json, encoding='utf-8')

sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
sha_path = OUT_DIR / 'model_config.sha256'
sha_path.write_text(f'{sha}  model_config.json\n', encoding='utf-8')

print(f'Wrote model_config.json ({len(config_json):,} chars)')
print(f'SHA-256: {sha}')
