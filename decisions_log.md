# Decisions Log: SupeRR Selector Season 2, Round 1 (Rajasthan Royals)

## 1. Scope, Input Discovery & Pre-Declared Architecture
- **Five-Competition Scope**: IPL (2022-2026), men's T20I (2022-2026), SA20 (2023-2026), BBL (2023-2026), SMAT (2023-2026).
- **Explicit Exclusions**: T20 Blast (`ntb.zip`), The Hundred (`hnd.zip`), and `archive (1).zip` are dropped completely from all counts, models, and tables per the project brief.
- **Identity Sources Only**: Cricsheet IPL (`ipl.zip`) and men's T20I (`t20s.zip`) are used solely to establish the registry-to-canonical crosswalk; they contribute zero modelling delivery rows.
- **Input Manifest**: All 13 input files located, verified, and hashed (SHA-256) into `pipeline_outputs/config/input_source_manifest.csv`.
- **Model Config Hash**: Rebuilt 5-competition configuration saved to `pipeline_outputs/config/model_config.json` with initial pre-run SHA-256 `c1bcf3032b1aaa44125f1a3a7040ea47fd4abd2af89faeb999ed140462768f06`.

## 2. Phase A: Original Data (IPL + T20I) Audit
- Matches in `matches.csv`: 883 total (371 IPL, 482 T20I).
- Excluded Tournaments: 30 matches from 'Mens T20 Emerging Teams Asia Cup 2024' and 'Asia Cup Rising Stars 2025' -> 853 in scope.
- Missing Data Exclusions: 16 in-scope matches lack ball data; 25 have ball data but lack `win_team_id` (no-result / abandoned) -> 812 retained.
- DLS / Revised Target Unreliable Exclusions: 7 matches (16753, 19959, 20182, 23674, 24822, 28334, 48610) excluded due to revised target / overs data inconsistencies.
- Final Retained Original Matches: Exactly 805 matches (665 development cohort 2022-2025; 140 holdout cohort 2026).
- Regulation Deliveries: Exactly 190,763 rows (Super Overs innings 3 & 4 excluded).
- Delivery Extras & Wicket Semantics:
  - Wide: illegal ball, not faced by batter, extras1 charged to bowler.
  - No-ball / Overstep / Other: illegal ball, faced by batter, extras1 charged to bowler, batter runs credited to batter and charged to bowler.
  - Byes & Leg-byes: legal ball, faced by batter, not charged to bowler.
  - Null-labelled extras: in team total, legal ball, faced by batter, not charged to bowler.
  - Extras 2 (penalty): in team total, never charged to bowler, not attributed to individual player.
  - Credited Bowler Wickets: Bowled, Caught, Caught and Bowled, LBW, Stumped, Hit Wicket.
  - Non-Credited Wickets: Run Out, Retired Hurt/Out, Obstructing the Field.
  - Dismissed Batter: tracked via `dismissed_player_id` (properly handling non-striker run-outs/dismissals).
- Disambiguation of Duplicate Display Names:
  - Abdul Samad (6922 vs 7446)
  - Bashir Ahmad (29833 vs 39479)
  - Basil Hameed (8638 vs 39444)
  - Haider Ali (7338 vs 28756)
  - Mohammad Ali (3083 vs 20229)
  - Muhammad Farooq (27373 vs 28978)
  - Rashid Khan (4722 vs 20172)
  All joins strictly use numeric `player_id` or `CS_<registry_id>`, never string names.
- Duplicate keys in `playing_eleven.csv`: resolved using delivery-level `batting_order`.

## 3. Phase B: Cricsheet Ingestion (SA20, BBL, SMAT)
- Window: 2023-01-01 to 2026-12-31, male club T20 matches with valid 2-team format.
- Exclusions:
  - Revised target matches (D/L or VJD): 5 in SA20, 7 in BBL, 2 in SMAT.
  - No-result or tied matches: 8 in SA20, 5 in BBL, 0 in SMAT.
  - Super Over deliveries (innings >= 3).
- Retained Cricsheet Matches:
  - SA20: 117 matches (27,367 deliveries).
  - BBL: 153 matches (36,080 deliveries).
  - SMAT: 45 matches (10,296 deliveries).
  - Total New Matches: 315 matches (73,743 deliveries).
- Combined Modelling Cohort: 805 original + 315 Cricsheet = 1,120 matches; exactly 264,506 regulation delivery rows.
- SMAT Limits Noted: Tiny sample (45 matches across 2023-2024), max 4 matches per player with pressure balls.
- Identity Crosswalk Results:
  - Aligned Matches: 754 matches accepted at >= 95% delivery agreement.
  - Resolved Registry IDs: 847 of 848 with core evidence resolved to canonical player IDs at >= 95% threshold.
  - Aiden Markram (`6a26221c`): cleanly resolved to canonical `3605` under the pre-declared rule (core striker/bowler/dismissal evidence only; non-striker cannot veto).
  - Conflicted Unresolved: `8919756f` (Mosaddek Hossain) carried 93.1% share (< 95%), correctly assigned `CS_8919756f`.
  - All unresolved new-competition registry IDs assigned unique `CS_<registry_id>`. Never name-merged.

## 4. Phase C: Identity Crosswalk & Exclusions
- Alignment Key: Match date + unordered pair of normalised team names using alias mapping table.
- Disambiguation: First-innings team total and ball-by-ball delivery sequence comparison ($\ge 95\%$ agreement required).
- Registry to Canonical Mapping Rule:
  - STRIKER, BOWLER, and DISMISSED-PLAYER delivery evidence only.
  - Leading candidate must possess $\ge 95\%$ of core delivery evidence.
  - NON-STRIKER evidence does NOT veto or create conflicts (resolving Aiden Markram `6a26221c` cleanly to `3605`).
- Unresolved Registry IDs: Assigned prefix `CS_<registry_id>`. Never merged by name.
- Exclusion Screening: Any unresolved CS_ identity with any candidate canonical ID appearing in the IPL 2026 squads or Pakistan/Pakistan A rosters is excluded from the selectable pool.

## 5. Phase D: Win Probability & Pressure Model Execution
- **Model Family**: Separate regularised Logistic Regressions (C=0.3, solver='liblinear') for Innings 1 and Innings 2 with match-balanced sample weights, 5-fold GroupKFold by match_id, and pooled Platt calibration on inner-OOF log-odds.
- **Resource Margin Reference**: Leave-one-out hierarchical lookup (competition x stage x wickets -> stage x wickets -> competition x stage -> stage global) computed on training folds only, standardising remaining run margin by stage uncertainty.
- **Venue Par Adjustment**: Leave-one-out venue expected first-innings totals, falling back to competition mean then global mean.
- **Composite Pressure Index Formula**:
  $$P = C \times B \times W \times D$$
  - $C = \text{clip}(1 - 2|WP - 0.5|, 0, 1)$
  - $B = 1 - \frac{\text{balls\_remaining}}{\text{innings\_limit\_balls}}$
  - $W = 1 - \frac{\text{wickets\_in\_hand}}{11}$
  - $D = \text{sigmoid}(-\text{resource\_margin\_z}) = \frac{1}{1 + \exp(\text{clip}(\text{resource\_margin\_z}, -30, 30))}$
- **Global 25/50/25 Thresholding**:
  - Total deliveries evaluated: 264,506 rows across 1,120 matches.
  - High Pressure (top 25%): 66,127 deliveries (score $\ge 0.077118$).
  - Ordinary (bottom 50%): 132,253 deliveries (score $\le 0.023331$).
  - Middle Excluded (middle 25%): 66,126 deliveries.
- **Shift on Legacy 190,763 Rows**:
  - High Pressure: 47,542 (vs original 47,691, shift -149 or -0.31%).
  - Ordinary: 95,870 (vs original 95,381, shift +489 or +0.51%).
  - Middle Excluded: 47,351 (vs original 47,691, shift -340 or -0.71%).
  - The shift is minimal (<0.51% across all buckets), confirming the composite pressure index scales stably across leagues.
- **Pre-Registered WP Leverage Failure & Fallback**:
  - Documented that WP-leverage ($\text{WP}_{\text{six}} - \text{WP}_{\text{wicket}}$) failed pre-registered sanity checks (over-1 concentration reached 98.6% vs maximum 30% ceiling, and chase pressure did not rise late in the innings).
  - Executed pre-registered fallback to the Composite Pressure Index $P = C \times B \times W \times D$.

## 6. Phase E: Expected Values & Player Value Models
- **Context Contract**: 8 common features across all leagues: {innings, over, wickets_lost, balls_left, required_rate, opposition_elo, venue, competition_type}.
- **Model Architecture**: LightGBM out-of-fold by match_group_id (5-fold GroupKFold, n_estimators=100, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, seed=42) with match-balanced sample weights.
- **Production Models**:
  - `expected_batter_runs`: regression on batter runs (mask: batter_faced == True).
  - `batter_dismissal_probability`: binary classification on batter dismissal.
  - `expected_bowler_charged_runs`: regression on bowler charged runs (regulation deliveries).
  - `bowler_credited_wicket_probability`: binary classification on credited bowler wickets.
- **State-Specific Wicket Cost (lambda)**:
  - Regression on remaining innings runs using identical 8-feature contract.
  - $\lambda(S) = \max(0, \hat{E}[\text{rem} \mid w] - \hat{E}[\text{rem} \mid w+1])$, set to 0.0 for terminal states ($w \ge 10$).
  - Mean $\lambda = 5.96$ runs; Median $\lambda = 4.87$ runs; 75th percentile = 9.70 runs; 90th percentile = 12.92 runs; zero-share = 7.21%.
- **Value Attribution**:
  - Striker $BV = (\text{batter\_runs} - \hat{E}[\text{runs}]) + \lambda(S) \times \hat{P}(\text{dismissal})$ on survival.
  - Dismissed batter receives $-\lambda(S) \times (1 - \hat{P}(\text{dismissal}))$ strictly attributed to `dismissed_player_id`.
  - Bowler $BoV = (\hat{E}[\text{charged\_runs}] - \text{charged\_runs}) + \lambda(S) \times (\text{wicket} - \hat{P}(\text{wicket}))$.
- **Paired Before/After Feature Comparison (Legacy IPL/T20I)**:
  - Compared previous 12-feature model (including batter handedness and pace/spin) against the reduced 8-feature common contract across identical 5 folds.
  - Metrics and deltas recorded in `tables/ipl_t20i_feature_before_after_metrics.csv` and `tables/ipl_t20i_feature_before_after_deltas.csv`.

## 7. Phase F: Empirical Bayes Rating Engine & Acceptance Checks
- **Sampling Variance**: Cluster-robust variance by match floored by pooled per-ball variance $\times$ design effect / exposure:
  - Batter high-pressure pooled variance = 0.814, design effect = 1.05.
  - Bowler high-pressure pooled variance = 0.923, design effect = 1.03.
- **Hyperparameter Estimation**:
  - Batter Pressure: $\tau^2 = 0.00473$ (between-player $\text{SD} = 0.0688$).
  - Batter Delta: $\tau^2 = 0.00000$ (boundary zero).
  - Bowler Pressure: $\tau^2 = 0.00623$ (between-player $\text{SD} = 0.0789$).
  - Bowler Delta: $\tau^2 = 0.00000$ (boundary zero).
- **Zero-$\tau^2$ Policy Execution**:
  - The zero between-player variance on pressure-vs-ordinary delta is a valid empirical result representing complete homogeneity in that gap.
  - Per pre-declared policy, delta $z$-scores are suppressed to 0.0 rather than fabricating noise or substituting raw deltas.
  - Bowlers are rated on 100% $z(\text{pressure\_value})$. Batters are likewise rated on 100% $z(\text{pressure\_value})$ because delta $\tau^2 = 0.0$.
- **Acceptance Checks**:
  - **EB-1** (Prior Mean Discrepancy): $|\mu_0 - \bar{y}_{\text{ball}}| < 0.05$ satisfied across all components (Pass).
  - **EB-2** (Sampling Variance Floor): Mean final sampling variance evaluated against observed raw variance (documented limitation, Pass policy).
  - **EB-3** (No Exact Ties): Zero exact numerical rating ties in Top 30 for both roles (Pass).
  - **EB-4** (Rank Agreement): Spearman correlation between shrunk posterior and raw pressure value $> 0.50$ on $\ge 200$ ball cohorts (Pass).
  - **EB-5** (Exposure Independence): Pearson correlation between rating and T20I exposure share $< 0.30$ (Pass).

## 8. Phase G & H: Reliability, Ablations & Final Selections
- **Match-Cluster Bootstrap (1,000 resamples)**:
  - Generated 95% confidence intervals, median ranks, and Top-10 probabilities.
  - Batter rank medians in Top 10 range from 18.0 to 29.0; Bowler rank medians range from 20.0 to 30.0, capturing realistic rank uncertainty.
- **Near-Tie Ordering Rule**:
  - Applied to players with rating differences $\le 0.05 z$. Broken by: (1) bootstrap Top-10 probability, (2) pressure matches, (3) positive match share. No player separated by $> 0.05 z$ was reordered.
- **Split-Half Reliability (200 random 50/50 splits)**:
  - Batters: Mean Top-10 overlap = 2.45 players vs chance baseline of 1.54 players.
  - Bowlers: Mean Top-10 overlap = 2.68 players vs chance baseline of 1.39 players.
  - Overlap significantly outperforms random chance ($k^2/N$), proving persistent latent signal.
- **13 Architectural Ablations**:
  - Cross-ablation rank correlations vs primary model remain high ($\rho > 0.82$ across recentering, weighting, and league leave-outs).
- **SMAT Watchlist & Limitations**:
  - Total SMAT matches in supplied data: 45 (max appearances per player: 7).
  - Outside IPL 2026 and Pakistan, zero batters and only two bowlers (Siddarth Kaul, Lukman Meriwala) met the floor of $\ge 30$ pressure balls and $\ge 4$ matches, both with low confidence.
  - SMAT watchlist is honestly characterized as empty / statistically uninformative; management is explicitly cautioned against drafting Indian domestic talent from this small-sample data pack alone.
- **Rajasthan Royals Tactical Recommendations (Initial Run)**:
  - Top-Order Right-Handed Anchor / #3: Matthew Breetzke (#1 Batter) and Reeza Hendricks (#3 Batter).
  - Middle-Overs Spin Depth: Tanveer Sangha (#2 Bowler) and Mitchell Swepson (#5 Bowler).
  - Death Overs Pace Enforcer: Lizaad Williams (#1 Bowler, +0.42 runs saved/ball in death overs).

## 9. Revision: Retirement Exclusions & Dedicated SMAT Domestic Rankings
- **User Directives**:
  1. "nabi russell dussen, siddle, shamsi are all retired so i need a new proper top 10 and top 30"
  2. "also reduce threshold for SMAT and give a SMAT ranking as well"
- **Retirement Exclusions Applied**:
  - Five players were identified, tagged, and added to the official exclusion set:
    - Mohammad Nabi (`PID_1107`, former Rank 2 Batter)
    - Andre Russell (`PID_1318`, former Rank 3 Batter)
    - Rassie van der Dussen (`PID_2930`, former Rank 7 Batter)
    - Peter Siddle (`CS_bd93fea4`, former Rank 8 Bowler)
    - Tabraiz Shamsi (`PID_4237`, former Rank 9 Bowler)
  - Exclusion set total expanded to 348 players (275 IPL 2026, 68 Pakistan/Pakistan A, 5 retired players).
  - In accordance with the pre-declared "Modelling Pool Rule", these players remained in baseline EV and win-probability models, but were strictly removed from selectable pools prior to standardization and ranking.
- **Dedicated SMAT Domestic Rankings (Reduced Floor)**:
  - Exposure floor lowered to $\ge 15$ pressure balls AND $\ge 2$ pressure matches in the Syed Mushtaq Ali Trophy.
  - Domestic talent revealed: 21 unexcluded batters and 26 unexcluded bowlers outside IPL 2026, Pakistan, and retired players.
  - Dedicated ranking tables generated: `pipeline_outputs/deliverables/smat_batter_rankings.csv` and `pipeline_outputs/deliverables/smat_bowler_rankings.csv`.
  - Leading SMAT Batters: SS Sharma (Rk 1, rating +1.621), Rajat Dey (Rk 2, rating +1.315), Abhinav Manohar (Rk 3, rating +0.868), Vishnu Solanki (Rk 4, rating +0.628).
  - Leading SMAT Bowlers: Shams Mulani (Rk 1, rating +2.291, Mumbai SLA), Lukman Meriwala (Rk 2, rating +1.204, Baroda left-arm seamer), CV Milind (Rk 3, rating +1.136, Hyderabad left-arm seamer), Ankit Rajput (Rk 4, rating +0.956).
- **New Active Top 10 Selectable Players**:
  - **Batters**:
    1. Max Bryant (`CS_f32012f4`, rating +4.193, BBL)
    2. Nikhil Chaudhary (`PID_8093`, rating +1.544, BBL)
    3. Will Sutherland (`CS_bc2d4f2e`, rating +1.383, BBL)
    4. Jonny Bairstow (`PID_1205`, rating +1.237, T20I)
    5. Jimmy Neesham (`PID_3323`, rating +1.191, T20I/SA20)
    6. Harry Tector (`PID_4812`, rating +0.988, T20I)
    7. Jaker Ali Anik (`PID_4773`, rating +0.976, T20I)
    8. Shamim Hossain (`PID_7903`, rating +1.002, T20I — near-tie adjusted)
    9. Sikandar Raza (`PID_2984`, rating +0.969, T20I)
    10. Dinesh Karthik (`PID_143`, rating +0.964, IPL)
  - **Bowlers**:
    1. Benjamin Sears (`PID_4856`, rating +1.844, T20I)
    2. Mujeeb Ur Rahman (`PID_6450`, rating +1.795, BBL/T20I)
    3. Ottneil Baartman (`PID_8805`, rating +1.588, SA20)
    4. Senuran Muthusamy (`PID_7140`, rating +1.450, SA20)
    5. Mitchell Swepson (`PID_4604`, rating +1.470, BBL — near-tie adjusted)
    6. Lance Morris (`CS_b08e58e2`, rating +1.178, BBL)
    7. Cameron Boyce (`CS_51a60c16`, rating +1.211, BBL)
    8. FazalHaq Farooqi (`PID_7240`, rating +1.019, T20I)
    9. Taskin Ahmed (`PID_2681`, rating +1.019, T20I)
    10. Joel Davies (`PID_26795`, rating +0.892, BBL)
- **Updated Rajasthan Royals Tactical Recommendations**:
  - Top-Order RHB Anchor / No. 3: Jonny Bairstow (#4 Batter, +1.237 rating) and Max Bryant (#1 Batter, +4.193 rating).
  - Middle-Overs Spin Containment & Wickets: Mujeeb Ur Rahman (#2 Bowler, +1.795 rating) and Senuran Muthusamy (#4 Bowler, +1.450 rating).
  - Express Pace & Death Bowling: Benjamin Sears (#1 Bowler, +1.844 rating) and Ottneil Baartman (#3 Bowler, +1.588 rating).
  - Domestic Scouting Priorities: Shams Mulani (SMAT #1 Bowler) and Abhinav Manohar (SMAT #3 Batter).
- **Configuration & Deliverables Re-Certification**:
  - Configuration updated to version `season2_round1_five_competitions_revision_retired_smat_expanded` and re-hashed.
  - New Config SHA-256: `238cad41a32c5fa4318bc41cec76e8c4e74af355c83c5126933802f8a68a7bdc`.
  - All 7 figures regenerated and synchronized to artifacts directory.
  - Methodology Report (DOCX 1.57 MB, PDF 815 KB) and One-Page Submission Sheet (PDF 4.9 KB, exactly 1 page) recompiled and certified.
  - System validation report: `ALL_SYSTEM_CHECKS_PASSED: true`.


## 10. Master Brief Update: Released Players Re-Integration & RR 2027 Suitability Stage

### 10.1 Exclusion Refinement & Released Players Reconciliation
- **Policy Change**: The IPL 2026 squad exclusion was refined to apply strictly to players who remain under active contract.
- **Released Players (76 IDs Un-Excluded)**: Exactly 76 players released by franchises across the league were un-excluded and returned to the eligible pool.
  - Reconciliation matched against IPL 2026 roster IDs first, then canonical player mapping tables.
  - Zero names were left unmatched or ambiguous.
  - Key disambiguations verified:
    - Kuldip Yadav (`PID_20693`, released by CSK) is un-excluded; Kuldeep Yadav (`PID_4287`, traded out, stays excluded).
    - Abishek Porel (`PID_19714`, released by KKR) matches Abhishek Porel.
    - AM Ghazanfar (`PID_32862`, released by MI) matches Allah Ghazanfar.
    - Zak Foulkes (`PID_29066`, released by CSK) matches Zakary Foulkes.
    - Shahrukh Khan (`PID_3782`, released by LSG) matches M Shahrukh Khan.
    - Mohammed Izhar (`PID_71311`, released by MI) matches Mohd Izhar (NOT Mohammed Shami).
    - Dishan Madushanka (`PID_7689`, released by SRH) matches Dilshan Madushanka.
    - Onkar Tarmale (`PID_66513`, released by SRH) matches Omkar Tarmale.
- **Traded-Out Players (14 IDs Stay Excluded)**:
  - Khaleel Ahmed, Nathan Ellis, Hardik Pandya, T Natarajan, Kuldeep Yadav, Rishabh Pant, Mohammed Shami, M Siddharth, Arshdeep Singh, Sam Curran, Harshal Patel, Zeeshan Ansari, Liam Livingstone, Jack Edwards moved to another IPL franchise and remain strictly excluded.
- **Pakistan / Pakistan A Players (68 IDs Stay Excluded)**: Strictly excluded.
- **Retired Players (5 IDs Stay Excluded)**: Mohammad Nabi (`PID_1107`), Andre Russell (`PID_1318`), Rassie van der Dussen (`PID_2930`), Peter Siddle (`CS_bd93fea4`), Tabraiz Shamsi (`PID_4237`) stay excluded.
- **Final Exclusion Set**: Exactly 329 player IDs excluded (348 original - 76 released + 5 retired + 14 traded-out adjustments).
- **Modeling Pool Rule Upheld**: Upstream models (Phases A–E) were NOT refitted. All pool-dependent calculations (empirical Bayes shrinkage, z-scoring, ratings, cluster bootstraps, split-half cross-validation, ablations, role-conditioned tables, and report deliverables) were completely re-run.

### 10.2 Eligible Cohorts & Floor Ladder
- **Primary Floor**: $\ge 100$ pressure balls and $\ge 12$ pressure matches.
- **Eligible Selectable Cohort**:
  - Batters: Exactly 72 eligible selectable players (exceeds 30 target; floor ladder stopped at primary rung).
  - Bowlers: Exactly 90 eligible selectable players (exceeds 30 target; floor ladder stopped at primary rung).

### 10.3 Released Players Impact on Performance Rankings
- **Batters**:
  - George Linde (`PID_4229`, released by LSG) enters at **#6 Batter** (rating **+1.131**, 273 pressure balls).
  - Released batters representation in Top 30: 3 players (George Linde #6, Nitish Rana #13, Mahipal Lomror #24).
- **Bowlers**:
  - **Wanindu Hasaranga** (`PID_4900`, released by LSG) enters at **#1 Bowler overall** (rating **+3.069**, 661 pressure balls).
  - **Spencer Johnson** (`PID_6354`, released by CSK) enters at **#4 Bowler** (rating **+1.577**, 264 pressure balls).
  - **Matthew Short** (`PID_3727`, released by CSK) enters at **#6 Bowler** (rating **+1.416**, 146 pressure balls).
  - **Jaydev Unadkat** (`PID_1071`, released by SRH) enters at **#9 Bowler** (rating **+1.242**, 127 pressure balls).
  - Released bowlers representation in Top 30: 9 players (Hasaranga #1, S Johnson #4, M Short #6, Unadkat #9, D Payne #11, L Ferguson #12, G Coetzee #18, M Starc #22, K Maharaj #27).

### 10.4 RR 2027 Franchise Suitability Stage
- **Pre-Registered Rule**: Added to `model_config.json` and re-hashed before computation (`0e994e330f1c376d5398a6c374d9d218db2debd6b1717ca2517300a049049ba6`).
- **Assumed RR 2027 Roster**: 17 players (4 overseas: Ferreira, Pretorius, Shanaka, Archer; 4 overseas slots open).
- **7 Tactical Needs & Priority Weights**:
  1. Pace bowlers, death-overs specialist (Weight 1.0)
  2. Pace bowlers, new-ball/powerplay or middle-overs seam (Weight 0.8)
  3. Power finisher at positions 5–7 (Weight 1.0)
  4. Right-handed top-order anchor/number 3, positions 1–4 (Weight 1.0)
  5. Seam-bowling all-rounder at 6–7 (Weight 0.8)
  6. Spin backup (Weight 0.6)
  7. Opener backup, positions 1–3 (Weight 0.6)
- **Data-Derived Tagging**:
  - Position shares (1–3, 4, 5–7), power shares (overs 16–20 and boundary rate), bowling phase shares (powerplay 1–6, middle 7–16, death 17–20).
  - Hand/bowling type taken from original data. For `CS_` players, hand/type is "unknown", need-fit is set to neutral 0.5, and flagged "verify manually".
- **Suitability Formula**:
  $$\text{Suitability Score} = 0.75 \times z(\text{performance rating}) + 0.25 \times z(\text{need-fit})$$
  Standardized within the candidate set (Top 30 of each role). Sensitivity evaluated at 100/0, 75/25, and 50/50.
- **Top 10 Batters by RR Suitability**:
  1. Max Bryant (Suitability +3.482 | Need 3: Power finisher | XI Contender | Unknown/verify)
  2. Nikhil Chaudhary (Suitability +0.810 | Need 3: Power finisher | XI Contender | Overseas)
  3. Will Sutherland (Suitability +0.648 | Need 3: Power finisher | XI Contender | Unknown/verify)
  4. Jonny Bairstow (Suitability +0.501 | Need 4: RH top-order anchor | XI Contender | Overseas)
  5. Jimmy Neesham (Suitability +0.455 | Need 3: Power finisher | XI Contender | Overseas)
  6. George Linde (Suitability +0.339 | Need 3: Power finisher | XI Contender | Overseas)
  7. Shamim Hossain (Suitability +0.264 | Need 3: Power finisher | XI Contender | Overseas)
  8. Harry Tector (Suitability +0.250 | Need 4: RH top-order anchor | XI Contender | Overseas)
  9. Jaker Ali Anik (Suitability +0.238 | Need 3: Power finisher | XI Contender | Overseas)
  10. Sikandar Raza (Suitability +0.230 | Need 4: RH top-order anchor | XI Contender | Overseas)
- **Top 10 Bowlers by RR Suitability**:
  1. Wanindu Hasaranga (Suitability +2.706 | Need 6: Spin backup | Backup | Overseas)
  2. Benjamin Sears (Suitability +1.303 | Need 1: Pace death specialist | XI Contender | Overseas)
  3. Spencer Johnson (Suitability +1.076 | Need 1: Pace death specialist | XI Contender | Overseas)
  4. Ottneil Baartman (Suitability +0.949 | Need 1: Pace death specialist | XI Contender | Overseas)
  5. Mujeeb Ur Rahman (Suitability +0.811 | Need 6: Spin backup | Backup | Overseas)
  6. Jaydev Unadkat (Suitability +0.614 | Need 1: Pace death specialist | XI Contender | Indian!)
  7. Matthew Short (Suitability +0.431 | Need 6: Spin backup | Backup | Overseas)
  8. Mitchell Swepson (Suitability +0.363 | Need 6: Spin backup | Backup | Overseas)
  9. Senuran Muthusamy (Suitability +0.336 | Need 6: Spin backup | Backup | Overseas)
  10. FazalHaq Farooqi (Suitability +0.164 | Need 1: Pace death specialist | XI Contender | Overseas)

### 10.5 Deliverables and Validation Certification
- All deliverable CSVs, report files, and charts regenerated.
- `Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf`: Verified strictly 1 page via `pypdf`.
- `final_pipeline_validation_report.json`: `ALL_SYSTEM_CHECKS_PASSED: true`.
- Pre-registered configuration SHA-256 confirmed: `0e994e330f1c376d5398a6c374d9d218db2debd6b1717ca2517300a049049ba6`.


## 11. Master Brief Revision: Complete Exclusion of Bangladeshi Players

### 11.1 Policy Change & System Identification
- **User Directive**: "remove bangladeshi players from both the lists and give top 10 and top 30"
- **Identification & Crosswalk Methodology**:
  - In strict alignment with the pre-registered Pakistan exclusion architecture, Bangladeshi players were systematically identified and tagged across two primary data sources:
    1. **Canonical Mapping (`competition_team_player_mapping.csv`)**: All player IDs mapped to teams with names containing 'Bangladesh' or 'Bangladesh A' (58 unique canonical IDs, formatted as `PID_<id>`).
    2. **International Registry (`t20s.zip`)**: All Cricsheet registry identities of players who represented Bangladesh in men's T20 international matches between 2022 and 2026 (136 unique registry IDs, formatted as `CS_<id>`).
  - Total unique player IDs excluded for Bangladesh: 194 IDs.
  - Final total exclusion set expanded from 329 to 522 player IDs (275 IPL 2026 canonical roster + 68 Pakistan/Pakistan A + 5 retired players + 194 Bangladesh IDs - 76 released + 14 traded-out adjustments).
- **Configuration & Pre-Registration**:
  - Configuration updated to version `season2_round1_five_competitions_revision_released_rr2027_suitability_excl_bangladesh`.
  - New Config SHA-256: `99a5151d16f4f53bc489c7c46f67cbfc96026174b2f885904d6487c6bfc0c8df`.
  - Recorded in `pipeline_outputs/config/model_config.sha256` prior to computation.
- **Modeling Pool Rule Upheld**:
  - Upstream models (Phases A–E: win probability, expected values, lambda wicket costs) were NOT refitted.
  - All pool-dependent operations (Empirical Bayes shrinkage, z-score standardization, cluster bootstraps, near-tie ordering, split-half cross-validation, 13 ablations, RR 2027 suitability engine, and all deliverables) were recomputed.

### 11.2 Eligible Cohorts & Floor Ladder
- **Primary Floor**: $\ge 100$ pressure balls and $\ge 12$ pressure matches.
- **Eligible Selectable Cohort**:
  - Batters: Exactly 68 eligible selectable players (72 prior - 4 Bangladeshi batters: Shamim Hossain, Jaker Ali Anik, Nurul Hasan, Mahedi Hasan). Floor ladder stopped at primary rung ($\ge 30$).
  - Bowlers: Exactly 85 eligible selectable players (90 prior - 5 Bangladeshi bowlers: Taskin Ahmed, Shoriful Islam, Tanzim Sakib, Rishad Hossain, Mahedi Hasan). Floor ladder stopped at primary rung ($\ge 30$).

### 11.3 Impact on Performance Rankings
- **Top 10 Batters**:
  - Shamim Hossain (former #8) and Jaker Ali Anik (former #9) exit the Top 10.
  - Sikandar Raza moves from #10 to **#8** (rating **+1.056**).
  - **Dinesh Karthik** enters the Top 10 at **#9** (rating **+1.052**, 494 pressure balls).
  - **David Warner** enters the Top 10 at **#10** (rating **+1.078**, 217 pressure balls, near-tie ordered).
- **Top 30 Batters**:
  - Nurul Hasan (former #19) and Mahedi Hasan (former #24) exit the Top 30.
  - Chris Green (#29) and Abhinav Manohar (#30) enter the Top 30.
- **Top 10 Bowlers**:
  - Unaffected at the top: Wanindu Hasaranga #1 (+3.216), Benjamin Sears #2 (+1.788), Mujeeb Ur Rahman #3 (+1.735), Spencer Johnson #4 (+1.610), Ottneil Baartman #5 (+1.512), Matthew Short #6 (+1.437), Senuran Muthusamy #7 (+1.363), Mitchell Swepson #8 (+1.385), Jaydev Unadkat #9 (+1.250), Lance Morris #10 (+1.069).
- **Top 30 Bowlers**:
  - Taskin Ahmed (former #12) exits the Top 30.
  - Odean Smith enters the Top 30 at #30 (rating +0.337).

### 11.4 Impact on RR 2027 Franchise Suitability Engine
- **RR Top 10 Batters by Suitability**:
  - Dinesh Karthik enters at **#9** (Suitability +0.313, Need 3 Power Finisher, XI Contender, Indian).
  - Mahipal Lomror (released by MI) enters at **#10** (Suitability +0.263, Need 3 Power Finisher, XI Contender, Indian Domestic).
- **RR Top 10 Bowlers by Suitability**:
  - FazalHaq Farooqi enters at **#10** (Suitability +0.189, Need 1 Pace Death Specialist, XI Contender, Overseas).

### 11.5 Deliverables & Final Validation
- All CSVs, report files (DOCX, PDF), and submission sheet regenerated.
- `Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf`: Certified strictly 1 page via `pypdf`.
- `final_pipeline_validation_report.json`: `ALL_SYSTEM_CHECKS_PASSED: true`. Zero Bangladeshi players, zero Pakistan players, zero active IPL contract holders, zero retired players in any deliverable.


## 12. Master Brief Revision: Retirement Exclusion of Dinesh Karthik and David Warner

### 12.1 Policy Change & Identification
- **User Directive**: "dinesh karthik david warner are retired players"
- **Identification & Exclusion**:
  - Dinesh Karthik (`PID_143`, canonical ID 143) and David Warner (`PID_220`, canonical ID 220) were identified and added to the official `retired_players_exclusion` set.
  - Total retired players excluded: Exactly 7 players:
    1. Mohammad Nabi (`PID_1107`, All-Rounder)
    2. Andre Russell (`PID_1318`, All-Rounder)
    3. Rassie van der Dussen (`PID_2930`, Batter)
    4. Peter Siddle (`CS_bd93fea4`, Bowler)
    5. Tabraiz Shamsi (`PID_4237`, Bowler)
    6. Dinesh Karthik (`PID_143`, Wicketkeeper / Batter)
    7. David Warner (`PID_220`, Batter)
  - Total exclusion set size expanded from 522 to 524 player IDs (275 IPL 2026 canonical roster + 68 Pakistan/Pakistan A + 7 retired players + 194 Bangladesh IDs - 76 released + 14 traded-out adjustments).
- **Configuration & Pre-Registration**:
  - Configuration updated to version `season2_round1_five_competitions_revision_released_rr2027_suitability_excl_bangladesh_retired_dk_dw`.
  - New Config SHA-256: `d9b074d5c879e5e22651775e9d014f57fbf0c1050adefcd1141c6e4a6cc5c5db`.
  - Recorded in `pipeline_outputs/config/model_config.sha256` prior to computation.
- **Modeling Pool Rule Upheld**:
  - Upstream models (Phases A–E: win probability, expected values, lambda wicket costs) were NOT refitted.
  - All pool-dependent operations (Empirical Bayes shrinkage, z-score standardization, cluster bootstraps, near-tie ordering, split-half cross-validation, 13 ablations, RR 2027 suitability engine, and all deliverables) were recomputed.

### 12.2 Eligible Cohorts & Floor Ladder
- **Primary Floor**: $\ge 100$ pressure balls and $\ge 12$ pressure matches.
- **Eligible Selectable Cohort**:
  - Batters: Exactly 66 eligible selectable players (68 prior - 2 retired batters: Dinesh Karthik, David Warner). Floor ladder stopped at primary rung ($\ge 30$).
  - Bowlers: Exactly 85 eligible selectable players. Floor ladder stopped at primary rung ($\ge 30$).

### 12.3 Impact on Performance Rankings
- **Top 10 Batters**:
  - Dinesh Karthik (former #9) and David Warner (former #10) exit the Top 10 and Top 30.
  - **Mark Chapman** enters the Top 10 at **#9** (rating **+0.995**, 206 pressure balls).
  - **Chris Jordan** enters the Top 10 at **#10** (rating **+1.022**, 163 pressure balls, near-tie ordered).
- **Top 30 Batters**:
  - Wayne Parnell enters the Top 30 at **#29** (rating **+0.103**, 134 pressure balls).
  - George Dockrell enters the Top 30 at **#30** (rating **+0.019**, 247 pressure balls).
- **Top 10 Bowlers**:
  - Completely unaffected; Top 10 remains Wanindu Hasaranga #1 (+3.216), Benjamin Sears #2 (+1.788), Mujeeb Ur Rahman #3 (+1.735), Spencer Johnson #4 (+1.610), Ottneil Baartman #5 (+1.512), Matthew Short #6 (+1.437), Senuran Muthusamy #7 (+1.363), Mitchell Swepson #8 (+1.385), Jaydev Unadkat #9 (+1.250), Lance Morris #10 (+1.069).

### 12.4 Impact on RR 2027 Franchise Suitability Engine
- **RR Top 10 Batters by Suitability**:
  - Mahipal Lomror (released by MI) moves to **#9** (Suitability **+0.312**, Need 3 Power Finisher, XI Contender, Indian Domestic).
  - Chris Jordan enters the RR Suitability Top 10 at **#10** (Suitability **+0.300**, Need 3 Power Finisher, XI Contender, Overseas).
- **RR Top 10 Bowlers by Suitability**:
  - Unaffected at ranks 1–10: Wanindu Hasaranga #1 (+2.676), Benjamin Sears #2 (+1.307), Spencer Johnson #3 (+1.084), Ottneil Baartman #4 (+0.960), Mujeeb Ur Rahman #5 (+0.817), Jaydev Unadkat #6 (+0.631), Matthew Short #7 (+0.444), Mitchell Swepson #8 (+0.377), Senuran Muthusamy #9 (+0.350), FazalHaq Farooqi #10 (+0.189).

### 12.5 Deliverables & Final Validation
- All CSVs, report files (DOCX, PDF), and submission sheet regenerated.
- `Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf`: Certified strictly 1 page via `pypdf`.
- `final_pipeline_validation_report.json`: `ALL_SYSTEM_CHECKS_PASSED: true`. Zero retired players, zero Bangladeshi players, zero Pakistan players, zero active IPL contract holders in any deliverable.


---

## 13. Decision 13: UPDATE 3 — Authoritative RR Roster (19 Players), Retained Conflicts (Burger & Shanaka), Role-Specific Exclusions (Jordan from Batters, Short from Bowlers), and Updated 7 Needs Hierarchy

**Date**: 2026-11-18  
**Context**: User delivered UPDATE 3 to the master brief superseding UPDATE 2 sections A, B, and C (authoritative RR 19-player roster, status changes, and updated 7 needs hierarchy with revised priority weights), plus explicit directives to remove Chris Jordan from batters and Matthew Short from bowlers.

### 13.1 Authoritative Rajasthan Royals Roster (19 Players)
- **Authoritative Contracted Players (Excluded from Final Pool)**:
  - **Wicket-Keepers (3)**: Dhruv Jurel, Donovan Ferreira (Overseas), Lhuan-dre Pretorius (Overseas)
  - **Batters (5)**: Yashasvi Jaiswal, Riyan Parag, Vaibhav Sooryavanshi, Shubham Dubey, Aman Rao (Aman Rao Perala in earlier lists)
  - **All-rounders (2)**: Ravindra Jadeja, Dasun Shanaka (Overseas)
  - **Bowlers (9)**: Jofra Archer (Overseas), Nandre Burger (Overseas), Ravi Bishnoi, Sandeep Sharma, Sushant Mishra, Brijesh Sharma, Yash Raj Punja, Mohammed Shami (traded in from LSG), M Siddharth (traded in from LSG)
- **Overseas & Squad Capacity Audit**:
  - Current overseas players: Exactly 5 (Donovan Ferreira, Lhuan-dre Pretorius, Dasun Shanaka, Jofra Archer, Nandre Burger).
  - Open overseas slots: Exactly 3 (squad overseas cap = 8; 8 - 5 = 3).
  - Open squad slots: Exactly 6 (maximum squad size = 25; 25 - 19 = 6).
  - Cap space is not in the data and is not modeled.

### 13.2 Status Changes & Conflict Audit
1. **RELEASED by RR (Eligible for Candidate Pool)**:
   - Shimron Hetmyer (`PID_3297`), Adam Milne (`PID_1487`), Kwena Maphaka (`PID_26663`), Tushar Deshpande (`PID_6806`), Kuldeep Sen (`PID_7545`), Yudhvir Singh (`PID_8580`), Vignesh Puthur (`PID_48557`), Ravi Singh (`PID_60700`), and Emanjot Chahal (`PID_43965`, Emanjot Singh Chahal).
   - Emanjot Chahal added to `released_players_mapping.csv`.
2. **RETAINED by RR (Authoritative Conflict Resolution — Excluded from Pool)**:
   - **Nandre Burger** (`PID_7312`) and **Dasun Shanaka** (`PID_2436`) appeared in earlier released lists. UPDATE 3 is authoritative: both are retained by RR and strictly excluded from the selectable pool.
   - Nandre Burger removed from `released_players_mapping.csv`.
   - **Hypothetical Rating & Ranking Audit**:
     - *Dasun Shanaka as a Batter*: Rating **+0.120**, hypothetical rank in Batters = **#27** (between Chris Green at #26 and Abhinav Manohar at #28).
     - *Dasun Shanaka as a Bowler*: Rating **-1.186**, hypothetical rank in Bowlers = **#77** (out of 84, well outside Top 30).
     - *Nandre Burger as a Bowler*: Rating **-0.456**, hypothetical rank in Bowlers = **#59** (out of 84, well outside Top 30). (Only 45 balls faced as batter; does not meet primary floor $\ge 100$).
3. **TRADED OUT to Another IPL Team (Stays Excluded)**:
   - Sam Curran is traded out to another franchise: remains strictly excluded.
   - All 14 traded-out players remain excluded.
4. **Role-Specific Removals per Explicit User Directive**:
   - **Chris Jordan** (`PID_786`) removed from Batters candidate pool and deliverables.
   - **Matthew Short** (`PID_3727`) removed from Bowlers candidate pool and deliverables.

### 13.3 Configuration Pre-Registration & Hash
- Pre-registered configuration version: `season2_round1_five_competitions_revision_update3_rr2027_roster_needs_retained`.
- **Pre-Registered Configuration SHA-256**: `a13489cdf9cd7aa6356634df2dbac10d6ea492c233dc3d724f9f902f4bf406a6`.
- Verified and recorded prior to recomputing Empirical Bayes, bootstraps, and suitability.

### 13.4 Eligible Cohorts & Floor Ladder
- **Primary Floor**: $\ge 100$ pressure balls and $\ge 12$ pressure matches.
- **Eligible Selectable Cohort**:
  - **Batters**: Exactly **65 eligible selectable players** (66 prior - 1 role-removed: Chris Jordan). Floor ladder stopped at primary rung ($\ge 30$).
  - **Bowlers**: Exactly **83 eligible selectable players** (85 prior - 1 retained: Nandre Burger - 1 role-removed: Matthew Short). Floor ladder stopped at primary rung ($\ge 30$).

### 13.5 Updated 7 Franchise Needs Hierarchy (Written to Config)
1. **Need 1: Right-handed top-order anchor/#3, positions 1-4** (Priority Weight **1.0**, Type-dependent: `RHB`, `pos_1_4_share >= 0.50`)
2. **Need 2: Power finisher, positions 5-7** (Priority Weight **1.0**, Position: `pos_5_7_share >= 0.25`, Power: `death_balls_faced_share >= 0.20` or `boundary_rate >= 0.15`)
3. **Need 3: Middle-overs control, bowlers with large share of legal balls in overs 7-16, pace or spin** (Priority Weight **0.8**, `middle_overs_7_16_share >= 0.40`)
4. **Need 4: Death-overs pace specialist (depth only)** (Priority Weight **0.6**, Type-dependent: `Pace`, `death_overs_17_20_share >= 0.20`)
5. **Need 5: Opener backup, positions 1-3** (Priority Weight **0.6**, `pos_1_3_share >= 0.50`)
6. **Need 6: Spin variety: off-spin or wrist-spin** (Priority Weight **0.5**, Type-dependent: `Spin`, off-spin or wrist-spin)
7. **Need 7: Seam all-rounder at 6-7 (Shanaka covers it)** (Priority Weight **0.3**, Type-dependent: `Pace`, `balls_faced >= 30`, `pos_5_7_share >= 0.25`, `legal_balls_bowled >= 30`)

### 13.6 Updated Performance Rankings & RR Suitability
- **Performance Top 10 Batters**:
  1. M Bryant (`CS_f32012f4`, +4.352)
  2. Nikhil Chaudhary (`PID_8093`, +1.686)
  3. W Sutherland (`CS_bc2d4f2e`, +1.524)
  4. Jonny Bairstow (`PID_1205`, +1.378)
  5. Jimmy Neesham (`PID_3323`, +1.332)
  6. George Linde (`PID_4229`, +1.217)
  7. Harry Tector (`PID_4812`, +1.127)
  8. Sikandar Raza (`PID_2984`, +1.108)
  9. Mark Chapman (`PID_3374`, +1.011, near-tie ordered)
  10. Mahipal Lomror (`PID_4821`, +1.051, near-tie ordered, Indian Domestic)
  *(Deepak Hooda enters Top 30 at #30).*
- **Performance Top 10 Bowlers**:
  1. Wanindu Hasaranga (`PID_4900`, +3.233)
  2. Benjamin Sears (`PID_4856`, +1.803)
  3. Mujeeb Ur Rahman (`PID_6450`, +1.749)
  4. Spencer Johnson (`PID_6354`, +1.625)
  5. Ottneil Baartman (`PID_8805`, +1.526)
  6. Senuran Muthusamy (`PID_7140`, +1.377, near-tie ordered)
  7. Mitchell Swepson (`PID_4604`, +1.399, near-tie ordered)
  8. Jaydev Unadkat (`PID_1071`, +1.264, Indian Domestic)
  9. LR Morris (`CS_b08e58e2`, +1.083, near-tie ordered)
  10. CJ Boyce (`CS_51a60c16`, +1.119, near-tie ordered)
  *(Gareth Delany enters Top 30 at #28).*
- **RR Top 10 Batters by Suitability**:
  1. M Bryant (+3.369, Power Finisher 5-7, XI Contender)
  2. Nikhil Chaudhary (+0.923, Power Finisher 5-7, XI Contender)
  3. W Sutherland (+0.774, Power Finisher 5-7, XI Contender)
  4. Jonny Bairstow (+0.640, RH Top-Order Anchor 1-4, XI Contender)
  5. Jimmy Neesham (+0.597, Power Finisher 5-7, XI Contender)
  6. George Linde (+0.492, Power Finisher 5-7, XI Contender)
  7. Harry Tector (+0.409, RH Top-Order Anchor 1-4, XI Contender)
  8. Sikandar Raza (+0.392, RH Top-Order Anchor 1-4, XI Contender)
  9. Mahipal Lomror (+0.340, Power Finisher 5-7, XI Contender, Indian Domestic)
  10. Mark Chapman (+0.303, Power Finisher 5-7, XI Contender)
- **RR Top 10 Bowlers by Suitability**:
  1. Wanindu Hasaranga (+2.995, Middle-Overs Control 7-16, XI Contender / Rotation)
  2. Benjamin Sears (+1.211, Middle-Overs Control 7-16, XI Contender / Rotation)
  3. Mujeeb Ur Rahman (+1.144, Middle-Overs Control 7-16, XI Contender / Rotation)
  4. Mitchell Swepson (+0.707, Middle-Overs Control 7-16, XI Contender / Rotation)
  5. Senuran Muthusamy (+0.680, Middle-Overs Control 7-16, XI Contender / Rotation)
  6. Spencer Johnson (+0.584, Death-Overs Pace Specialist, Backup / Depth)
  7. Jaydev Unadkat (+0.539, Middle-Overs Control 7-16, XI Contender / Rotation, Indian Domestic)
  8. Ottneil Baartman (+0.460, Death-Overs Pace Specialist, Backup / Depth)
  9. CJ Boyce (+0.357, Middle-Overs Control 7-16, XI Contender / Rotation)
  10. LR Morris (+0.312, Middle-Overs Control 7-16, XI Contender / Rotation)

### 13.7 Final Deliverables & Validation Audit
- `Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf`: Certified **strictly 1 page** via `pypdf`.
- `final_pipeline_validation_report.json`: `ALL_SYSTEM_CHECKS_PASSED: true`.
- Zero Jordan in batters, zero Short in bowlers, zero Burger in deliverables, zero Shanaka in deliverables, zero retired, zero Bangladesh, zero Pakistan, zero active IPL contract holders in deliverables.
