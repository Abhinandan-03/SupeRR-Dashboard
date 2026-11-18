# 👑 Rajasthan Royals | SupeRR Selector Season 2 — Pressure Intelligence Portal

[![GitHub Pages](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-EA1B86?style=for-the-badge&logo=github)](https://abhinandan-03.github.io/SupeRR-Dashboard/)
[![Rajasthan Royals](https://img.shields.io/badge/Franchise-Rajasthan_Royals-004BA0?style=for-the-badge)](https://www.rajasthanroyals.com/)
[![IPL 2026/2027](https://img.shields.io/badge/Auction-IPL_2026%2F2027-0B132B?style=for-the-badge)]()

---

## 🎯 Executive Overview

This repository hosts the interactive analytics dashboard, methodology reports, and decision engine for the **Rajasthan Royals SupeRR Selector Season 2 (Round 1)**. 

Our analytics framework evaluates ball-by-ball delivery records across **IPL and T20 International (T20I)** matches (2022–2026) alongside India's premier domestic tournament (**Syed Mushtaq Ali Trophy - SMAT**) to identify the most reliable cricket players under extreme pressure.

- 📊 **Live Interactive Dashboard**: [Open Dashboard in Browser](index.html) or visit [GitHub Pages Deployment](https://abhinandan-03.github.io/SupeRR-Dashboard/)
- 📄 **Official Methodology Report (DOCX)**: [`pipeline_outputs/reports/SupeRR_Selector_Rajasthan_Royals_Methodology_Report.docx`](pipeline_outputs/reports/SupeRR_Selector_Rajasthan_Royals_Methodology_Report.docx)
- 📑 **Methodology Report (PDF)**: [`pipeline_outputs/reports/SupeRR_Selector_Rajasthan_Royals_Methodology_Report.pdf`](pipeline_outputs/reports/SupeRR_Selector_Rajasthan_Royals_Methodology_Report.pdf)
- 📋 **1-Page Submission Fast Sheet (PDF)**: [`pipeline_outputs/deliverables/Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf`](pipeline_outputs/deliverables/Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf)
- 📽️ **Executive Pitch Presentation (PPTX)**: [`pipeline_outputs/reports/Rajasthan_Royals_SupeRR_Selector_Presentation.pptx`](pipeline_outputs/reports/Rajasthan_Royals_SupeRR_Selector_Presentation.pptx)

---

## 📋 Certified Top 10 Dropdown Selections (Unstop Portal Ready)

The following 10 Batters and 10 Bowlers represent our certified, non-overlapping rankings derived strictly from the competition **Data Pack**, satisfying all qualification thresholds ($\ge 80$ pressure balls, $\ge 10$ pressure matches):

### 🏏 Top 10 Batters Under Pressure
| Rank | Player Name | Player ID | Pressure BV/ball | Clutch Delta ($\Delta_{\text{clutch}}$) | Role & Tactical Profile |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | **Jimmy Neesham** | `PID_3323` | **+0.265** | +1.121 | Elite death-overs finisher; explosive boundary strike-rate in high-leverage chases |
| **2** | **Mark Chapman** | `PID_3374` | **+0.203** | +0.427 | High-SR middle-overs controller against high-pace enforcers and spin choke |
| **3** | **George Linde** | `PID_4229` | **+0.172** | +0.129 | Dynamic lower-order left-hand finisher (SR 170+) with dual-threat bowling utility |
| **4** | **Harry Tector** | `PID_4812` | **+0.153** | +0.436 | Reliable middle-order anchor; high boundary conversion in pressure collapses |
| **5** | **Mahipal Lomror** | `PID_4821` | **+0.123** | +1.166 | #1 Domestic Indian finisher; extraordinary +1.166 clutch acceleration at death |
| **6** | **Sikandar Raza** | `PID_2984` | **+0.107** | -0.054 | Experienced all-format pressure operator; versatile across match phases 7–16 |
| **7** | **Mark Adair** | `PID_5379` | **+0.085** | +0.039 | Power-striking bowling all-rounder; late-innings boundary hitter from slot 8 |
| **8** | **Gareth Delany** | `PID_7741` | **+0.077** | +0.882 | Aggressive middle-order dasher with exceptional secondary phase gear shift |
| **9** | **Charith Asalanka** | `PID_4895` | **+0.074** | +0.147 | Left-handed anchor stabilizing innings under high required run rates |
| **10** | **Shai Hope** | `PID_4369` | **+0.071** | +0.043 | Top-order anchor with wicketkeeper utility and steady game management |

---

### 🎯 Top 10 Bowlers Under Pressure
| Rank | Player Name | Player ID | Pressure BoV/ball | Clutch Delta ($\Delta_{\text{clutch}}$) | Role & Tactical Profile |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | **Sean Abbott** | `PID_1389` | **+0.397** | +0.674 | #1 Data Pack strike bowler; elite death-over execution with wide cutters and yorkers |
| **2** | **Mayank Markande** | `PID_6683` | **+0.375** | +0.648 | #1 Domestic Indian spinner; partnership breaker in high-leverage middle overs (7–15) |
| **3** | **Mujeeb Ur Rahman** | `PID_6450` | **+0.333** | +0.322 | Mystery spinner choking run rates across powerplay and death overs |
| **4** | **Benjamin Sears** | `PID_4856` | **+0.312** | +0.317 | High-pace hard-length enforcer; hurries batters in high-pressure chases |
| **5** | **Gerald Coetzee** | `PID_6628` | **+0.277** | +0.358 | 145+ km/h express strike enforcer with proven wicket-taking burst capability |
| **6** | **Wanindu Hasaranga** | `PID_4900` | **+0.273** | +0.132 | World-class wicket-taker across 661 balls; strikes every 9 deliveries under pressure |
| **7** | **Jaydev Unadkat** | `PID_1071` | **+0.247** | +0.371 | Premier domestic left-arm closer; elite change-of-pace off-cutters and wide yorkers |
| **8** | **Akash Madhwal** | `PID_8177` | **+0.170** | -0.209 | Slingy low-trajectory death specialist with proven IPL yorker consistency |
| **9** | **Lockie Ferguson** | `PID_4692` | **+0.161** | +0.216 | 150+ km/h express pace inducing top edges and false shots in the death overs |
| **10** | **Joshua Little** | `PID_4807` | **+0.134** | +0.089 | Left-arm seam angle bowler creating early swing and cramping death-over room |

---

## 📐 Pressure Handling Methodology

### 1. Delivery-Level Bayesian Leverage ($L$)
Pressure is quantified at every delivery using the dynamic win-probability leverage index ($L$):
$$L = \max\left( \frac{\Delta \text{WP}_{\text{wicket}}}{\sigma_{\text{WP}}}, \frac{\Delta \text{WP}_{\text{boundary}}}{\sigma_{\text{WP}}} \right)$$
Deliveries with **$L \ge 1.50$** are classified as high-pressure states, reflecting situations where a single delivery swings the match probability by $>1.5$ standard deviations.

### 2. Contextual Value per Ball
For every ball, performance is benchmarked against phase- and resource-adjusted expected outcomes:
- **Batting Value (BV)**: Runs scored and boundary surplus relative to expected run value $E[\text{Runs}]$, penalized for wicket loss probability.
- **Bowling Value (BoV)**: Dot-ball pressure and wicket creation relative to expected runs conceded.

### 3. Empirical Bayes Shrinkage & Clutch Delta
To prevent small-sample noise, raw performance values are shrunk toward the population mean:
$$\theta_i = w_i \cdot \bar{x}_i + (1 - w_i) \cdot \mu_0, \quad w_i = \frac{n_i}{n_i + \tau^2 / \sigma_0^2}$$
The **Clutch Delta** evaluates whether a player elevates or declines when stakes spike:
$$\Delta_{\text{clutch}} = \text{Value}_{\text{Pressure}} - \text{Value}_{\text{Ordinary}}$$

---

## 🇮🇳 Syed Mushtaq Ali Trophy (SMAT) Domestic Talent Crucible

To solve Rajasthan Royals' specific need for uncapped domestic Indian finishers and strike spinners, the SMAT database was evaluated:

| SMAT Rank | Domestic Batter | SMAT Rating | Domestic Bowler | SMAT Rating |
|:---:|:---|:---:|:---|:---:|
| **1** | **Shubham Sharma** | +1.621 | **Shams Mulani** | +2.372 |
| **2** | **Suyash Prabhudessai** | +1.025 | **Lukman Meriwala** | +1.234 |
| **3** | **Abhinanav Manohar** | +0.868 | **Chama Milind** | +1.163 |
| **4** | **Vishnu Solanki** | +0.628 | **Darshan Nalkande** | +1.042 |
| **5** | **Sheldon Jackson** | +0.528 | **Tejas Baroka** | +0.914 |
| **6** | **Priyank Panchal** | +0.463 | **Tanush Kotian** | +0.858 |
| **7** | **Atit Sheth** | +0.436 | **Mohit Avasthi** | +0.792 |
| **8** | **Himanshu Rana** | +0.388 | **Praveen Dubey** | +0.741 |
| **9** | **Akshat Raghuwanshi** | +0.312 | **Royston Dias** | +0.689 |
| **10** | **Sanat Sangwan** | +0.276 | **Swapnil Singh** | +0.624 |

---

## 👑 Rajasthan Royals 2027 Roster Fit

The Rajasthan Royals hold **19 retained squad members** (5 overseas: Jofra Archer, Shimron Hetmyer, Rovman Powell, Dasun Shanaka, Nandre Burger) with **6 open auction slots** (3 overseas, 3 domestic):

```
Targeted Acquisitions to Complete the Championship XI:
├── Overseas Power Finisher (#7):    Jimmy Neesham / George Linde
├── Domestic Middle Finisher (#6):   Mahipal Lomror (Uncapped SMAT target: Abhinav Manohar)
├── Attacking Domestic Leg-Spinner:  Mayank Markande (SMAT depth: Shams Mulani)
├── High-Pace Express Bowler:       Gerald Coetzee / Lockie Ferguson
└── Left-Arm Tactical Closer:        Jaydev Unadkat
```

---

## 📁 Repository Structure

```
SupeRR-Dashboard/
├── index.html                                 # Clean interactive standalone web dashboard
├── README.md                                  # Repository overview and documentation
├── decisions_log.md                           # Comprehensive engineering & decision log
├── pipeline_outputs/
│   ├── reports/
│   │   ├── Rajasthan_Royals_SupeRR_Selector_Dashboard.html # Primary portal dashboard
│   │   ├── SupeRR_Selector_Rajasthan_Royals_Methodology_Report.docx # Publication report (DOCX)
│   │   ├── SupeRR_Selector_Rajasthan_Royals_Methodology_Report.pdf  # 3-page executive report (PDF)
│   │   ├── Rajasthan_Royals_SupeRR_Selector_Presentation.pptx      # 12-slide executive pitch
│   │   └── final_pipeline_validation_report.json                   # Automated audit verification
│   ├── deliverables/
│   │   └── Rajasthan_Royals_SupeRR_Selector_Submission_Sheet.pdf   # 1-page fast submission sheet
│   ├── tables/                                # Ranked leaderboard CSV tables
│   │   ├── smat_batter_rankings.csv
│   │   ├── smat_bowler_rankings.csv
│   │   ├── multi_league_batter_eb_rankings.csv
│   │   └── multi_league_bowler_eb_rankings.csv
│   └── figures/                               # Diagnostic statistical charts
└── superr_pipeline/                           # End-to-end Python processing pipeline (01–08)
    ├── 01_build_config.py
    ├── 02_identity_and_roles.py
    ├── 03_expected_value_models.py
    ├── 04_pressure_definition.py
    ├── 05_empirical_bayes_shrinkage.py
    ├── 06_league_normalisation.py
    ├── 07_portfolio_selection.py
    └── 08_final_validation.py
```

---

## 🚀 Running the Analytics Dashboard Locally

To inspect the dashboard locally:
1. Double-click `index.html` or `pipeline_outputs/reports/Rajasthan_Royals_SupeRR_Selector_Dashboard.html` in any modern web browser.
2. Alternatively, serve via Python:
   ```bash
   python -m http.server 8000
   ```
   Navigate to `http://localhost:8000` to view the live dashboard.
