#!/usr/bin/env python3
"""
Step 2: Ingest Original Data, Align Identities (Cricsheet IPL/T20I),
Ingest Cricsheet SA20, BBL, SMAT, and Build Combined Delivery Base.
"""

from pathlib import Path
import json, hashlib, zipfile, math, re, unicodedata, collections, difflib, sys, time
import polars as pl
import numpy as np
import yaml

DATA = Path('.')
INPUT_PACK = DATA / 'inputs/data_pack/Data Pack'
OUT_ID = DATA / 'pipeline_outputs/identity'
OUT_MODEL = DATA / 'pipeline_outputs/modeling'
OUT_CONFIG = DATA / 'pipeline_outputs/config'
LOGS = DATA / 'pipeline_outputs/logs'

for p in [OUT_ID, OUT_MODEL, OUT_CONFIG, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

# 1. Verify Config Hash
CFG_PATH = OUT_CONFIG / 'model_config.json'
HASH_PATH = OUT_CONFIG / 'model_config.sha256'
cfg_bytes = CFG_PATH.read_bytes()
actual_sha = hashlib.sha256(cfg_bytes).hexdigest()
expected_sha = HASH_PATH.read_text().split()[0]
assert actual_sha == expected_sha, f"Config hash mismatch: {actual_sha} != {expected_sha}"
print(f"VERIFIED CONFIG HASH: {actual_sha}")
config = json.loads(cfg_bytes.decode('utf-8'))

ALIASES = config['identity']['alignment_method']['alias_table']
ALIAS_MAP = {}
def norm_base(s):
    s = unicodedata.normalize('NFKC', str(s)).casefold().strip()
    return ''.join(ch for ch in s if ch.isalnum())

for row in ALIASES:
    ALIAS_MAP[norm_base(row['alias'])] = row['canonical']

def canon_team(s):
    raw = str(s).strip()
    n = norm_base(raw)
    return ALIAS_MAP.get(n, re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', raw)).strip())

# 2. Canonical match/team lookups
comps = pl.read_csv(INPUT_PACK / 'competitions.csv')
teams = pl.read_csv(INPUT_PACK / 'teams.csv')
team_lookup = {(int(r['comp_id']), int(r['team_id'])): str(r['team_name']) for r in teams.iter_rows(named=True)}

matches = pl.read_csv(INPUT_PACK / 'matches.csv')
matches = matches.join(comps.select(['comp_id', 'comp_name', 'match_format', 'season']), on='comp_id', how='left')
matches = matches.with_columns(pl.col('match_date').str.slice(0, 10).alias('match_date_key'))

# In-scope filter
in_scope_mask = (
    (pl.col('match_date_key') >= '2022-01-01') & (pl.col('match_date_key') <= '2026-12-31') &
    (pl.col('comp_name').str.contains(r'^IPL \d{4}$') | (pl.col('match_format') == 'T20I')) &
    (~pl.col('comp_name').is_in(['Mens T20 Emerging Teams Asia Cup 2024', 'Asia Cup Rising Stars 2025']))
)
legacy_matches = matches.filter(in_scope_mask)

# Team lookups
team_a_list = []
team_b_list = []
pair_list = []
team_lookup_issues = []

for r in legacy_matches.iter_rows(named=True):
    mid = r['match_id']
    cid = int(r['comp_id'])
    hid = int(r['home_team_id']) if r['home_team_id'] is not None else None
    aid = int(r['away_team_id']) if r['away_team_id'] is not None else None
    
    ht_csv = team_lookup.get((cid, hid)) if hid else None
    at_csv = team_lookup.get((cid, aid)) if aid else None
    
    if ht_csv is None or at_csv is None:
        team_lookup_issues.append({'match_id': mid, 'issue': 'missing_teams_csv_lookup', 'home_team': r['home_team'], 'away_team': r['away_team']})
    
    ta = canon_team(ht_csv if ht_csv is not None else r['home_team'])
    tb = canon_team(at_csv if at_csv is not None else r['away_team'])
    team_a_list.append(ta)
    team_b_list.append(tb)
    pair = tuple(sorted([norm_base(ta), norm_base(tb)]))
    pair_list.append(pair)

legacy_matches = legacy_matches.with_columns([
    pl.Series('team_a_norm', team_a_list),
    pl.Series('team_b_norm', team_b_list),
])

# 3. Ball-by-ball canonical delivery sequence extraction
bb_cols = ['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num', 'runs', 'is_wicket',
           'striker_id', 'non_striker_id', 'bowler_id', 'dismissed_player_id', 'extras1_runs', 'extras_2_runs', 'extras_type']
bb = pl.read_csv(INPUT_PACK / 'ball_by_ball.csv', columns=bb_cols, infer_schema_length=10000,
                 schema_overrides={'extras_2_runs': pl.Float64, 'extras1_runs': pl.Float64, 'runs': pl.Float64})
bb = bb.filter(pl.col('match_id').is_in(legacy_matches['match_id'].to_list()) & pl.col('innings').is_in([1, 2]))
bb = bb.sort(['match_id', 'innings', 'innings_ball_num', 'overid', 'over_ball_num'])

canonical_by_match = {}
for g_df in bb.partition_by('match_id', as_dict=False):
    mid = int(g_df['match_id'][0])
    seq = []
    rows = []
    first_total = 0
    for r in g_df.iter_rows(named=True):
        inn = int(r['innings'])
        over = int(r['overid'])
        ball = int(r['over_ball_num'])
        runs = int(r['runs']) if r['runs'] is not None else 0
        wk = bool(r['is_wicket'])
        tup = (inn, over, ball, runs, wk)
        seq.append(tup)
        rows.append(r)
        if inn == 1:
            e1 = float(r['extras1_runs'] or 0)
            e2 = float(r['extras_2_runs'] or 0)
            first_total += (runs + e1 + e2)
    canonical_by_match[mid] = {'seq': seq, 'rows': rows, 'n': len(seq), 'first_total': int(first_total)}

# Candidate index: date + unordered pair
candidates = collections.defaultdict(list)
for i, r in enumerate(legacy_matches.iter_rows(named=True)):
    key = (r['match_date_key'], pair_list[i])
    candidates[key].append(r)

# 4. Identity Alignment with Cricsheet IPL and T20I
def load_header(raw):
    head = raw.split(b'\ninnings:', 1)[0]
    return yaml.load(head, Loader=yaml.CSafeLoader) or {}

def match_date(info):
    ds = info.get('dates') or []
    return str(ds[0])[:10] if ds else None

def extract_seq_and_rows(doc):
    info = doc['info']
    people = info.get('registry', {}).get('people', {}) or {}
    inn = doc.get('innings', {})
    if isinstance(inn, dict):
        items = list(inn.items())
    else:
        items = []
        for part in inn:
            if isinstance(part, dict) and part:
                items.append(next(iter(part.items())))
    seq = []
    rows = []
    first_total = 0
    for ix, (iname, ivalue) in enumerate(items[:2], start=1):
        deliveries = ivalue.get('deliveries', []) or []
        over_counter = collections.Counter()
        for d0 in deliveries:
            if not isinstance(d0, dict) or not d0:
                continue
            ball_key, ev = next(iter(d0.items()))
            over_zero = int(math.floor(float(ball_key) + 1e-9))
            over = over_zero + 1
            over_counter[over] += 1
            ordinal = over_counter[over]
            runs = ev.get('runs', {}) or {}
            batter_runs = int(runs.get('batsman', 0) or 0)
            total_runs = int(runs.get('total', batter_runs) or 0)
            wickets = ev.get('wickets', ev.get('wicket', [])) or []
            if isinstance(wickets, dict):
                wickets = [wickets]
            seq.append((ix, over, ordinal, batter_runs, bool(wickets)))
            rows.append({'innings': ix, 'overid': over, 'over_ball_num': ordinal, 'event': ev, 'people': people, 'seq_tuple': seq[-1]})
            if ix == 1:
                first_total += total_runs
    return seq, rows, first_total, people

def sequence_compare(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    den = max(len(a), len(b), 1)
    return matched / den, matched, den, sm.get_matching_blocks()

alignment_rows = []
collision_rows = []
unmatched_rows = []
alias_usage = collections.defaultdict(set)
registry_evidence = []

archives = [('Cricsheet IPL', 'ipl.zip', 'ipl'), ('Cricsheet men T20I', 't20s.zip', 't20i')]
header_counts = collections.Counter()

print("Beginning Cricsheet alignment to canonical matches...", flush=True)
for source, fn, kind in archives:
    z = zipfile.ZipFile(DATA / fn)
    for name in z.namelist():
        if not name.lower().endswith(('.yaml', '.yml')):
            continue
        raw = z.read(name)
        try:
            doc_h = load_header(raw)
            info = doc_h.get('info', {})
        except Exception as e:
            unmatched_rows.append({'source': source, 'file': name, 'reason': 'header_parse_error', 'detail': str(e)})
            continue
        dt = match_date(info)
        if not dt or not ('2022-01-01' <= dt <= '2026-12-31'):
            continue
        if info.get('gender') != 'male' or info.get('match_type') != 'T20':
            continue
        if kind == 'ipl':
            if info.get('competition') != 'IPL':
                continue
        else:
            if info.get('team_type') != 'international':
                continue
        teams_raw = info.get('teams') or []
        if len(teams_raw) != 2:
            continue
        team_norm = [canon_team(v) for v in teams_raw]
        for raw_team, cn in zip(teams_raw, team_norm):
            if norm_base(raw_team) in ALIAS_MAP and norm_base(ALIAS_MAP[norm_base(raw_team)]) != norm_base(raw_team):
                alias_usage[(source, raw_team)].add(cn)
        key = (dt, tuple(sorted([norm_base(team_norm[0]), norm_base(team_norm[1])])))
        header_counts[(source, 'filtered')] += 1
        cand = candidates.get(key, [])
        if not cand:
            unmatched_rows.append({'source': source, 'file': name, 'date': dt, 'team1': teams_raw[0], 'team2': teams_raw[1], 'reason': 'no_date_team_candidate', 'n_candidates': 0})
            continue
        
        try:
            doc = yaml.load(raw, Loader=yaml.CSafeLoader)
            cseq, crows, ctotal, people = extract_seq_and_rows(doc)
        except Exception as e:
            unmatched_rows.append({'source': source, 'file': name, 'date': dt, 'team1': teams_raw[0], 'team2': teams_raw[1], 'reason': 'full_yaml_parse_error', 'detail': str(e)})
            continue
        
        scored = []
        for cr in cand:
            can = canonical_by_match.get(int(cr['match_id']))
            if can is None or can['n'] == 0:
                score, matched, den, blocks = 0.0, 0, max(len(cseq), 1), []
                first_total = None
            else:
                score, matched, den, blocks = sequence_compare(cseq, can['seq'])
                first_total = can['first_total']
            scored.append({'row': cr, 'score': score, 'matched': matched, 'den': den, 'blocks': blocks, 'canonical_first_total': first_total})
        
        if len(scored) > 1:
            exact_total = [q for q in scored if q['canonical_first_total'] is not None and q['canonical_first_total'] == ctotal]
            pool = exact_total if exact_total else scored
            bestscore = max(q['score'] for q in pool)
            best = [q for q in pool if abs(q['score'] - bestscore) < 1e-12]
            collision_rows.append({
                'source': source, 'file': name, 'date': dt, 'team1': teams_raw[0], 'team2': teams_raw[1],
                'n_candidates_initial': len(scored), 'first_innings_total_cricsheet': ctotal,
                'n_candidates_matching_first_total': len(exact_total), 'n_candidates_after_total': len(pool),
                'best_agreement': bestscore, 'n_best_tied': len(best),
                'candidate_match_ids': '|'.join(str(q['row']['match_id']) for q in pool),
                'decision': 'ambiguous_unaligned' if len(best) != 1 else 'unique_best_candidate'
            })
            if len(best) != 1:
                unmatched_rows.append({'source': source, 'file': name, 'date': dt, 'team1': teams_raw[0], 'team2': teams_raw[1], 'reason': 'ambiguous_same_date_team_key', 'n_candidates': len(scored)})
                continue
            chosen = best[0]
        else:
            chosen = scored[0]
        
        cr = chosen['row']
        can = canonical_by_match.get(int(cr['match_id']))
        if can is None or can['n'] == 0:
            unmatched_rows.append({'source': source, 'file': name, 'date': dt, 'team1': teams_raw[0], 'team2': teams_raw[1], 'reason': 'canonical_match_has_no_ball_data', 'canonical_match_id': int(cr['match_id'])})
            continue
        
        score, matched, den, blocks = sequence_compare(cseq, can['seq'])
        accepted = (score >= 0.95)
        alignment_rows.append({
            'source': source, 'cricsheet_file': name, 'cricsheet_date': dt, 'team1': teams_raw[0], 'team2': teams_raw[1],
            'canonical_match_id': int(cr['match_id']), 'canonical_competition': cr['comp_name'], 'canonical_match_date': cr['match_date_key'],
            'cricsheet_deliveries': len(cseq), 'canonical_deliveries': can['n'], 'matched_events': matched, 'agreement_denominator': den,
            'agreement': score, 'accepted_ge_95': accepted
        })
        
        if not accepted:
            continue
        
        for block in blocks:
            if block.size == 0:
                continue
            for off in range(block.size):
                c_item = crows[block.a + off]
                can_r = can['rows'][block.b + off]
                ev = c_item['event']
                
                # Striker
                batter_name = ev.get('batsman', ev.get('batter'))
                if batter_name and people.get(batter_name) and can_r['striker_id'] is not None:
                    registry_evidence.append({
                        'registry_id': str(people[batter_name]), 'registry_name': str(batter_name),
                        'canonical_player_id': int(can_r['striker_id']), 'role': 'striker',
                        'canonical_match_id': int(cr['match_id']), 'source': source, 'evidence_status': 'aligned_event'
                    })
                
                # Bowler
                bowler_name = ev.get('bowler')
                if bowler_name and people.get(bowler_name) and can_r['bowler_id'] is not None:
                    registry_evidence.append({
                        'registry_id': str(people[bowler_name]), 'registry_name': str(bowler_name),
                        'canonical_player_id': int(can_r['bowler_id']), 'role': 'bowler',
                        'canonical_match_id': int(cr['match_id']), 'source': source, 'evidence_status': 'aligned_event'
                    })
                
                # Non-striker (tracked for record, but rule dictates: never vetos or creates conflicts!)
                nonstr_name = ev.get('non_striker')
                if nonstr_name and people.get(nonstr_name) and can_r['non_striker_id'] is not None:
                    registry_evidence.append({
                        'registry_id': str(people[nonstr_name]), 'registry_name': str(nonstr_name),
                        'canonical_player_id': int(can_r['non_striker_id']), 'role': 'non_striker',
                        'canonical_match_id': int(cr['match_id']), 'source': source, 'evidence_status': 'aligned_event'
                    })
                
                # Dismissed player
                wlist = ev.get('wickets', ev.get('wicket', [])) or []
                if isinstance(wlist, dict):
                    wlist = [wlist]
                for ww in wlist:
                    pout = ww.get('player_out')
                    if pout and people.get(pout) and can_r['dismissed_player_id'] is not None:
                        registry_evidence.append({
                            'registry_id': str(people[pout]), 'registry_name': str(pout),
                            'canonical_player_id': int(can_r['dismissed_player_id']), 'role': 'dismissed_player',
                            'canonical_match_id': int(cr['match_id']), 'source': source, 'evidence_status': 'aligned_event'
                        })

print(f"Alignment completed. Aligned matches: {len([q for q in alignment_rows if q['accepted_ge_95']])}", flush=True)

# 5. Build Registry-to-Canonical Player ID Crosswalk
# PRE-DECLARED RULE: Use STRIKER, BOWLER, and DISMISSED-PLAYER evidence only!
# Leading candidate must carry >= 95% of that core evidence.
# NON-STRIKER evidence does not veto or create conflicts (resolving Markram 6a26221c -> 3605).
ev_df = pl.DataFrame(registry_evidence) if registry_evidence else pl.DataFrame()
core_roles = ['striker', 'bowler', 'dismissed_player']
core_ev = ev_df.filter(pl.col('role').is_in(core_roles)) if not ev_df.is_empty() else pl.DataFrame()

crosswalk_rows = []
role_conflicts = []

if not core_ev.is_empty():
    counts = (
        core_ev.group_by(['registry_id', 'canonical_player_id'])
        .len()
        .sort(['registry_id', 'len'], descending=[False, True])
    )
    totals = core_ev.group_by('registry_id').len().rename({'len': 'total_core_len'})
    joined = counts.join(totals, on='registry_id')
    joined = joined.with_columns((pl.col('len') / pl.col('total_core_len')).alias('share'))
    
    # Collect all candidate IDs and names per registry
    reg_names_map = (
        ev_df.group_by('registry_id')
        .agg(pl.col('registry_name').unique())
    )
    names_lookup = {r['registry_id']: '|'.join(sorted(r['registry_name'])) for r in reg_names_map.iter_rows(named=True)}
    
    all_cands = (
        counts.group_by('registry_id')
        .agg([
            pl.col('canonical_player_id').unique().alias('cand_ids'),
            pl.col('len').sum().alias('ev_sum')
        ])
    )
    cand_lookup = {r['registry_id']: r['cand_ids'] for r in all_cands.iter_rows(named=True)}
    
    leading = joined.group_by('registry_id').first()
    for r in leading.iter_rows(named=True):
        rid = r['registry_id']
        lead_id = int(r['canonical_player_id'])
        share = float(r['share'])
        tot = int(r['total_core_len'])
        cands = cand_lookup.get(rid, [lead_id])
        rnames = names_lookup.get(rid, '')
        
        if share >= 0.95:
            status = 'resolved'
            canon_id = lead_id
        else:
            status = 'conflict_unresolved'
            canon_id = None
            role_conflicts.append({'registry_id': rid, 'leading_share': share, 'total_evidence': tot})
            
        crosswalk_rows.append({
            'registry_id': rid,
            'canonical_player_id': canon_id,
            'candidate_player_ids': '|'.join(map(str, cands)),
            'n_candidate_player_ids': len(cands),
            'registry_names': rnames,
            'leading_candidate_share': share,
            'evidence_rows': tot,
            'status': status
        })

regmap_df = pl.DataFrame(crosswalk_rows)
reg_id_map = {}
for r in regmap_df.iter_rows(named=True):
    if r['status'] == 'resolved' and r['canonical_player_id'] is not None:
        reg_id_map[r['registry_id']] = f"PID_{int(r['canonical_player_id'])}"
    else:
        reg_id_map[r['registry_id']] = f"CS_{r['registry_id']}"

# 6. Inventory of Registry IDs from Modelling Archives: SA20, BBL, SMAT ONLY (NO T20 Blast!)
comp_meta = {'sat.zip': 'SA20', 'bbl.zip': 'BBL', 'sma.zip': 'SMAT'}
new_inventory = []
print("Inventorying modelling archives (SA20, BBL, SMAT)...", flush=True)
for fn, compname in comp_meta.items():
    z = zipfile.ZipFile(DATA / fn)
    for name in z.namelist():
        if not name.lower().endswith(('.yaml', '.yml')):
            continue
        raw = z.read(name)
        try:
            doc_h = load_header(raw)
            info = doc_h.get('info', {})
        except Exception:
            continue
        dt = match_date(info)
        if not dt or not ('2023-01-01' <= dt <= '2026-12-31') or info.get('gender') != 'male' or info.get('match_type') != 'T20':
            continue
        reg = info.get('registry', {}).get('people', {}) or {}
        for pname, rid in reg.items():
            new_inventory.append({'competition': compname, 'file': name, 'match_date': dt, 'registry_id': str(rid), 'registry_name': str(pname), 'venue': info.get('venue')})

inv_df = pl.DataFrame(new_inventory).unique()
invagg = (
    inv_df.group_by('registry_id')
    .agg([
        pl.col('registry_name').unique().sort().alias('names'),
        pl.col('competition').unique().sort().alias('comps'),
        pl.col('file').n_unique().alias('matches_seen')
    ])
    .with_columns([
        pl.col('names').list.join('|').alias('registry_names'),
        pl.col('comps').list.join('|').alias('competitions'),
    ])
    .drop(['names', 'comps'])
)

resolved_lookup = {r['registry_id']: r for r in regmap_df.iter_rows(named=True)}
inv_status = []
inv_pid = []
inv_canon = []
for r in invagg.iter_rows(named=True):
    rid = r['registry_id']
    info = resolved_lookup.get(rid)
    if info and info['status'] == 'resolved' and info['canonical_player_id'] is not None:
        inv_status.append('resolved')
        inv_pid.append(str(int(info['canonical_player_id'])))
        inv_canon.append(int(info['canonical_player_id']))
        reg_id_map.setdefault(rid, f"PID_{int(info['canonical_player_id'])}")
    else:
        inv_status.append('new_CS_identity')
        inv_pid.append(f"CS_{rid}")
        inv_canon.append(None)
        reg_id_map.setdefault(rid, f"CS_{rid}")

invagg = invagg.with_columns([
    pl.Series('identity_status', inv_status),
    pl.Series('player_id', inv_pid),
    pl.Series('canonical_player_id', inv_canon)
])

# Save Identity Crosswalk Outputs
pl.DataFrame(alignment_rows).write_csv(OUT_ID / 'match_alignment_results.csv')
pl.DataFrame(collision_rows).write_csv(OUT_ID / 'candidate_collision_audit.csv')
pl.DataFrame(unmatched_rows).write_csv(OUT_ID / 'unmatched_and_rejected_matches.csv')
ev_df.write_csv(OUT_ID / 'registry_role_evidence.csv')
regmap_df.write_csv(OUT_ID / 'registry_to_canonical_player_id_crosswalk.csv')
inv_df.write_csv(OUT_ID / 'new_competition_registry_inventory_long.csv')
invagg.write_csv(OUT_ID / 'new_competition_registry_inventory_by_id.csv')

au_rows = []
for (src, raw_a), vals in sorted(alias_usage.items()):
    for val in sorted(vals):
        au_rows.append({'source': src, 'raw_alias': raw_a, 'canonical_team_name': val, 'normalised_alias_key': norm_base(raw_a)})
pl.DataFrame(au_rows).write_csv(OUT_ID / 'aliases_actually_used.csv')
pl.DataFrame(team_lookup_issues).write_csv(OUT_ID / 'teams_lookup_issues.csv')

# 7. Ingest Original IPL / T20I Retained Delivery Base (805 matches)
print("Loading vetted original IPL/T20I deliveries from all_805_revision_oof_pressure_labels.csv.gz...", flush=True)
legacy_labels = pl.read_csv('inputs/all_805_revision_oof_pressure_labels.csv.gz')

# Join with raw ball_by_ball for extra metadata (batting_type, bowling_type, player names)
raw_cols = ['match_id', 'innings', 'overid', 'over_ball_num', 'striker_id', 'bowler_id',
            'striker_name', 'non_striker_name', 'bowler_name', 'batting_type', 'bowling_type',
            'batting_team_id', 'bowling_team_id', 'dismissal_type']
raw_extra = pl.read_csv(INPUT_PACK / 'ball_by_ball.csv', columns=raw_cols)

join_keys = ['match_id', 'innings', 'overid', 'over_ball_num', 'striker_id', 'bowler_id']
legacy_joined = legacy_labels.join(raw_extra, on=join_keys, how='left')

# Map team names and winner
retained_805_matches = legacy_matches.filter(
    pl.col('match_id').is_in(legacy_labels['match_id'].unique().to_list()) &
    pl.col('win_team_id').is_not_null()
)
match_lookup = {r['match_id']: r for r in retained_805_matches.iter_rows(named=True)}

# Standardize columns for legacy
legacy_comp = np.where(legacy_joined['competition_type'].is_in(['IPL', 'T20I']), legacy_joined['competition_type'],
                       np.where(legacy_joined['match_id'].map_elements(lambda m: match_lookup[m]['comp_name'], return_dtype=pl.String).str.contains(r'^IPL \d{4}$'), 'IPL', 'T20I'))

bat_teams = []
bowl_teams = []
win_labels = []
first_teams = []
second_teams = []

# Map team names
for r in legacy_joined.iter_rows(named=True):
    mid = r['match_id']
    meta = match_lookup[mid]
    cid = int(meta['comp_id'])
    bt = team_lookup.get((cid, int(r['batting_team_id'])), str(r['batting_team_id']))
    bwt = team_lookup.get((cid, int(r['bowling_team_id'])), str(r['bowling_team_id']))
    bat_teams.append(bt)
    bowl_teams.append(bwt)
    win_labels.append(int(int(r['batting_team_id']) == int(meta['win_team_id'])))

legacy_df = legacy_joined.with_columns([
    pl.Series('competition', legacy_comp),
    pl.Series('competition_type', legacy_comp),
    pl.lit('Original CSV').alias('source'),
    pl.lit('legacy').alias('source_type'),
    pl.col('match_id').cast(pl.String).alias('source_match_id'),
    ('LEGACY::' + pl.Series(legacy_comp) + '::' + pl.col('match_id').cast(pl.String)).alias('match_group_id'),
    ('LEGACY::' + pl.Series(legacy_comp) + '::' + pl.col('match_id').cast(pl.String)).alias('match_id_global'),
    ('LEGACY_VENUE_ID::' + pl.col('venue_id').fill_null('UNK').cast(pl.String)).alias('venue'),
    pl.col('venue_id').cast(pl.String).alias('venue_diag'),
    pl.Series('batting_team_name', bat_teams),
    pl.Series('bowling_team_name', bowl_teams),
    pl.Series('win_label', win_labels),
    pl.col('striker_id').map_elements(lambda x: f"PID_{int(x)}" if x is not None else None, return_dtype=pl.String).alias('striker_player_id'),
    pl.col('bowler_id').map_elements(lambda x: f"PID_{int(x)}" if x is not None else None, return_dtype=pl.String).alias('bowler_player_id'),
    pl.col('dismissed_player_id').map_elements(lambda x: f"PID_{int(x)}" if x is not None else None, return_dtype=pl.String).alias('dismissed_player_id_model'),
    pl.col('striker_name').fill_null('').alias('striker_name_model'),
    pl.col('bowler_name').fill_null('').alias('bowler_name_model'),
    pl.col('overid').cast(pl.Float64).alias('over'),
    pl.col('legal_over_before').cast(pl.Float64).alias('legal_over_before'),
    pl.col('balls_left_before').cast(pl.Float64).alias('balls_left'),
    pl.col('balls_left_before').cast(pl.Float64).alias('balls_left_before'),
    pl.col('wickets_before').cast(pl.Float64).alias('wickets_lost'),
    pl.col('wickets_before').cast(pl.Float64).alias('wickets_before'),
    (10.0 - pl.col('wickets_before')).clip(0.0, 10.0).alias('wickets_in_hand'),
    pl.col('score_before').cast(pl.Float64).alias('score_before'),
    pl.col('target').cast(pl.Float64).alias('target'),
    pl.when(pl.col('innings') == 2)
      .then((pl.col('target') - pl.col('score_before')).clip(0.0, None) * 6.0 / pl.col('balls_left_before').clip(1.0, None))
      .otherwise(0.0)
      .alias('required_rate'),
    pl.when(pl.col('innings') == 2)
      .then((pl.col('target') - pl.col('score_before')).clip(0.0, None) * 6.0 / pl.col('balls_left_before').clip(1.0, None))
      .otherwise(0.0)
      .alias('required_run_rate'),
    pl.col('batter_runs').cast(pl.Float64).alias('batter_runs'),
    pl.col('team_runs_event').cast(pl.Float64).alias('team_runs_event'),
    pl.col('bowler_charged_runs').cast(pl.Float64).alias('bowler_charged_runs'),
    pl.col('batter_faced').cast(pl.Boolean).alias('batter_faced'),
    pl.col('legal_ball').cast(pl.Boolean).alias('legal_ball'),
    pl.col('is_wicket_bool').cast(pl.Boolean).alias('is_wicket_bool'),
    pl.col('is_wicket_bool').cast(pl.Int64).alias('batter_dismissal_target'),
    pl.col('dismissal_type').fill_null('').str.to_lowercase().str.replace_all('_', ' ').str.replace_all('-', ' ').alias('dismissal_type_norm'),
    pl.lit(True).alias('model_observation'),
    pl.col('batting_type').fill_null('UNK').str.to_uppercase()
      .map_elements(lambda x: 'RHB' if 'RIGHT' in x else ('LHB' if 'LEFT' in x else 'UNK'), return_dtype=pl.String)
      .alias('batter_hand'),
    pl.col('pace_or_spin').fill_null('UNK').str.to_titlecase().alias('pace_or_spin'),
    pl.lit(None).alias('non_striker_player_id')
])

# Bowler credited wickets
bowler_credit = pl.col('dismissal_type_norm').is_in(['bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'stump', 'hit wicket']).cast(pl.Int64)
legacy_df = legacy_df.with_columns(bowler_credit.alias('bowler_credited_wicket_target'))

# Final innings totals
legacy_totals = (
    legacy_df.group_by(['match_id', 'innings'])
    .agg(pl.col('team_runs_event').sum().alias('final_innings_total'))
)
legacy_first_totals = legacy_totals.filter(pl.col('innings') == 1).select(['match_id', pl.col('final_innings_total').alias('first_innings_total')])
legacy_df = legacy_df.join(legacy_totals, on=['match_id', 'innings'], how='left')
legacy_df = legacy_df.join(legacy_first_totals, on='match_id', how='left')

legacy_df = legacy_df.with_columns([
    (pl.col('final_innings_total') - pl.col('score_before')).clip(0.0, None).alias('remaining_innings_runs_target'),
    pl.when(pl.col('innings') == 1)
      .then(pl.col('first_innings_total') - pl.col('score_before'))
      .otherwise(np.nan)
      .alias('remaining_runs_first_innings')
])

# 8. Ingest Cricsheet Modelling Matches (SA20, BBL, SMAT)
cricsheet_rows = []
cricsheet_matches_meta = []
cricsheet_qa = []

print("Parsing Cricsheet deliveries from SA20, BBL, SMAT...", flush=True)
for comp, fn in [('SA20', 'sat.zip'), ('Big Bash League', 'bbl.zip'), ('Syed Mushtaq Ali Trophy', 'sma.zip')]:
    z = zipfile.ZipFile(DATA / fn)
    for name in z.namelist():
        if not name.lower().endswith(('.yaml', '.yml')):
            continue
        raw = z.read(name)
        try:
            doc_h = load_header(raw)
            info = doc_h.get('info', {})
        except Exception as e:
            cricsheet_qa.append({'competition': comp, 'file': name, 'status': 'yaml_error', 'reason': str(e)})
            continue
        dt = match_date(info)
        if not dt or not ('2023-01-01' <= dt <= '2026-12-31') or info.get('gender') != 'male' or info.get('match_type') != 'T20':
            continue
        
        outcome = info.get('outcome', {}) or {}
        winner = outcome.get('winner')
        method = outcome.get('method')
        result = outcome.get('result')
        mid = Path(name).stem
        
        if method in ['D/L', 'VJD'] or 'revised_target' in info:
            cricsheet_qa.append({'competition': comp, 'file': name, 'match_id': mid, 'date': dt, 'status': 'excluded', 'reason': 'revised_target'})
            continue
        if not winner or result in ['no result', 'tie', 'tied']:
            cricsheet_qa.append({'competition': comp, 'file': name, 'match_id': mid, 'date': dt, 'status': 'excluded', 'reason': 'no_result_or_tied'})
            continue
            
        try:
            doc = yaml.load(raw, Loader=yaml.CSafeLoader)
            innlist = doc.get('innings', []) or []
            innings_objs = []
            if isinstance(innlist, dict):
                innings_objs = [v for _, v in list(innlist.items())[:2]]
            else:
                for part in innlist[:2]:
                    if isinstance(part, dict) and part:
                        innings_objs.append(next(iter(part.values())))
            if len(innings_objs) < 2:
                cricsheet_qa.append({'competition': comp, 'file': name, 'match_id': mid, 'date': dt, 'status': 'excluded', 'reason': 'fewer_than_two_innings'})
                continue
            teams_info = info.get('teams', []) or []
            if len(teams_info) != 2:
                continue
        except Exception as e:
            cricsheet_qa.append({'competition': comp, 'file': name, 'match_id': mid, 'date': dt, 'status': 'yaml_parse_error', 'reason': str(e)})
            continue
            
        people = info.get('registry', {}).get('people', {}) or {}
        overs = int(info.get('overs', 20) or 20)
        bpo = int(info.get('balls_per_over', 6) or 6)
        limit_balls = float(overs * bpo)
        venue = 'CRICSHEET_VENUE::' + str(info.get('venue') or 'UNK')
        
        team_first = str(innings_objs[0].get('team', ''))
        team_second = str(innings_objs[1].get('team', ''))
        if not team_first or not team_second or team_first == team_second:
            continue
            
        matchkey = f"CRIC::{comp}::{mid}"
        innings_final = []
        
        for ix, innobj in enumerate(innings_objs, 1):
            team = str(innobj.get('team', ''))
            delivs = innobj.get('deliveries', []) or []
            score = 0.0
            wickets = 0
            legal_elapsed = 0
            rows = []
            
            for seq_i, d0 in enumerate(delivs, 1):
                if not isinstance(d0, dict) or not d0:
                    continue
                _, ev = next(iter(d0.items()))
                runs_ev = ev.get('runs', {}) or {}
                extras_ev = ev.get('extras', {}) or {}
                
                batter = str(ev.get('batsman', ev.get('batter', ''))) or None
                nonstr = str(ev.get('non_striker', '')) or None
                bowler = str(ev.get('bowler', '')) or None
                
                b_runs = float(runs_ev.get('batsman', 0) or 0)
                tot_runs = float(runs_ev.get('total', b_runs) or 0)
                
                extra_norm = {str(k).lower().replace('_', ''): float(v or 0) for k, v in extras_ev.items()}
                is_wide = extra_norm.get('wides', 0) > 0
                is_noball = (extra_norm.get('noballs', 0) > 0 or extra_norm.get('noball', 0) > 0)
                is_penalty = extra_norm.get('penalty', 0) > 0
                
                wlist = ev.get('wickets', ev.get('wicket', [])) or []
                if isinstance(wlist, dict):
                    wlist = [wlist]
                    
                if is_penalty and not batter and not bowler:
                    score += tot_runs
                    continue
                    
                is_legal = (not is_wide and not is_noball)
                is_faced = (not is_wide)
                bowler_charge = b_runs + extra_norm.get('wides', 0) + extra_norm.get('noballs', 0) + extra_norm.get('noball', 0)
                
                sid = reg_id_map.get(str(people.get(batter)), f"CS_{people.get(batter)}") if batter and people.get(batter) else None
                nid = reg_id_map.get(str(people.get(nonstr)), f"CS_{people.get(nonstr)}") if nonstr and people.get(nonstr) else None
                bid = reg_id_map.get(str(people.get(bowler)), f"CS_{people.get(bowler)}") if bowler and people.get(bowler) else None
                
                dis_id = None
                is_bowler_wicket = 0
                dtype = ''
                if wlist:
                    pout = wlist[0].get('player_out')
                    dtype = str(wlist[0].get('kind', '')).strip().lower().replace('_', ' ').replace('-', ' ')
                    if pout and people.get(pout):
                        dis_id = reg_id_map.get(str(people.get(pout)), f"CS_{people.get(pout)}")
                    if dtype in ['bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'stump', 'hit wicket']:
                        is_bowler_wicket = 1
                
                over_before = math.floor(legal_elapsed / 6)
                balls_left_before = max(limit_balls - legal_elapsed, 0.0)
                
                row = {
                    'match_id': matchkey,
                    'match_group_id': matchkey,
                    'source_match_id': mid,
                    'source': 'Cricsheet',
                    'source_type': 'cricsheet',
                    'competition': comp,
                    'competition_type': comp,
                    'match_date': dt,
                    'innings': ix,
                    'innings_ball_num': seq_i,
                    'over': float(over_before),
                    'legal_over_before': float(over_before),
                    'overid': int(over_before + 1),
                    'over_ball_num': seq_i,
                    'innings_limit_balls': float(limit_balls),
                    'legal_elapsed_before': float(legal_elapsed),
                    'balls_left': float(balls_left_before),
                    'balls_left_before': float(balls_left_before),
                    'wickets_lost': float(wickets),
                    'wickets_before': float(wickets),
                    'wickets_in_hand': float(max(10 - wickets, 0)),
                    'score_before': float(score),
                    'win_label': int(norm_base(team) == norm_base(winner)),
                    'batting_team_name': team,
                    'bowling_team_name': team_second if ix == 1 else team_first,
                    'first_innings_team': team_first,
                    'second_innings_team': team_second,
                    'venue': venue,
                    'venue_diag': str(info.get('venue') or ''),
                    'striker_player_id': sid,
                    'non_striker_player_id': nid,
                    'bowler_player_id': bid,
                    'dismissed_player_id_model': dis_id,
                    'striker_name_model': batter or '',
                    'bowler_name_model': bowler or '',
                    'batter_hand': 'UNK',
                    'pace_or_spin': 'UNK',
                    'batter_runs': float(b_runs),
                    'team_runs_event': float(tot_runs),
                    'bowler_charged_runs': float(bowler_charge),
                    'batter_faced': bool(is_faced),
                    'legal_ball': bool(is_legal),
                    'is_wicket_bool': bool(wlist),
                    'batter_dismissal_target': int(bool(wlist)),
                    'bowler_credited_wicket_target': int(is_bowler_wicket),
                    'dismissal_type': dtype,
                    'dismissal_type_norm': dtype,
                    'model_observation': True
                }
                rows.append(row)
                score += tot_runs
                if wlist:
                    wickets += len(wlist)
                if is_legal:
                    legal_elapsed += 1
                    
            innings_final.append({'innings': ix, 'team': team, 'final_total': score, 'rows': rows})
            
        first_total = float(innings_final[0]['final_total'])
        for ir in innings_final:
            for row in ir['rows']:
                row['first_innings_total'] = float(first_total)
                row['final_innings_total'] = float(ir['final_total'])
                if row['innings'] == 1:
                    row['target'] = np.nan
                    row['required_rate'] = 0.0
                    row['required_run_rate'] = 0.0
                    row['remaining_runs_first_innings'] = float(first_total - row['score_before'])
                else:
                    row['target'] = float(first_total + 1.0)
                    row['required_rate'] = float(max(first_total + 1.0 - row['score_before'], 0.0) * 6.0 / max(row['balls_left'], 1.0))
                    row['required_run_rate'] = row['required_rate']
                    row['remaining_runs_first_innings'] = np.nan
                row['remaining_innings_runs_target'] = float(max(float(ir['final_total']) - row['score_before'], 0.0))
                cricsheet_rows.append(row)
                
        cricsheet_matches_meta.append({
            'match_id': matchkey, 'match_group_id': matchkey, 'source_match_id': mid, 'competition': comp,
            'match_date': dt, 'first_innings_team': team_first, 'second_innings_team': team_second,
            'winner': winner, 'venue': venue
        })
        cricsheet_qa.append({'competition': comp, 'file': name, 'match_id': mid, 'date': dt, 'status': 'included', 'reason': ''})

print(f"Cricsheet parsing complete. Ingested {len(cricsheet_matches_meta)} matches, {len(cricsheet_rows)} deliveries.", flush=True)

# Build Cricsheet DataFrame using explicit schema
cric_schema = {
    'match_id': pl.String, 'match_group_id': pl.String, 'source_match_id': pl.String, 'source': pl.String,
    'source_type': pl.String, 'competition': pl.String, 'competition_type': pl.String, 'match_date': pl.String,
    'innings': pl.Int64, 'innings_ball_num': pl.Int64, 'over': pl.Float64, 'legal_over_before': pl.Float64,
    'overid': pl.Int64, 'over_ball_num': pl.Int64, 'innings_limit_balls': pl.Float64, 'legal_elapsed_before': pl.Float64,
    'balls_left': pl.Float64, 'balls_left_before': pl.Float64, 'wickets_lost': pl.Float64, 'wickets_before': pl.Float64,
    'wickets_in_hand': pl.Float64, 'score_before': pl.Float64, 'win_label': pl.Int64, 'batting_team_name': pl.String,
    'bowling_team_name': pl.String, 'first_innings_team': pl.String, 'second_innings_team': pl.String,
    'venue': pl.String, 'venue_diag': pl.String, 'striker_player_id': pl.String, 'non_striker_player_id': pl.String,
    'bowler_player_id': pl.String, 'dismissed_player_id_model': pl.String, 'striker_name_model': pl.String,
    'bowler_name_model': pl.String, 'batter_hand': pl.String, 'pace_or_spin': pl.String, 'batter_runs': pl.Float64,
    'team_runs_event': pl.Float64, 'bowler_charged_runs': pl.Float64, 'batter_faced': pl.Boolean,
    'legal_ball': pl.Boolean, 'is_wicket_bool': pl.Boolean, 'batter_dismissal_target': pl.Int64,
    'bowler_credited_wicket_target': pl.Int64, 'dismissal_type': pl.String, 'dismissal_type_norm': pl.String,
    'model_observation': pl.Boolean, 'first_innings_total': pl.Float64, 'final_innings_total': pl.Float64,
    'target': pl.Float64, 'required_rate': pl.Float64, 'required_run_rate': pl.Float64,
    'remaining_runs_first_innings': pl.Float64, 'remaining_innings_runs_target': pl.Float64
}

cric_df = pl.DataFrame(cricsheet_rows, schema=cric_schema)

# 9. Pre-Match Elo Estimation Across All 5 Competitions
# Unique match-sides for Elo
legacy_match_sides = (
    legacy_df.select([
        'match_group_id', 'competition', 'match_date', 'batting_team_name', 'bowling_team_name', 'innings'
    ])
    .filter(pl.col('innings') == 1)
    .rename({'match_group_id': 'match_id', 'batting_team_name': 'first_innings_team', 'bowling_team_name': 'second_innings_team'})
    .drop('innings')
    .unique()
)

# Join winner
legacy_winners = legacy_matches.select([
    ('LEGACY::' + pl.Series(np.where(legacy_matches['comp_name'].str.contains(r'^IPL \d{4}$'), 'IPL', 'T20I')) + '::' + pl.col('match_id').cast(pl.String)).alias('match_id'),
    pl.col('win_team_id').map_elements(lambda w: team_lookup.get((int(legacy_matches['comp_id'][0]), int(w)), str(w)) if w is not None else '', return_dtype=pl.String).alias('winner')
])
legacy_match_sides = legacy_match_sides.join(legacy_winners, on='match_id', how='left')

cric_match_sides = pl.DataFrame(cricsheet_matches_meta).select(['match_id', 'competition', 'match_date', 'first_innings_team', 'second_innings_team', 'winner']).unique()

all_matches_meta = pl.concat([legacy_match_sides, cric_match_sides], how='vertical').sort(['competition', 'match_date', 'match_id'])

# Pre-match Elo ratings
elo_first_map = {}
elo_second_map = {}

for comp_name in sorted(all_matches_meta['competition'].unique().to_list()):
    c_matches = all_matches_meta.filter(pl.col('competition') == comp_name)
    ratings = collections.defaultdict(lambda: 1500.0)
    
    for date_str in sorted(c_matches['match_date'].unique().to_list()):
        day_matches = c_matches.filter(pl.col('match_date') == date_str)
        day_rows = list(day_matches.iter_rows(named=True))
        
        for mr in day_rows:
            mid = mr['match_id']
            t1 = mr['first_innings_team']
            t2 = mr['second_innings_team']
            elo_first_map[mid] = ratings[t1]
            elo_second_map[mid] = ratings[t2]
            
        updates = collections.defaultdict(float)
        counts = collections.defaultdict(int)
        for mr in day_rows:
            t1 = mr['first_innings_team']
            t2 = mr['second_innings_team']
            winner = mr['winner']
            ra, rb = ratings[t1], ratings[t2]
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            eb = 1.0 - ea
            sa = 1.0 if norm_base(winner) == norm_base(t1) else 0.0
            sb = 1.0 - sa
            updates[t1] += 24.0 * (sa - ea)
            updates[t2] += 24.0 * (sb - eb)
            counts[t1] += 1
            counts[t2] += 1
            
        for team, delta in updates.items():
            ratings[team] += (delta / max(counts[team], 1))

print("Pre-match Elo calculation completed.", flush=True)

# 10. Map Elo to legacy and cricsheet
legacy_first_elo = [elo_first_map.get(mid, 1500.0) for mid in legacy_df['match_group_id'].to_list()]
legacy_second_elo = [elo_second_map.get(mid, 1500.0) for mid in legacy_df['match_group_id'].to_list()]

legacy_opp_elo_bat = []
legacy_opp_elo_bowl = []
for i, r in enumerate(legacy_df.iter_rows(named=True)):
    ix = int(r['innings'])
    e1 = legacy_first_elo[i]
    e2 = legacy_second_elo[i]
    if ix == 1:
        legacy_opp_elo_bat.append(e2)
        legacy_opp_elo_bowl.append(e1)
    else:
        legacy_opp_elo_bat.append(e1)
        legacy_opp_elo_bowl.append(e2)

legacy_df = legacy_df.with_columns([
    pl.Series('opposition_elo_batter', legacy_opp_elo_bat),
    pl.Series('opposition_elo_bowler', legacy_opp_elo_bowl)
])

cric_first_elo = [elo_first_map.get(mid, 1500.0) for mid in cric_df['match_group_id'].to_list()]
cric_second_elo = [elo_second_map.get(mid, 1500.0) for mid in cric_df['match_group_id'].to_list()]

cric_opp_elo_bat = []
cric_opp_elo_bowl = []
for i, r in enumerate(cric_df.iter_rows(named=True)):
    ix = int(r['innings'])
    e1 = cric_first_elo[i]
    e2 = cric_second_elo[i]
    if ix == 1:
        cric_opp_elo_bat.append(e2)
        cric_opp_elo_bowl.append(e1)
    else:
        cric_opp_elo_bat.append(e1)
        cric_opp_elo_bowl.append(e2)

cric_df = cric_df.with_columns([
    pl.Series('opposition_elo_batter', cric_opp_elo_bat),
    pl.Series('opposition_elo_bowler', cric_opp_elo_bowl)
])

# Align common columns for combined dataset
COMMON_COLS = [
    'match_id', 'match_group_id', 'source_match_id', 'source', 'source_type',
    'competition', 'competition_type', 'match_date', 'innings', 'innings_ball_num',
    'over', 'legal_over_before', 'overid', 'over_ball_num', 'innings_limit_balls',
    'legal_elapsed_before', 'balls_left', 'balls_left_before', 'wickets_lost',
    'wickets_before', 'wickets_in_hand', 'score_before', 'target', 'required_rate',
    'required_run_rate', 'first_innings_total', 'final_innings_total',
    'remaining_innings_runs_target', 'remaining_runs_first_innings', 'win_label',
    'batting_team_name', 'bowling_team_name', 'venue', 'venue_diag',
    'striker_player_id', 'non_striker_player_id', 'bowler_player_id',
    'dismissed_player_id_model', 'striker_name_model', 'bowler_name_model',
    'batter_hand', 'pace_or_spin', 'batter_runs', 'team_runs_event',
    'bowler_charged_runs', 'batter_faced', 'legal_ball', 'is_wicket_bool',
    'batter_dismissal_target', 'bowler_credited_wicket_target', 'dismissal_type',
    'model_observation', 'opposition_elo_batter', 'opposition_elo_bowler'
]

leg_sel = legacy_df.select([pl.col(c).cast(cric_schema.get(c, legacy_df[c].dtype)) for c in COMMON_COLS])
cric_sel = cric_df.select([pl.col(c).cast(cric_schema.get(c, legacy_df[c].dtype)) for c in COMMON_COLS])

combined_df = pl.concat([leg_sel, cric_sel], how='vertical')

# Save Deliveries Parquet and GZ CSV
print(f"Writing combined delivery base: {len(combined_df):,} deliveries across {combined_df['match_group_id'].n_unique()} matches...", flush=True)
combined_df.write_parquet(OUT_MODEL / 'combined_delivery_base.parquet')
combined_df.write_csv(OUT_MODEL / 'combined_delivery_base.csv.gz', compression='gzip')

# Match level audit and source counts
src_counts = (
    combined_df.group_by('competition')
    .agg([
        pl.col('match_group_id').n_unique().alias('matches'),
        pl.len().alias('deliveries'),
        pl.col('striker_player_id').n_unique().alias('unique_batters'),
        pl.col('bowler_player_id').n_unique().alias('unique_bowlers')
    ])
    .sort('matches', descending=True)
)
src_counts.write_csv(OUT_MODEL / 'new_competition_source_counts.csv')
pl.DataFrame(cricsheet_qa).write_csv(OUT_MODEL / 'new_competition_match_ingestion_audit.csv')

# Build Identity Name Map
identity_names = {}
mapping_df = pl.read_csv(INPUT_PACK / 'competition_team_player_mapping.csv')
for r in mapping_df.filter(pl.col('player_id').is_not_null()).iter_rows(named=True):
    pid = f"PID_{int(r['player_id'])}"
    pname = str(r['player_name'] or '').strip()
    if pname and pid not in identity_names:
        identity_names[pid] = pname

for r in legacy_df.filter(pl.col('striker_player_id').is_not_null()).select(['striker_player_id', 'striker_name_model']).unique().iter_rows(named=True):
    pid = r['striker_player_id']
    if pid not in identity_names and r['striker_name_model']:
        identity_names[pid] = r['striker_name_model']

for r in legacy_df.filter(pl.col('bowler_player_id').is_not_null()).select(['bowler_player_id', 'bowler_name_model']).unique().iter_rows(named=True):
    pid = r['bowler_player_id']
    if pid not in identity_names and r['bowler_name_model']:
        identity_names[pid] = r['bowler_name_model']

for r in invagg.iter_rows(named=True):
    pid = r['player_id']
    pname = r['registry_names'].split('|')[0] if r['registry_names'] else ''
    if pid not in identity_names and pname:
        identity_names[pid] = pname

with open(OUT_MODEL / 'identity_name_map.json', 'w') as f:
    json.dump(identity_names, f, indent=2)

summary = {
    'config_sha256': actual_sha,
    'total_deliveries': len(combined_df),
    'total_matches': combined_df['match_group_id'].n_unique(),
    'legacy_matches': legacy_df['match_group_id'].n_unique(),
    'legacy_deliveries': len(legacy_df),
    'cricsheet_matches': cric_df['match_group_id'].n_unique(),
    'cricsheet_deliveries': len(cric_df),
    'competitions': src_counts.to_dicts()
}
with open(OUT_MODEL / 'combined_base_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Ingestion and alignment successfully complete!", flush=True)
print(json.dumps(summary, indent=2), flush=True)
