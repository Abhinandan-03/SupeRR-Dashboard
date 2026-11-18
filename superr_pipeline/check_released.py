import polars as pl
from pathlib import Path
import re

# Load mapping and squads
mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
ipl26 = mp.filter(pl.col('comp_name') == 'IPL 2026')
squads = pl.read_csv('inputs/squads.csv')

# Load playing eleven for broader name lookup
pe = pl.read_csv('inputs/data_pack/Data Pack/playing_eleven.csv')
pe_names = pe.select(['player_id', 'player_name']).unique()

print(f"IPL 2026 mapping entries: {len(ipl26)}")
print(f"Squads entries: {len(squads)}")

raw_released_text = """
MS Dhoni, Aman Khan, Dian Forrester, Kuldip Yadav, Macneil Noronha, Matthew Short, Matt Henry, Akash Madhwal, Prashant Veer, Spencer Johnson, Zak Foulkes | 
Trent Boult, Deepak Chahar, AM Ghazanfar, Atharva Ankolekar, Keshav Maharaj, Mohammed Izhar, Danish Malewar, Mayank Markande, Mayank Rawat, Mahipal Lomror, Robin Minz, Ruchit Ahir, Krish Bhagat | 
Mitchell Starc, Nitish Rana, Abishek Porel, Pathum Nissanka, Ben Duckett, Karun Nair, Sahil Parakh, Dushmantha Chameera, Rehan Ahmed, Ajay Mandal | 
Avesh Khan, Matthew Breetzke, Anrich Nortje, George Linde, Wanindu Hasaranga, Naman Tiwari, Arshin Kulkarni | 
Nehal Wadhera, Lockie Ferguson, Mitchell Owen, Pyla Avinash, Praveen Dubey, Vishal Nishad | 
Released by Rajasthan Royals: Shimron Hetmyer, Nandre Burger, Adam Milne, Kwena Maphaka, Tushar Deshpande, Kuldeep Sen, Yudhvir Singh, Vignesh Puthur, Ravi Singh | 
Travis Head, Kamindu Mendis, Jaydev Unadkat, Dishan Madushanka, Onkar Tarmale, Amit Kumar, Shivam Mavi, Gerald Coetzee, David Payne | 
Jacob Bethell, Richard Gleeson, Mangesh Yadav, Satvik Deswal, Vihaan Malhotra, Swapnil Singh | 
Ishant Sharma, Luke Wood, Kulwant Khejroliya, Shahrukh Khan, Connor Esterhuizen
"""

# Extract names
chunks = raw_released_text.replace('\n', ' ').split('|')
released_names = []
for chunk in chunks:
    chunk = chunk.replace('Released by Rajasthan Royals:', '')
    parts = [p.strip() for p in chunk.split(',') if p.strip()]
    released_names.extend(parts)

print(f"Total released names parsed: {len(released_names)}")

# Known aliases dictionary
aliases = {
    'Abishek Porel': 'Abhishek Porel',
    'AM Ghazanfar': 'Allah Ghazanfar',
    'Zak Foulkes': 'Zakary Foulkes',
    'Shahrukh Khan': 'M Shahrukh Khan',
    'Kuldip Yadav': 'Kuldip Yadav',
    'Kuldeep Yadav': 'Kuldeep Yadav'
}

# Build lookup dictionaries
# Name normalization helper
def norm(s):
    if not s: return ''
    return re.sub(r'[^a-z0-9]', '', s.lower())

ipl26_records = ipl26.to_dicts()
squads_records = squads.to_dicts()

matches = []
unmatched = []

for raw_name in released_names:
    target_name = aliases.get(raw_name, raw_name)
    n_target = norm(target_name)
    
    # 1. Look in ipl26 mapping
    found = None
    for r in ipl26_records:
        r_name = r['player_name']
        if norm(r_name) == n_target or norm(aliases.get(r_name, r_name)) == n_target:
            found = {'raw_name': raw_name, 'matched_name': r_name, 'player_id': r['player_id'], 'team_name': r['team_name'], 'source': 'ipl2026_mapping'}
            break
            
    # Check squads.csv if not found in mapping
    if not found:
        for s in squads_records:
            s_name = s['player']
            if norm(s_name) == n_target or norm(aliases.get(s_name, s_name)) == n_target:
                # Find player_id from pe or mp
                pid = None
                p_cand = pe.filter(pl.col('player_name').str.to_lowercase() == s_name.lower())
                if len(p_cand) > 0:
                    pid = p_cand['player_id'][0]
                else:
                    m_cand = mp.filter(pl.col('player_name').str.to_lowercase() == s_name.lower())
                    if len(m_cand) > 0:
                        pid = m_cand['player_id'][0]
                found = {'raw_name': raw_name, 'matched_name': s_name, 'player_id': pid, 'team_name': s['team_name'], 'source': 'squads_csv'}
                break
                
    if not found:
        # Check global mp or pe
        p_cand = pe.filter(pl.col('player_name').str.to_lowercase() == target_name.lower())
        if len(p_cand) > 0:
            pid = p_cand['player_id'][0]
            found = {'raw_name': raw_name, 'matched_name': p_cand['player_name'][0], 'player_id': pid, 'team_name': 'Known Player', 'source': 'playing_eleven'}
        else:
            m_cand = mp.filter(pl.col('player_name').str.to_lowercase() == target_name.lower())
            if len(m_cand) > 0:
                pid = m_cand['player_id'][0]
                found = {'raw_name': raw_name, 'matched_name': m_cand['player_name'][0], 'player_id': pid, 'team_name': m_cand['team_name'][0], 'source': 'competition_team_player_mapping'}

    if found:
        matches.append(found)
    else:
        unmatched.append(raw_name)

print(f"\nMatched count: {len(matches)}")
print(f"Unmatched count: {len(unmatched)}")
if unmatched:
    print("Unmatched:", unmatched)

for m in matches:
    print(f"{m['raw_name']} -> ID: {m['player_id']} | Matched: {m['matched_name']} | Team: {m['team_name']} ({m['source']})")
