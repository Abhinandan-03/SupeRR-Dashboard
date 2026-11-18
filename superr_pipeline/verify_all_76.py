import polars as pl
from pathlib import Path
import re

mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
ipl26 = mp.filter(pl.col('comp_name') == 'IPL 2026')
squads = pl.read_csv('inputs/squads.csv')

raw_released_chunks = [
    # Chunk 1: CSK
    ("Chennai Super Kings", ["MS Dhoni", "Aman Khan", "Dian Forrester", "Kuldip Yadav", "Macneil Noronha", "Matthew Short", "Matt Henry", "Akash Madhwal", "Prashant Veer", "Spencer Johnson", "Zak Foulkes"]),
    # Chunk 2: MI
    ("Mumbai Indians", ["Trent Boult", "Deepak Chahar", "AM Ghazanfar", "Atharva Ankolekar", "Keshav Maharaj", "Mohammed Izhar", "Danish Malewar", "Mayank Markande", "Mayank Rawat", "Mahipal Lomror", "Robin Minz", "Ruchit Ahir", "Krish Bhagat"]),
    # Chunk 3: DC
    ("Delhi Capitals", ["Mitchell Starc", "Nitish Rana", "Abishek Porel", "Pathum Nissanka", "Ben Duckett", "Karun Nair", "Sahil Parakh", "Dushmantha Chameera", "Rehan Ahmed", "Ajay Mandal"]),
    # Chunk 4: LSG
    ("Lucknow Super Giants", ["Avesh Khan", "Matthew Breetzke", "Anrich Nortje", "George Linde", "Wanindu Hasaranga", "Naman Tiwari", "Arshin Kulkarni"]),
    # Chunk 5: PBKS
    ("Punjab Kings", ["Nehal Wadhera", "Lockie Ferguson", "Mitchell Owen", "Pyla Avinash", "Praveen Dubey", "Vishal Nishad"]),
    # Chunk 6: RR
    ("Rajasthan Royals", ["Shimron Hetmyer", "Nandre Burger", "Adam Milne", "Kwena Maphaka", "Tushar Deshpande", "Kuldeep Sen", "Yudhvir Singh", "Vignesh Puthur", "Ravi Singh"]),
    # Chunk 7: SRH
    ("Sunrisers Hyderabad", ["Travis Head", "Kamindu Mendis", "Jaydev Unadkat", "Dishan Madushanka", "Onkar Tarmale", "Amit Kumar", "Shivam Mavi", "Gerald Coetzee", "David Payne"]),
    # Chunk 8: RCB
    ("Royal Challengers Bengaluru", ["Jacob Bethell", "Richard Gleeson", "Mangesh Yadav", "Satvik Deswal", "Vihaan Malhotra", "Swapnil Singh"]),
    # Chunk 9: GT / Other
    ("Gujarat Titans / Other", ["Ishant Sharma", "Luke Wood", "Kulwant Khejroliya", "Shahrukh Khan", "Connor Esterhuizen"])
]

aliases = {
    'ms dhoni': 'mahendra singh dhoni',
    'abishek porel': 'abishek porel',
    'abhishek porel': 'abishek porel',
    'am ghazanfar': 'allah ghazanfar',
    'zak foulkes': 'zakary foulkes',
    'shahrukh khan': 'm shahrukh khan',
    'mohammed izhar': 'mohd izhar',
    'onkar tarmale': 'omkar tarmale',
    'dishan madushanka': 'dilshan madushanka'
}

def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

all_ipl_players = ipl26.to_dicts()

print(f"Total IPL 2026 rows in mapping: {len(all_ipl_players)}")

matched = []
unmatched = []

for team_hint, players in raw_released_chunks:
    for p in players:
        target = aliases.get(p.lower(), p)
        n_target = norm(target)
        found = None
        
        # 1. Match directly against IPL 2026 mapping
        for r in all_ipl_players:
            if norm(r['player_name']) == n_target:
                found = {'raw_name': p, 'matched_name': r['player_name'], 'player_id': r['player_id'], 'team_name': r['team_name'], 'method': 'ipl2026_direct_name'}
                break
                
        # 2. Check Gerald Coetzee
        if not found and norm(p) == norm('Gerald Coetzee'):
            found = {'raw_name': p, 'matched_name': 'Gerald Coetzee', 'player_id': 6628, 'team_name': 'Sunrisers Hyderabad', 'method': 'known_coetzee_union'}
            
        # 3. Check squads.csv
        if not found:
            for s in squads.to_dicts():
                s_name = aliases.get(s['player'].lower(), s['player'])
                if norm(s_name) == n_target:
                    # Lookup player_id from mp
                    cand = mp.filter(pl.col('player_name').str.to_lowercase() == s['player'].lower())
                    pid = cand['player_id'][0] if len(cand) > 0 else None
                    found = {'raw_name': p, 'matched_name': s['player'], 'player_id': pid, 'team_name': s['team_name'], 'method': 'squads_csv'}
                    break
                    
        # 4. Check global mapping
        if not found:
            cand = mp.filter(pl.col('player_name').str.to_lowercase() == target.lower())
            if len(cand) > 0:
                found = {'raw_name': p, 'matched_name': cand['player_name'][0], 'player_id': cand['player_id'][0], 'team_name': cand['team_name'][0], 'method': 'global_mapping'}
                
        if found:
            matched.append(found)
        else:
            unmatched.append(p)

print(f"\nTotal names in released brief: {sum(len(p[1]) for p in raw_released_chunks)}")
print(f"Total matched: {len(matched)}")
print(f"Total unmatched: {len(unmatched)}")
if unmatched:
    print("Unmatched list:", unmatched)

for m in matched:
    print(f"{m['raw_name']:<20} -> ID: {str(m['player_id']):<6} | {m['matched_name']:<25} | {m['team_name']:<25} ({m['method']})")
