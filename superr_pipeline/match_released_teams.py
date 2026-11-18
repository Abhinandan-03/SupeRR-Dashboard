import polars as pl
from pathlib import Path
import re

mp = pl.read_csv('inputs/data_pack/Data Pack/competition_team_player_mapping.csv')
ipl26 = mp.filter(pl.col('comp_name') == 'IPL 2026')
squads = pl.read_csv('inputs/squads.csv')

print(f"Unique teams in IPL 2026: {ipl26['team_name'].unique().to_list()}")
print(f"Unique teams in squads: {squads['team_name'].unique().to_list()}")

# Group released text by chunks (each chunk is an IPL team!)
raw_released_chunks = [
    # Chunk 1: CSK
    ("Chennai Super Kings", ["MS Dhoni", "Aman Khan", "Dian Forrester", "Kuldip Yadav", "Macneil Noronha", "Matthew Short", "Matt Henry", "Akash Madhwal", "Prashant Veer", "Spencer Johnson", "Zak Foulkes"]),
    # Chunk 2: MI
    ("Mumbai Indians", ["Trent Boult", "Deepak Chahar", "AM Ghazanfar", "Atharva Ankolekar", "Keshav Maharaj", "Mohammed Izhar", "Danish Malewar", "Mayank Markande", "Mayank Rawat", "Mahipal Lomror", "Robin Minz", "Ruchit Ahir", "Krish Bhagat"]),
    # Chunk 3: DC / KKR
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
    # Chunk 9: GT / KKR
    ("Gujarat Titans / Other", ["Ishant Sharma", "Luke Wood", "Kulwant Khejroliya", "Shahrukh Khan", "Connor Esterhuizen"])
]

aliases = {
    'ms dhoni': 'mahendra singh dhoni',
    'abishek porel': 'abhishek porel',
    'am ghazanfar': 'allah ghazanfar',
    'zak foulkes': 'zakary foulkes',
    'shahrukh khan': 'm shahrukh khan',
    'gerald coetzee': 'gerald coetzee'
}

def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

all_ipl_players = ipl26.to_dicts()

print("\n--- MATCHING RELEASED PLAYERS ---")
all_matched_ids = []
all_matched_records = []
unmatched = []

for team_hint, players in raw_released_chunks:
    print(f"\nEvaluating chunk for: {team_hint}")
    for p in players:
        p_norm = norm(aliases.get(p.lower(), p))
        found = []
        for r in all_ipl_players:
            r_norm = norm(r['player_name'])
            if p_norm == r_norm:
                found.append(r)
            elif p_norm in r_norm or r_norm in p_norm:
                # Substring check with caution
                if len(p_norm) >= 5 and (p_norm in r_norm or r_norm in p_norm):
                    # Check careful cases
                    if 'kuldip' in p_norm and 'kuldeep' in r_norm:
                        continue
                    if 'kuldeep' in p_norm and 'kuldip' in r_norm:
                        continue
                    if 'izhar' in p_norm and 'shami' in r_norm:
                        continue
                    found.append(r)
                    
        # Check Gerald Coetzee who was added separately to exclusion set
        if not found and norm(p) == norm('Gerald Coetzee'):
            found.append({'player_id': 6628, 'player_name': 'Gerald Coetzee', 'team_name': 'Sunrisers Hyderabad'})
            
        # Check squads.csv
        if not found:
            for s in squads.to_dicts():
                s_norm = norm(aliases.get(s['player'].lower(), s['player']))
                if p_norm == s_norm:
                    # Look up player_id across entire mapping
                    cand = mp.filter(pl.col('player_name').str.to_lowercase() == s['player'].lower())
                    pid = cand['player_id'][0] if len(cand) > 0 else None
                    found.append({'player_id': pid, 'player_name': s['player'], 'team_name': s['team_name']})
                    break

        if found:
            # Pick best match matching team hint if multiple
            best = found[0]
            for f in found:
                if norm(team_hint.split()[0]) in norm(f['team_name']):
                    best = f
                    break
            print(f"  [MATCH] {p} -> ID: {best['player_id']} | Name: {best['player_name']} | Team: {best['team_name']}")
            all_matched_records.append({'released_name': p, 'player_id': best['player_id'], 'player_name': best['player_name'], 'team_name': best['team_name']})
            if best['player_id'] is not None:
                all_matched_ids.append(best['player_id'])
        else:
            print(f"  [UNMATCHED] {p}")
            unmatched.append(p)

print(f"\nTotal parsed: {sum(len(p[1]) for p in raw_released_chunks)}")
print(f"Total matched: {len(all_matched_records)}")
print(f"Total matched IDs: {len(all_matched_ids)}")
print(f"Unmatched names: {unmatched}")
