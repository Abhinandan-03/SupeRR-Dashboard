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

matched = []

for team_hint, players in raw_released_chunks:
    for p in players:
        target = aliases.get(p.lower(), p)
        n_target = norm(target)
        found = None
        
        # 1. Match directly against IPL 2026 mapping
        for r in all_ipl_players:
            if norm(r['player_name']) == n_target:
                found = {
                    'released_name': p,
                    'matched_name': r['player_name'],
                    'player_id': int(r['player_id']),
                    'formatted_id': f"PID_{int(r['player_id'])}",
                    'roster_team': r['team_name'],
                    'match_method': 'IPL 2026 direct roster match'
                }
                break
                
        # 2. Check Gerald Coetzee
        if not found and norm(p) == norm('Gerald Coetzee'):
            found = {
                'released_name': p,
                'matched_name': 'Gerald Coetzee',
                'player_id': 6628,
                'formatted_id': "PID_6628",
                'roster_team': 'Sunrisers Hyderabad',
                'match_method': 'Pre-declared Coetzee union'
            }
            
        if found:
            matched.append(found)
        else:
            print(f"FAILED TO MATCH: {p}")

print(f"Total matched: {len(matched)}")
df_matched = pl.DataFrame(matched)
out_csv = Path('pipeline_outputs/config/released_players_mapping.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
df_matched.write_csv(out_csv)
print(f"Saved {len(df_matched)} matched released players to {out_csv}")
