"""Check if IPC section numbers match their descriptions in Neo4j"""
from config.neo4j_config import Neo4jConnection
import pandas as pd

# Connect to Neo4j
conn = Neo4jConnection()
conn.connect()

# Get all IPC sections from Neo4j
query = """
MATCH (s:IPCSection) 
RETURN s.section_number as section, s.offense as offense, s.punishment as punishment
ORDER BY toInteger(s.section_number)
LIMIT 50
"""
neo4j_data = conn.execute_query(query)

# Load CSV data
df = pd.read_csv('FIR_DATASET.csv')

print("="*80)
print("CHECKING IPC SECTION MISMATCHES")
print("="*80)

mismatches = []

for record in neo4j_data[:20]:  # Check first 20
    section_num = record['section']
    neo4j_offense = record['offense']
    
    # Extract numeric part if format is "Section_X"
    if section_num.startswith('Section_'):
        section_num_clean = section_num.replace('Section_', '')
    else:
        section_num_clean = section_num
    
    # Find corresponding row in CSV
    # CSV URL format: https://lawrato.com/indian-kanoon/ipc/section-{number}
    csv_row = df[df['URL'].str.contains(f'section-{section_num_clean}', case=False, na=False)]
    
    if not csv_row.empty:
        csv_offense = csv_row.iloc[0]['Offense']
        if pd.isna(csv_offense):
            csv_offense = "N/A"
        else:
            csv_offense = str(csv_offense)
        
        print(f"\n--- Section {section_num} ---")
        print(f"Neo4j:  {neo4j_offense[:100]}")
        print(f"CSV:    {csv_offense[:100]}")
        
        if neo4j_offense.strip() != csv_offense.strip():
            print("❌ MISMATCH!")
            mismatches.append(section_num)
        else:
            print("✅ Match")
    else:
        print(f"\n⚠️ Section {section_num} not found in CSV")

print(f"\n{'='*80}")
print(f"Total mismatches found: {len(mismatches)}")
if mismatches:
    print(f"Mismatched sections: {', '.join(mismatches)}")
print(f"{'='*80}")

conn.close()
