import pandas as pd
import re

def fix_labels(csv_path="data/queries.csv", output_path="data/queries_fixed.csv"):
    df = pd.read_csv(csv_path, on_bad_lines='skip', quotechar='"')
    defaults = {"num_qubits": 2, "max_time": 10, "num_points": 100, "epochs": 1000, "decoherence_rate": 0.0}
    
    for i, row in df.iterrows():
        query = str(row["query"]).lower()
        # Extract qubits
        qubit_match = re.search(r'(\d+)\s*(qubit|qubits)', query)
        if qubit_match:
            df.at[i, "num_qubits"] = min(max(int(qubit_match.group(1)), 2), 6)
        # Extract time
        time_match = re.search(r'(\d+)\s*(time|unit|units)', query)
        if time_match:
            df.at[i, "max_time"] = min(max(int(time_match.group(1)), 5), 15)
        # Extract decoherence
        deco_match = re.search(r'(\d+\.?\d*)\s*(deco|decoherence|noise)', query)
        if deco_match:
            df.at[i, "decoherence_rate"] = min(max(float(deco_match.group(1)), 0.0), 0.1)
        # Fill defaults for numerics
        for col in defaults:
            if pd.isna(df.at[i, col]):
                df.at[i, col] = defaults[col]
    
    df.dropna(subset=["query"], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Fixed {len(df)} rows. Sample:\n{df.head()}")

if __name__ == "__main__":
    fix_labels()