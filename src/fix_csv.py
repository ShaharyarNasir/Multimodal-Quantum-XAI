import pandas as pd
df = pd.read_csv("data/queries.csv", on_bad_lines='skip', quotechar='"')
defaults = {"num_qubits": 2, "max_time": 10,
            "num_points": 100, "epochs": 1000, "decoherence_rate": 0.0}
for col in defaults:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(defaults[col])
df.dropna(subset=["query"], inplace=True)
df.to_csv("data/queries_fixed.csv", index=False)
