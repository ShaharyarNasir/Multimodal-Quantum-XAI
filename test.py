# from datasets import load_from_disk
# dataset = load_from_disk("data/processed_dataset")
# print(len(dataset))  # Should print ~600 (or fewer if NaNs dropped)

import pandas as pd
df = pd.read_csv("data/queries_fixed.csv")
print(df.head(3))  # Verify alignments