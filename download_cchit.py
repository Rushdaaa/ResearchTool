from datasets import load_dataset
import pandas as pd

# "artifacts" is a CONFIG for this dataset, not a split
ds = load_dataset("thearod5/CCHIT", "artifacts")  # config name

# Convert to pandas + save
df = ds["artifacts"].to_pandas()  # most configs use a single split named "train"
out_path = "CCHIT_artifacts.csv"
df.to_csv(out_path, index=False)

print(f"Saved {out_path} with {len(df):,} rows.")
print("Columns:", list(df.columns))
print("Sample rows:\n", df.head(5).to_string(index=False))
