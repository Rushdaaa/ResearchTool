import pandas as pd

# Load the converted CSV
df = pd.read_csv('datasets/promise/Promise+.csv')

print("=== Promise+ Dataset Analysis ===")
print(f"Total requirements: {len(df)}")
print(f"Total projects: {df['ProjectID'].nunique()}")
print(f"Project ID range: {df['ProjectID'].min()} - {df['ProjectID'].max()}")

print("\n=== Project ID Distribution ===")
project_counts = df['ProjectID'].value_counts().sort_index()
print(project_counts)

print("\n=== Classification Distribution ===")
class_counts = df['_class_'].value_counts().sort_index()
print(class_counts)

print("\n=== Sample Requirements by Class ===")
for class_type in sorted(df['_class_'].unique()):
    sample = df[df['_class_'] == class_type].iloc[0]
    print(f"\n{class_type}: {sample['RequirementText'][:80]}...")
