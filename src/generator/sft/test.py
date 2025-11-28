import pandas as pd

df = pd.read_parquet('datasets/gsm8k/test-00000-of-00001.parquet')

print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("=" * 60)

# 展示前3条数据的完整内容
for i in range(min(3, len(df))):
    print(f"\n{'='*60}")
    print(f"Row {i}")
    print("=" * 60)
    row = df.iloc[i]
    for col in df.columns:
        print(f"\n[{col}]:")
        print(row[col])
