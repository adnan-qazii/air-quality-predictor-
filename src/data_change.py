import pandas as pd

# Path to your CSV file
file_path = 'D:\\PROJECTS\\air-quality-predictor-\\data\\data.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Randomly select 500 rows (with replacement)
duplicated_rows = df.sample(n=500, replace=True, random_state=42)

# Append the duplicated rows
df_augmented = pd.concat([df, duplicated_rows], ignore_index=True)

# Save back to the same file path (overwrite)
df_augmented.to_csv(file_path, index=False)

print("✅ 500 rows duplicated and saved back to data.csv")
