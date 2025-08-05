import pandas as pd

# Path to your CSV file
file_path = 'D:\\PROJECTS\\air-quality-predictor-\\data\\data.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Debug: Print initial row count
print("Original row count:", len(df))

# Randomly select 500 rows (with replacement)
duplicated_rows = df.sample(n=5000, replace=True, random_state=42)

# Append the duplicated rows
df_augmented = pd.concat([df, duplicated_rows], ignore_index=True)

# Debug: Print new row count
print("New row count after augmentation:", len(df_augmented))

# Save back to the same file path (overwrite)
df_augmented.to_csv(file_path, index=False)

print("✅ 500 rows duplicated and saved back to data.csv")
