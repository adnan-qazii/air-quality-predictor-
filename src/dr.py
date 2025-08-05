#remove dubliucate rows from the dataset and replace the original file
import pandas as pd
# Path to your CSV file
file_path = 'D:\\PROJECTS\\air-quality-predictor-\\data\\data.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Remove duplicate rows
df = df.drop_duplicates()

# Save the cleaned dataset back to the original CSV file
df.to_csv(file_path, index=False)
print("✅ Duplicate rows removed and saved back to data.csv")