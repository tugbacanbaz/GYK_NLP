import os
import urllib.request
import pandas as pd

# Hedef klasör
base_dir = "data/dataset"
os.makedirs(base_dir, exist_ok=True)

#links
urls = [
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
]

#download csv files
for i, url in enumerate(urls, 1):
    filename = os.path.join(base_dir, f"goemotions_{i}.csv")
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already exists.")

#files
files = [f"goemotions_{i}.csv" for i in range(1, 4)]
paths = [os.path.join(base_dir, file) for file in files]

#read and merge csv files
dfs = [pd.read_csv(path) for path in paths]
merged_df = pd.concat(dfs, ignore_index=True)

#information
print(f"Merged df: {merged_df.shape}")
print(merged_df.head())

#save as a single csv file
output_path = os.path.join(base_dir, "goemotions_merged.csv")
merged_df.to_csv(output_path, index=False)
print(f"Merged dataset saved as: {output_path}")