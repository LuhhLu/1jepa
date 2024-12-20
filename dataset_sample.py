import os
import csv
import numpy as np
import random

# Path to the root folder
root_folder = "/teamspace/studios/sv-dataset/sav_train"
output_csv = "./data_path.csv"
output_npy = "./data_path_sample.npy"

# Data collection
data_entries = []

# Traverse each folder inside the root folder
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)

    # Check if it is a directory and its name matches the pattern
    if os.path.isdir(folder_path) and folder_name.startswith("sav_"):
        try:
            label = int(folder_name.split("_")[1])  # Extract label from folder name
        except ValueError:
            print(f"Skipping folder with invalid label: {folder_name}")
            continue

        # Traverse .mp4 files inside the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".mp4"):
                file_path = os.path.join(folder_path, file_name)
                data_entries.append((file_path, label))

# Write full dataset to CSV
with open(output_csv, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ')
    for entry in data_entries:
        csv_writer.writerow(entry)

# Randomly sample 10% of the data
sampled_entries = random.sample(data_entries, max(1, len(data_entries) // 10))

# Save sampled dataset to NPY
np.save(output_npy, sampled_entries)

print(f"CSV file generated at {output_csv}")
print(f"NPY file generated at {output_npy}")