import os
import csv

# Path to the root folder
root_folder = "/teamspace/studios/sv-dataset/sav_train"
output_csv = "./output.csv"

# Open the output CSV file
with open(output_csv, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=' ')

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
                    csv_writer.writerow([file_path, label])

print(f"CSV file generated at {output_csv}")