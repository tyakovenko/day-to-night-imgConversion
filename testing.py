#check the dataset download link via image sources; all of the images taken from UTC-SIT testing set 1; other training and testing sets contain more data

#new data set https://www.kaggle.com/datasets/stevemark/daynight-dataset
#got all the time stamps for 9:27 and 20:28
import os
import csv

# Folder containing the txt files
folder_path = "/media/taya/FLASHDATA/archive(6)/DNIM/time_stamp"

# Output CSV file
output_csv = "output.csv"

rows = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        txt_path = os.path.join(folder_path, filename)

        with open(txt_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # Expected format:
                # d2.jpg 20151101 09 27
                if len(parts) >= 4:
                    jpg_title = parts[0]

                    try:
                        number1 = int(parts[-2])
                        number2 = int(parts[-1])
                    except ValueError:
                        continue

                    # Apply filter
                    if (
                            (number1 == 9 and 20 <= number2 <= 30) or
                            (number1 == 21 and 20 <= number2 <= 30)
                    ):
                        rows.append([filename, jpg_title, number1, number2])

# Write filtered results to CSV
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["text_file_name", "jpg_image_title", "number1", "number2"])
    writer.writerows(rows)

print("Filtered CSV file created successfully.")