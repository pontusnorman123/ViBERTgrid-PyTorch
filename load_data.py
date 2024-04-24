from datasets import load_dataset

# this dataset uses the new Image feature :)
#dataset = load_dataset("Hyeoli/layoutlmv3_cord")
#dataset = load_dataset("pontusnorman123/swe_set2")
dataset_dict = load_dataset("pontusnorman123/sweset3")

import csv
import os
from datasets import DatasetDict, Dataset

# Function to create the directory structure
def create_directory_structure(base_path='dataset'):
    os.makedirs(base_path, exist_ok=True)
    for dtype in ['training_data', 'testing_data']:
        os.makedirs(f"{base_path}/{dtype}/_label_csv", exist_ok=True)
        os.makedirs(f"{base_path}/{dtype}/images", exist_ok=True)

# Function to write the CSV file for each image
def write_csv_files(data, label_dir, image_dir):
    for row in data:
        image_id = row['id']  # Assuming 'id' is a unique identifier for each image
        image_path = f"{image_dir}/{image_id}.png"
        csv_path = f"{label_dir}/{image_id}.csv"

        # Save the image
        row['image'].save(image_path)

        # Write the CSV for the corresponding image
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["left", "top", "right", "bot", "text", "data_class", "pos_neg"])  # CSV Header

            # Write data to CSV
            for bbox, word, tag in zip(row['bboxes'], row['words'], row['ner_tags']):
                left, top, right, bot = bbox
                data_class = tag
                pos_neg = 2 if data_class == 4 else 1
                writer.writerow([left, top, right, bot, word, data_class, 1])

# Main function to generate the CSV files and folder structure
def generate_csvs_and_structure(dataset_dict):
    create_directory_structure()

    # Writing CSV files and saving images for training and testing data
    for dtype, dir_name in zip(['train', 'test'], ['training_data', 'testing_data']):
        data = dataset_dict[dtype]
        label_dir = f"dataset/{dir_name}/_label_csv"
        image_dir = f"dataset/{dir_name}/images"
        write_csv_files(data, label_dir, image_dir)


# Run the function to generate CSVs and folder structure
generate_csvs_and_structure(dataset_dict)