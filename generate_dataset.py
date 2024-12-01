import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import random
import re
from tqdm import tqdm
import shutil

# Define the directories containing the dataset
dataset_frames_dir = "./ChicagoFSWild-Frames/train"  # Replace with your dataset path
dataset_labels_dir = "./ChicagoFSWild_fine_grained"  # Replace with your label path
validation_output_dir = "./validation_fine_grained"  # Directory to save validation folders

# Mediapipe Holistic solution
mp_holistic = mp.solutions.holistic

# Function to extract hand landmarks from an image
def extract_hand_landmarks(image_path, holistic, extract_left_hand):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    landmarks = None

    if extract_left_hand and results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    elif not extract_left_hand and results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    
    return landmarks

# Function to randomly pick folders for validation with an acceptable ratio of images
def get_balanced_validation_folders(train_dir, labels_dir):
    all_folders = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and os.path.exists(os.path.join(labels_dir, d))]

    while True:
        random.shuffle(all_folders)
        validation_folders = all_folders[:290]
        training_folders = all_folders[290:]

        # Count images in training and validation folders
        training_image_count = sum(len(files) for root, _, files in os.walk(train_dir) if root in training_folders and any(file.endswith(".jpg") for file in files))
        validation_image_count = sum(len(files) for root, _, files in os.walk(train_dir) if root in validation_folders and any(file.endswith(".jpg") for file in files))
        print(f'Train / Validation Ratio: {training_image_count / validation_image_count}')
        if validation_image_count > 0 and 4 <= training_image_count / validation_image_count <= 5:
            print(f"Training images: {training_image_count}, Validation images: {validation_image_count}")
            return validation_folders

# Function to check if a folder should extract left hand landmarks
def contains_any(source_string, substrings):
    for substring in substrings:
        if substring in source_string:
            return True
    return False

# Function to map images from frames directory to labels in fine-grained directory and save validation folders
def generate_mapped_landmarks_dataset(frames_dir, labels_dir, output_file, validation_output_file, validation_folders):
    data = {}
    validation_data = {}
    left_hand_list = ['ben_jarashow_72','soul_life','keith_gamache_jr','deafwomynpride','michael_moore','willearl2','fairytales','goatman','brooklyn',
                    'biggreeneggtn','gmmotorcross','picard90','amy_cohen','guthrie','unknown_female_2','myles_de_bastion','Ichaim','nancy_rourke',
                    'monica_gaina','chickadee_32']

    training_image_count = 0
    validation_image_count = 0

    # Copy validation folders to validation_output_dir
    for validation_folder in validation_folders:
        dest_folder = validation_folder.replace(frames_dir, validation_output_dir)
        shutil.copytree(validation_folder, dest_folder, dirs_exist_ok=True)

    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
        for root, _, files in tqdm(os.walk(frames_dir)):
            if root in validation_folders:
                dataset = validation_data
            else:
                dataset = data

            extract_left_hand = contains_any(root, left_hand_list)

            for file in files:
                if file.endswith(".jpg"):
                    # Construct corresponding label path
                    person_folder = os.path.basename(root)
                    label_folder = os.path.join(labels_dir, person_folder)
                    if not os.path.exists(label_folder):
                        continue

                    # Extract the ground truth label from the corresponding label folder
                    for label_root, _, label_files in os.walk(label_folder):
                        for label_file in label_files:
                            if label_file.startswith(file.split('.')[0]):  # Match the base filename
                                label = os.path.basename(label_root).split('-')[-1]  # Extract the label

                                # Skip label "None", "J", and "Z"
                                if label in ["None", "J", "Z"]:
                                    continue

                                # Ensure the label is in the correct format
                                if re.match(r'^[1-9]-[A-Y]$', os.path.basename(label_root)) is None:
                                    continue

                                image_path = os.path.join(root, file)
                                landmarks = extract_hand_landmarks(image_path, holistic, extract_left_hand)

                                # Group "U" and "R" as "UR"
                                if label in ["U", "R"]:
                                   label = "UR"
                                if label in ["K", "P"]:
                                   label = "KP"

                                if landmarks is not None:
                                    if label not in dataset:
                                        dataset[label] = []
                                    dataset[label].append(landmarks)
                                    if dataset == validation_data:
                                        validation_image_count += 1
                                    else:
                                        training_image_count += 1

    for label in data:
        data[label] = [np.array(landmark) for landmark in data[label]]
    for label in validation_data:
        validation_data[label] = [np.array(landmark) for landmark in validation_data[label]]

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    with open(validation_output_file, 'wb') as f:
        pickle.dump(validation_data, f)

    print(f"Number of training images: {training_image_count}")
    print(f"Number of validation images: {validation_image_count}")

if __name__ == "__main__":
    validation_folders = get_balanced_validation_folders(dataset_frames_dir, dataset_labels_dir)
    output_file = "2d_hand_landmarks_training.pkl"  # Training output pickle file
    validation_output_file = "2d_hand_landmarks_validation.pkl"  # Validation pickle file
    
    generate_mapped_landmarks_dataset(dataset_frames_dir, dataset_labels_dir, output_file, validation_output_file, validation_folders)
    print(f"Training dataset saved to {output_file}")
    print(f"Validation dataset saved to {validation_output_file}")
