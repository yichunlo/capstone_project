import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import Counter
from torchmetrics.text import CharErrorRate
import re

# Define the directories containing the datasets
validation_dir = "./validation"  # Replace with your dataset path
validation_fine_grained_dir = "./validation_fine_grained"  # Ground truth dataset path
output_csv = "combined_word_results.csv"  # CSV file to save the predicted words

left_hand_list = ['ben_jarashow_72', 'soul_life', 'keith_gamache_jr', 'deafwomynpride', 'michael_moore', 'willearl2',
                  'fairytales', 'goatman', 'brooklyn', 'biggreeneggtn', 'gmmotorcross', 'picard90', 'amy_cohen', 
                  'guthrie', 'unknown_female_2', 'myles_de_bastion', 'Ichaim', 'nancy_rourke', 'monica_gaina', 'chickadee_32']

# Mediapipe Holistic solution
mp_holistic = mp.solutions.holistic

# Function to calculate the sum of distances between the wrist and all finger points
def calculate_sum_of_distances(landmarks):
    wrist = landmarks[0]
    indices = [4, 8, 12, 16, 20]  # Indices for thumb, index, middle, ring, and pinky tips
    sum_distance = 0.0

    # Calculate the distance between wrist and specified finger points
    for i in indices:
        finger_point = landmarks[i]
        distance = np.linalg.norm([wrist.x - finger_point.x, wrist.y - finger_point.y, wrist.z - finger_point.z])
        sum_distance += distance

    return sum_distance

# Function to generate the word for each video based on consecutive predictions
def generate_word_from_predictions(video_path):
    images = sorted(os.listdir(video_path))
    word = ""
    current_char = ""
    count = 0

    for image_file in images:
        if not image_file.endswith(".jpg"):
            continue
        match = re.search(r'_(?P<label>[A-Z]+)\.jpg$', image_file)
        if match:
            label = match.group('label')
            if label == current_char:
                count += 1
            else:
                if count > 1:  # Only add characters that appear consecutively more than once
                    if current_char != word[-1:]:
                        word += current_char
                elif count == 2:  # Mark for voting-based re-evaluation
                    word += current_char + '2'
                current_char = label
                count = 1

    # Add the last character if it appears more than once
    if count > 1 and current_char != word[-1:]:
        word += current_char
    elif len(images) == 1:  # Special case: if there is only one labeled image, add it to the word
        word += current_char

    # Remove redundant consecutive "UR" or "KP" pairs
    cleaned_word = re.sub(r'(UR)+', 'UR', word)
    cleaned_word = re.sub(r'(KP)+', 'KP', cleaned_word)
    
    return cleaned_word

# Function to generate the ground truth word from the ground truth directory
def generate_ground_truth_word(ground_truth_path):
    label_folders = sorted([f for f in os.listdir(ground_truth_path) if os.path.isdir(os.path.join(ground_truth_path, f))])
    word = ""

    for label_folder in label_folders:
        match = re.match(r'^[1-9]-(?P<label>[A-Z]+)$', label_folder)
        if match:
            label = match.group('label')
            if label in ['J', 'Z']:
                return "no"
            word += label

    return word

# Function to process a video folder and segment characters based on valleys and peaks
def segment_characters(video_path, use_left_hand):
    mp_holistic_instance = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)
    image_indices = []
    sum_distances = []
    predicted_labels = []

    # Store hand landmarks for consecutive frame distance calculation
    hand_shapes = []

    # Iterate through each image in the video folder
    for idx, image_file in enumerate(tqdm(sorted(os.listdir(video_path)))):
        if not image_file.endswith(".jpg"):
            continue

        file_path = os.path.join(video_path, image_file)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Could not read image: {file_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_holistic_instance.process(image_rgb)

        if use_left_hand and results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks.landmark
            hand_shapes.append(landmarks)
            sum_distance = calculate_sum_of_distances(landmarks)
            image_indices.append(idx)
            sum_distances.append(sum_distance)
            predicted_label = image_file.split('_')[-1].split('.')[0]
            predicted_labels.append(predicted_label)
        elif not use_left_hand and results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark
            hand_shapes.append(landmarks)
            sum_distance = calculate_sum_of_distances(landmarks)
            image_indices.append(idx)
            sum_distances.append(sum_distance)
            predicted_label = image_file.split('_')[-1].split('.')[0]
            predicted_labels.append(predicted_label)
        else:
            print(f"No hand landmarks detected for image: {image_file}")

    # Apply Gaussian filter for smoothing
    sum_distances_smoothed = gaussian_filter1d(sum_distances, sigma=1.5)

    # Find peaks of the smoothed sum of distances
    peaks, _ = find_peaks(sum_distances_smoothed, distance=5)  # Adjust distance parameter as needed
    
    # Segment the video based on valleys between peaks
    predicted_word = ""
    for i in range(len(peaks) - 1):
        start_index = peaks[i]
        end_index = peaks[i + 1]
        # Use only the lowest valley point in the segment
        segment_labels = predicted_labels[start_index + 2:end_index - 2]
        if segment_labels:
            majority_label = Counter(segment_labels).most_common(1)[0][0]
            predicted_word += majority_label

    mp_holistic_instance.close()
    return predicted_word

# Function to generate words from predictions and ground truth
def generate_words(validation_dir, ground_truth_dir, output_csv):
    results = []

    for video_folder in tqdm(os.listdir(validation_dir)):
        video_path = os.path.join(validation_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        # Overlook videos with empty images
        images = [img for img in os.listdir(video_path) if img.endswith(".jpg")]
        if not images:
            continue

        # Check if video contains "J" or "Z" label and overlook it if so
        if any(re.search(r'_[JZ]\.jpg$', img) for img in images):
            continue

        # Use the original consecutive method by default
        predicted_word = generate_word_from_predictions(video_path)

        # If the video has more than 35 images and contains segments labeled as '2', use voting method
        if len(images) > 35 and '2' in predicted_word:
            use_left_hand = video_folder in left_hand_list
            voting_word = segment_characters(video_path, use_left_hand)
            # Replace segments marked with '2' using the voting result
            updated_word = ""
            voting_word_idx = 0
            for char in predicted_word:
                if char == '2':
                    updated_word += voting_word[voting_word_idx]
                    voting_word_idx += 1
                else:
                    updated_word += char
            predicted_word = updated_word

        # Get the ground truth word from the ground_truth_dir
        ground_truth_path = os.path.join(ground_truth_dir, video_folder)
        ground_truth_word = generate_ground_truth_word(ground_truth_path)
        if ground_truth_word == "no":
            continue

        results.append((video_folder, ground_truth_word, predicted_word))

        # Logging for debugging
        print(f"Video: {video_folder} | Predicted Word: {predicted_word} | Ground Truth: {ground_truth_word}")

    # Write the results to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Video', 'Ground Truth', 'Prediction', 'CER']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_cer = 0
        for video_folder, ground_truth_word, predicted_word in results:
            cer_metric = CharErrorRate()
            cer = cer_metric(predicted_word, ground_truth_word).item()
            total_cer += cer
            writer.writerow({
                'Video': video_folder,
                'Ground Truth': ground_truth_word,
                'Prediction': predicted_word,
                'CER': f"{cer:.4f}"
            })

    overall_cer = total_cer / len(results) if results else 0
    print(f"Overall CER: {overall_cer:.4f}")
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    # Generate words from predictions and ground truth
    ground_truth_dir = "./ChicagoFSWild_fine_grained"
    output_csv = "combine_word_results.csv"
    generate_words(validation_dir, ground_truth_dir, output_csv)
    print("Word generation completed successfully.")
