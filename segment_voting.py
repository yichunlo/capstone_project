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
output_csv = "segmented_predictions.csv"  # CSV file to save the predicted words

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

if __name__ == "__main__":
    results = []

    # Loop through all video folders under the validation directory
    for video_folder_name in os.listdir(validation_dir):
        video_folder_path = os.path.join(validation_dir, video_folder_name)

        # Skip if it's not a directory
        if not os.path.isdir(video_folder_path):
            continue

        # Skip videos with less than 35 images
        num_images = len([img for img in os.listdir(video_folder_path) if img.endswith(".jpg")])
        if num_images <= 35:
            continue

        # Determine if the folder is in the left_hand_list
        use_left_hand = video_folder_name in left_hand_list

        # Segment characters and predict word for the current video folder
        predicted_word = segment_characters(video_folder_path, use_left_hand)

        # Get the ground truth word from the validation_fine_grained directory
        ground_truth_path = os.path.join(validation_fine_grained_dir, video_folder_name)
        ground_truth_word = ""
        if os.path.exists(ground_truth_path):
            subfolders = sorted([f for f in os.listdir(ground_truth_path) if os.path.isdir(os.path.join(ground_truth_path, f))])
            for subfolder in subfolders:
                match = re.match(r'^[1-9]-(?P<label>[A-Z]+)$', subfolder)
                if match:
                    ground_truth_word += match.group('label')

        results.append((video_folder_name, predicted_word, ground_truth_word))
        print(f"Predicted word for {video_folder_name}: {predicted_word}, Ground truth: {ground_truth_word}")

    # Write the results to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Video', 'Predicted Word', 'Ground Truth', 'CER']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_cer = 0
        cer_metric = CharErrorRate()
        for video_folder_name, predicted_word, ground_truth_word in results:
            cer = cer_metric(predicted_word, ground_truth_word).item()
            total_cer += cer
            writer.writerow({
                'Video': video_folder_name,
                'Predicted Word': predicted_word,
                'Ground Truth': ground_truth_word,
                'CER': f"{cer:.4f}"
            })

    overall_cer = total_cer / len(results) if results else 0
    print(f"Overall CER: {overall_cer:.4f}")
    print(f"Segmented predictions saved to {output_csv}")
