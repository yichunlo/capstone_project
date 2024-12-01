import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


# Define the directories containing the datasets
validation_dir = "./validation"  # Replace with your dataset path
validation_fine_grained_dir = "./validation_fine_grained"  # Ground truth dataset path
plot_dir = "./plot"  # Directory to save plots

# Create the plot directory if it does not exist
os.makedirs(plot_dir, exist_ok=True)

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

# Function to process a video folder and plot the sum of distances
def plot_sum_of_distances(video_path, output_plot_path, use_left_hand):
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

    # Calculate the distances between consecutive frames
    frame_distances = [[], []]
    for i in range(len(hand_shapes) - 1):
        orig_frame_landmarks = np.array([[lm.x, lm.y] for lm in hand_shapes[i]])
        next_frame_landmarks = np.array([[lm.x, lm.y] for lm in hand_shapes[i + 1]])
        indices = [4, 8, 12, 16, 20]
        rows_arr1 = orig_frame_landmarks[indices, :2] - orig_frame_landmarks[0, :2]  # Selecting only the first 2 columns (x and y)
        rows_arr2 = next_frame_landmarks[indices, :2] - next_frame_landmarks[0, :2]  # Same for the second array
        distances = np.sqrt(np.sum((rows_arr1 - rows_arr2) ** 2, axis=1))
        distance_sum = np.sum(distances)
        frame_distances[0].append(i)
        frame_distances[1].append(distance_sum)

    # Apply Gaussian filter for smoothing
    frame_distances[1] = gaussian_filter1d(frame_distances[1], sigma=1.5)

    # Plot the sum of distances
    if sum_distances:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(image_indices, sum_distances, marker='o', label='Sum of Distances (Wrist to Finger Points)')
        ax.plot(frame_distances[0], frame_distances[1], marker='x', linestyle='--', color='r', label='Consecutive Frame Distances')

        # Annotate with predicted labels
        for x, y, label in zip(image_indices, sum_distances, predicted_labels):
            ax.text(x, y, label, fontsize=8, color='black')

        plt.xlabel('Image Index')
        plt.ylabel('Sum of Distances')
        plt.title(f'Sum of Distances vs. Image Index - {os.path.basename(video_path)}')
        plt.legend()
        plt.grid()

        # Save the plot without showing
        plt.savefig(output_plot_path)
        plt.close(fig)  # Close the figure to prevent it from being shown
    else:
        print("No hand landmarks detected in any of the images.")

    mp_holistic_instance.close()

# Function to check if the video should be processed based on ground truth labels
def should_process_video(video_folder):
    gt_folder = os.path.join(validation_fine_grained_dir, os.path.basename(video_folder))
    if not os.path.exists(gt_folder):
        return True

    # Iterate through each subfolder in the ground truth folder
    for subfolder in os.listdir(gt_folder):
        subfolder_path = os.path.join(gt_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for image_file in os.listdir(subfolder_path):
                if image_file.endswith(".jpg"):
                    label = image_file.split('_')[-1].split('.')[0]
                    if label in ['J', 'Z']:
                        return False
    return True


if __name__ == "__main__":
    # Loop through all video folders under the validation directory
    for video_folder_name in os.listdir(validation_dir):
        video_folder_path = os.path.join(validation_dir, video_folder_name)

        # Skip if it's not a directory
        if not os.path.isdir(video_folder_path):
            continue

        # Determine if the video should be processed based on ground truth labels
        if not should_process_video(video_folder_path):
            print(f"Skipping video {video_folder_name} due to presence of 'J' or 'Z' labels in ground truth.")
            continue

        # Determine if the folder is in the left_hand_list
        use_left_hand = video_folder_name in left_hand_list

        # Set output plot path in the 'plot' folder
        output_plot_path = os.path.join(plot_dir, f"sum_of_distances_plot_{video_folder_name}.png")

        # Plot the sum of distances for the current video folder
        plot_sum_of_distances(video_folder_path, output_plot_path, use_left_hand)
        print(f"Plot saved to {output_plot_path}")
