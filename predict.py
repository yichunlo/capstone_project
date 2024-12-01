import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

def contains_any(source_string, substrings):
    for substring in substrings:
        if substring in source_string:
            return True
    return False

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    scale_bone = landmarks[9] - wrist
    scale = np.linalg.norm(scale_bone)
    normalized = (landmarks - wrist) / scale
    return normalized.flatten()

def process_dataset(dataset_path, model):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)

    left_hand_list = ['ben_jarashow_72','soul_life','keith_gamache_jr','deafwomynpride','michael_moore','willearl2','fairytales','goatman','brooklyn',
                    'biggreeneggtn','gmmotorcross','picard90','amy_cohen','guthrie','unknown_female_2','myles_de_bastion','Ichaim','nancy_rourke',
                    'monica_gaina','chickadee_32']

    for person_folder in os.listdir(dataset_path):
        person_folder_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(person_folder_path):
            continue

        extract_left_hand = contains_any(str(person_folder_path), left_hand_list)

        print(f"Processing folder: {person_folder}")
        for label_folder in os.listdir(person_folder_path):
            label_folder_path = os.path.join(person_folder_path, label_folder)
            if not os.path.isdir(label_folder_path):
                continue

            for filename in tqdm(sorted(os.listdir(label_folder_path))):
                if not filename.endswith('.jpg'):
                    continue

                file_path = os.path.join(label_folder_path, filename)
                image = cv2.imread(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                landmarks = None
                if extract_left_hand and results.left_hand_landmarks:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                elif not extract_left_hand and results.right_hand_landmarks:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

                if landmarks is not None:
                    normalized_landmarks = normalize_landmarks(landmarks)
                    prediction = model.predict([normalized_landmarks])[0]

                    # Rename the file with the predicted label
                    new_filename = f"{os.path.splitext(filename)[0]}_{prediction}.jpg"
                    new_file_path = os.path.join(label_folder_path, new_filename)
                    os.rename(file_path, new_file_path)
                else:
                    print(f"No hand landmarks detected in {file_path}")

    holistic.close()

if __name__ == "__main__":
    model_path = "svm_model_stage2.pkl"  # Path to your trained SVM model
    dataset_path = "validation_fine_grained"  # Replace with the path to your dataset

    # Load the trained SVM model
    model = load_model(model_path)

    # Process the dataset
    process_dataset(dataset_path, model)

    print("Processing complete. Files have been renamed with predicted labels.")