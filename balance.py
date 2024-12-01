import pickle
import numpy as np
from sklearn.utils import resample

def balance_dataset(input_file, output_file):
    # Load the pickle file
    with open(input_file, 'rb') as f:
        landmarks_data = pickle.load(f)
    
    # Remove labels 'Z' and 'J'
    if 'Z' in landmarks_data:
        del landmarks_data['Z']
    if 'J' in landmarks_data:
        del landmarks_data['J']
    
    # Calculate mean count of landmarks
    counts = [len(landmarks) for landmarks in landmarks_data.values()]
    mean_count = int(np.mean(counts))
    
    print(f"Mean count of landmarks: {mean_count}")
    
    # Balance the dataset
    balanced_data = {}
    for label, landmarks in landmarks_data.items():
        if len(landmarks) > mean_count:
            # Downsample
            balanced_data[label] = resample(landmarks, n_samples=mean_count, random_state=42)
        elif len(landmarks) < mean_count:
            # Upsample
            balanced_data[label] = resample(landmarks, n_samples=mean_count, replace=True, random_state=42)
        else:
            # Keep as is
            balanced_data[label] = landmarks
        
        print(f"Label {label}: {len(landmarks)} -> {len(balanced_data[label])}")
    
    # Save the balanced data back to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(balanced_data, f)
    
    # Print the number of images in each set after balancing
    total_images = sum(len(landmarks) for landmarks in balanced_data.values())
    print(f"Total number of images after balancing: {total_images}")
    print(f"Balanced data saved to {output_file}")

if __name__ == "__main__":
    input_file = "2d_hand_landmarks_training.pkl"  # Replace with your input pickle file path if different
    output_file = "2d_hand_landmarks_balanced_training.pkl"  # Name of the new pickle file
    
    balance_dataset(input_file, output_file)
