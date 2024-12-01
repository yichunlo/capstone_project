import os
import re
from tqdm import tqdm

# Directories
validation_dir = "./validation_fine_grained"  # Directory containing the validation set in fine-grained structure

def analyze_label_accuracy(validation_dir):
    total_images = {}
    correct_predictions = {}
    total_correct = 0
    total_images_count = 0

    # Iterate through each person folder
    for person_folder in tqdm(os.listdir(validation_dir)):
        person_folder_path = os.path.join(validation_dir, person_folder)
        if not os.path.isdir(person_folder_path):
            continue

        # Iterate through each label folder within the person folder
        for label_folder in os.listdir(person_folder_path):
            label_folder_path = os.path.join(person_folder_path, label_folder)
            if not os.path.isdir(label_folder_path):
                continue

            # Extract the label from the folder name
            folder_label = label_folder.split('-')[-1]
            if folder_label in ['R', 'U']:
               folder_label = 'UR'
            elif folder_label in ['K', 'P']:
                folder_label = 'KP'

            if folder_label not in total_images:
                total_images[folder_label] = 0
                correct_predictions[folder_label] = 0

            # Iterate through each image in the label folder
            for image_file in os.listdir(label_folder_path):
                if image_file.endswith(".jpg"):
                    total_images[folder_label] += 1
                    total_images_count += 1

                    # Extract the label from the image filename
                    image_label_match = re.search(r'_(?P<label>[A-Z]+)\.jpg$', image_file)
                    if image_label_match:
                        image_label = image_label_match.group('label')

                        if image_label in ['R', 'U']:
                           image_label = 'UR'
                        elif image_label in ['K', 'P']:
                            image_label = 'KP'

                        # Check if the folder label matches the image label
                        if folder_label == image_label:
                            correct_predictions[folder_label] += 1
                            total_correct += 1

    # Calculate and print accuracy for each label
    for label in sorted(total_images.keys(), key=lambda x: (correct_predictions[x] / total_images[x] if total_images[x] > 0 else 0.0)):
        if total_images[label] > 0:
            accuracy = (correct_predictions[label] / total_images[label]) * 100
        else:
            accuracy = 0.0
        print(f"Label: {label}, Total images: {total_images[label]}, Correct predictions: {correct_predictions[label]}, Accuracy: {accuracy:.2f}%")

    # Calculate and print overall accuracy
    if total_images_count > 0:
        overall_accuracy = (total_correct / total_images_count) * 100
    else:
        overall_accuracy = 0.0
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    analyze_label_accuracy(validation_dir)
