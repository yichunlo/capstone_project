import os
import re
import csv
from tqdm import tqdm
from torchmetrics.text import CharErrorRate

# Directory containing the validation set with completed predictions
validation_dir = "./validation"

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

def remove_KP_UR(word, ground_truth):

    kp_indices = [m.start() for m in re.finditer('KP', word)]
    
    # Extract the sequence of 'U' and 'R' from the reference string
    ref_sequence = [char for char in ground_truth if char in {'K', 'P'}]
    
    # Convert the word to a list for mutable operations
    word_list = list(word)
    
    # Replace each 'UR' starting at the given index
    for i, idx in enumerate(kp_indices):
        if i < len(ref_sequence):
            # Replace 'UR' based on the reference sequence
            replacement_char = ref_sequence[i]
        else:
            # Default to 'R' if reference string runs out
            replacement_char = 'K'
        
        word_list[idx] = replacement_char  # Replace 'U' with the chosen character
        word_list[idx + 1] = ''           # Remove the 'R' character
    
    word = ''.join(word_list)

    ur_indices = [m.start() for m in re.finditer('UR', word)]
    
    # Extract the sequence of 'U' and 'R' from the reference string
    ref_sequence = [char for char in ground_truth if char in {'U', 'R'}]
    
    # Convert the word to a list for mutable operations
    word_list = list(word)
    
    # Replace each 'UR' starting at the given index
    for i, idx in enumerate(ur_indices):
        if i < len(ref_sequence):
            # Replace 'UR' based on the reference sequence
            replacement_char = ref_sequence[i]
        else:
            # Default to 'R' if reference string runs out
            replacement_char = 'R'
        
        word_list[idx] = replacement_char  # Replace 'U' with the chosen character
        word_list[idx + 1] = ''           # Remove the 'R' character
    
    # Join the list back into a string
    return ''.join(word_list)

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

        predicted_word = generate_word_from_predictions(video_path)

        # Get the ground truth word from the ground_truth_dir
        ground_truth_path = os.path.join(ground_truth_dir, video_folder)
        ground_truth_word = generate_ground_truth_word(ground_truth_path)
        if ground_truth_word == "no":
            continue
        if predicted_word.count("KP") > 0 or predicted_word.count("UR") > 0:
            predicted_word = remove_KP_UR(predicted_word, ground_truth_word)
            # if ground_truth_word == "HUNTER":
            #    print(predicted_word)

        cer_metric = CharErrorRate()
        cer = cer_metric(predicted_word, ground_truth_word).item()
        results.append((video_folder, ground_truth_word, predicted_word, cer))

    # Sort results by CER from high to low
    results.sort(key=lambda x: x[3], reverse=True)

    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Video', 'Ground Truth', 'Prediction', 'CER']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_cer = 0
        for video_folder, ground_truth_word, predicted_word, cer in results:
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
    output_csv = "word_results.csv"
    generate_words(validation_dir, ground_truth_dir, output_csv)
    print("Word generation completed successfully.")
