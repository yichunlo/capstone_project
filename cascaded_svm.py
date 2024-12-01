import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Function to load data from a pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to normalize hand landmarks
def normalize_landmarks(landmarks):
    # Wrist is at index 0
    wrist = landmarks[0]
    
    # Middle finger metacarpal: from wrist (0) to middle finger MCP (9)
    scale_bone = landmarks[9] - wrist
    scale = np.linalg.norm(scale_bone)
    
    # Centralize and scale
    normalized = (landmarks - wrist) / scale
    
    return normalized.flatten()

# Function to prepare data for training and validation
def prepare_data(data):
    X = []
    y = []
    for label, landmarks_list in data.items():
        for landmarks in landmarks_list:
            if landmarks is not None:
                normalized_landmarks = normalize_landmarks(landmarks)
                X.append(normalized_landmarks)
                y.append(label)
    return np.array(X), np.array(y)

# Function to train a cascaded SVM classifier
def train_cascaded_svm_classifier(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Number of training images: {len(X_train)}")
    print(f"Number of validation images: {len(X_test)}")

    # Stage 1: Coarse classification
    param_grid_stage1 = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    svm_stage1 = SVC(random_state=42)
    grid_search_stage1 = GridSearchCV(svm_stage1, param_grid_stage1, cv=5, n_jobs=-1, verbose=1)
    grid_search_stage1.fit(X_train, y_train)
    best_svm_stage1 = grid_search_stage1.best_estimator_

    # Stage 2: Fine classification
    # Use predictions from stage 1 to further train a more specialized model
    stage1_predictions = best_svm_stage1.predict(X_train)
    X_stage2, y_stage2 = X_train[stage1_predictions == y_train], y_train[stage1_predictions == y_train]

    param_grid_stage2 = {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf', 'poly']
    }
    svm_stage2 = SVC(random_state=42)
    grid_search_stage2 = GridSearchCV(svm_stage2, param_grid_stage2, cv=5, n_jobs=-1, verbose=1)
    grid_search_stage2.fit(X_stage2, y_stage2)
    best_svm_stage2 = grid_search_stage2.best_estimator_

    # Make predictions on the test set using the cascaded approach
    y_pred_stage1 = best_svm_stage1.predict(X_test)
    final_predictions = []
    for i, pred in enumerate(y_pred_stage1):
        if pred == y_test[i]:
            final_predictions.append(pred)
        else:
            final_predictions.append(best_svm_stage2.predict(X_test[i].reshape(1, -1))[0])

    # Print the best parameters and score for both stages
    print("Stage 1 Best parameters:", grid_search_stage1.best_params_)
    print("Stage 1 Best cross-validation score:", grid_search_stage1.best_score_)
    print("Stage 2 Best parameters:", grid_search_stage2.best_params_)
    print("Stage 2 Best cross-validation score:", grid_search_stage2.best_score_)

    # Print the classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test, final_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, final_predictions))

    return best_svm_stage1, best_svm_stage2

# Function to save the trained models
def save_models(models, model_paths):
    for model, path in zip(models, model_paths):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

# Function to validate the cascaded SVM model on a separate validation set
def validate_cascaded_model(validation_file, model_files):
    # Load validation data
    validation_data = load_data(validation_file)
    X_val, y_val = prepare_data(validation_data)

    print(f"Number of validation images: {len(X_val)}")

    # Check if validation data is empty
    if X_val.size == 0:
        print("No validation data available. Validation cannot be performed.")
        return

    # Load the trained models
    with open(model_files[0], 'rb') as f:
        model_stage1 = pickle.load(f)
    with open(model_files[1], 'rb') as f:
        model_stage2 = pickle.load(f)

    # Make predictions on the validation set using the cascaded approach
    y_pred_stage1 = model_stage1.predict(X_val)
    final_predictions = []
    for i, pred in enumerate(y_pred_stage1):
        if pred == y_val[i]:
            final_predictions.append(pred)
        else:
            final_predictions.append(model_stage2.predict(X_val[i].reshape(1, -1))[0])

    # Print the classification report and confusion matrix
    print("\nValidation Classification Report:")
    print(classification_report(y_val, final_predictions))
    print("\nValidation Confusion Matrix:")
    print(confusion_matrix(y_val, final_predictions))

    precision = precision_score(y_val, final_predictions, average='weighted')
    recall = recall_score(y_val, final_predictions, average='weighted')
    f1 = f1_score(y_val, final_predictions, average='weighted')

    print(f"\nValidation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # Paths to the balanced training and validation datasets
    training_file = "2d_hand_landmarks_balanced_training.pkl"  # Balanced training pickle file
    validation_file = "2d_hand_landmarks_validation.pkl"  # Validation pickle file
    model_files = ["svm_model_stage1.pkl", "svm_model_stage2.pkl"]  # Paths to save the trained models

    # Load and prepare the training data
    data = load_data(training_file)
    X, y = prepare_data(data)

    # Train the cascaded SVM classifier
    svm_model_stage1, svm_model_stage2 = train_cascaded_svm_classifier(X, y)

    # Save the models
    save_models([svm_model_stage1, svm_model_stage2], model_files)
    print(f"\nModels saved to {model_files}")

    # Perform validation
    validate_cascaded_model(validation_file, model_files)