# Fingerspelling Recognition

This is the capstone project for fingerspelling recognition. The model is trained by cascaded SVM method. To run the experiment, you have to put ```ChicagoFSWild_fine_grained``` and ```ChicagoFSWild_Frames``` folders under the ```fingerspelling_recognition``` folder.

## Usage

- To run the web, use

    ```=shell
    python app.py
    ```

    I use ```flask``` to create the web backend, and only simple HTML as UI, no other framework. The html files is under the ```templates``` directory.

## Experiment steps

### First stage: Train and Test

- To train the fingerspelling recognition model, first let's generate the datasets with:

    ```=shell
    python generate_dataset.py
    ```

    you will get the normalized landmarks as the training dataset, and also get the validation set. Notice that the datasets are randomly generated, so the result will be different in each time you run the script.

- Next, we have to balance the dataset to make the amount of each character becomes equal:

    ```=shell
    python balance.py
    ```

- The dataset used for training is ```2d_hand_landmarks_balanced_training.pkl```. To train the cascaded SVM model, run:

    ```=shell
    python cascaded_svm.py
    ```

    You will get a model named ```svm_model_stage2.pkl```.

- After the training is done, we can use ```mapping_validation.py``` and ```predict.py``` to predict the character on the images. Run:

    ```=shell
    python mapping_validation.py
    python predict.py
    ```

- If you want to see how good the result is for each character, you can run:

    ```=shell
    python analyze.py
    ```

### Second Stage: Check the Character Error Rate (CER) result

- To check the CER result, run:

    ```=shell
    python base_cer.py
    ```

    The CER is calculated by the module ```CharErrorRate``` class in ```torchmetrics.text```.

## Experiment Results

- Validation Precision: 0.8759
- Validation Recall: 0.8683
- Validation F1 Score: 0.8699
- Prediction:
 ![image](https://github.com/user-attachments/assets/f7ab0e4b-6b59-4105-b3ff-52ed2999f79a)
- Character Error Rate (CER): 0.3045
