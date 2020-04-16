# Age Classification
This project allows users to train and test an age classification model from scratch, or use a pre-trained age classification model to classify the faces detected in a set of test images.

## Requirements
Install [Anaconda](https://www.anaconda.com/distribution/) and set up a Conda virtual environment as follows:
 1. Create a new virtual environment: `conda create -n age_classification python=3.7`
 2. Activate the environment: `conda activate age_classification`
 3. Install pip: `conda install pip`
 4. Install PyTorch: `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
 5. Install Scikit-learn: `conda install scikit-learn`
 6. Install Matplotlib: `conda install matplotlib`
 7. Install Pandas: `conda install pandas`
 8. Install [Pretrained models](https://github.com/Cadene/pretrained-models.pytorch): `pip install pretrainedmodels`
 9. Install OpenCV: `conda install -c conda-forge opencv`

## Usage: Pre-trained Model
If you have a set of images containing faces for which you would like to classify the faces into one of three classes that are age ranges of <=12, 13-17, or >=18, the relevant files are `extract_faces.py` and `classify_faces.py`. 

 1. Call the `extract_faces` function, providing it with the parameters of a string file path of the DSFD weights file and a string file path to the test image directory. A pre-trained DSFD set of weights for three class age classification can be downloaded [here](https://drive.google.com/open?id=1LTbmKjHO3Jn7_NbJ9mf60_9zAtmbdmIq). The `extract_faces` boolean parameter must be set to true. The extracted faces will be saved to a `faces_testfoldername`directory (where `testfoldername` is whatever the name was of the folder containing the test images) in the same directory where original test images are located.
 2. Call the `classify_faces` function. This function requires three parameters, a string file path of the trained age classification model, a string file path to the directory containing the unclassified face images, and a batch size. This function will generate a CSV file named `face_classification_labels` containing the image path names and their corresponding class label, where for the three class classification model, label 0 means the face is less than 12 years old, label 1 means it is in between 12 to 17 years of age, and label 2 means it is 18 years or older. 

## Usage: Pre-trained Model: Checking Accuracy
If you would like to check the accuracy of the pre-trained model linked above, the `test.py` script can be used. The test dataset must be only face images (if not, the `extract_faces.py` script may be useful). The images must be located in one main directory (e.g. `test_images`), with three subdirectories (or less, depending on your test dataset), named `12_or_less`, `13_to_17`, `18_and_above`, where the faces fitting those age labels are located. An example directory with this structure is available in this repository, in the `three_class_test_images` folder; both full images as well as extracted faces divided into classes are provided. 

Before running the `test.py` script, make sure to open it and change the `images_root` variable's file path to the main directory where the three class subfolders are located. 

## Usage: Training from Scratch
If training an age classification model from scratch on a dataset of faces, the relevant files are `train.py`, `test.py`, and `model.py`. The `model.py` script can be easily modified to change the architecture as well as the number of output class predictions. The `train.py` script relies on having a dataset structured in PyTorch's `ImageFolder` format. This script can also be easily modified to make any desired changes in the training process. Lastly, the `test.py` script is designed for visualizing face images as well as classification results with a small dataset, however the visualization component can easily be commented out to get accuracy/precision/recall statistics on a large test set. 

## Additional Information
Please feel free to modify and change these scripts as desired. Thanks, and enjoy!
