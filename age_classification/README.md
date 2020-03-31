# Age Classification
This project allows users to train and test an age classification model from scratch, or use a pre-trained age classification model to classify the faces detected in a set of test images.

## Requirements

 - Python >= 3.6
 - PyTorch
 - Torchvision
 - Numpy
 - Scikit-learn
 - Matplotlib
 - [Pretrained models](https://github.com/Cadene/pretrained-models.pytorch)

## Usage: Pre-trained Model
If you have a set of images containing faces for which you would like to classify the faces into one of three classes that are age ranges of <=12, 13-17, or >=18, the relevant files are `extract_faces.py` and `classify_faces.py`. 

 1. Call the `extract_faces` function, providing it with the parameters of a string file path of the DSFD weights file and a string file path to the test image directory. The `extract_faces` boolean parameter must be set to true. The extracted faces will be saved to a `faces_testfoldername`directory (where `testfoldername` is whatever the name was of the folder containing the test images) in the same directory where original test images are located.
 2. Call the `classify_faces` function. This function requires three parameters, a string file path of the trained age classification model, a string file path to the directory containing the unclassified face images, and a batch size. This function will generate a `classified_faces` directory containing sub-directories with the classified images. 

## Usage: Training from Scratch
If training an age classification model from scratch on a dataset of faces, the relevant files are `train.py`, `test.py`, and `model.py`. The `model.py` script can be easily modified to change the architecture as well as the number of output class predictions. The `train.py` script relies on having a dataset structured in PyTorch's `ImageFolder` format. This script can also be easily modified to make any desired changes in the training process. Lastly, the `test.py` script is designed for visualizing face images as well as classification results with a small dataset, however the visualization component can easily be commented out to get accuracy/precision/recall statistics on a large test set. 

## Additional Information
Please feel free to modify and change these scripts as desired. Thanks, and enjoy!
