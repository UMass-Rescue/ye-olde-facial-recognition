# Face Clustering
This project allows users to extract faces from a set of images and cluster those faces based on similarity, thus allowing for the detection and examination of similar-looking faces across the set of images.

Running the scripts in this repository in the order listed below will complete a full pipeline of face detection, extraction, encoding, and lastly, clustering.

## Requirements

 - Python >= 3.6
 - PyTorch
 - Torchvision
 - OpenCV
 - Numpy
 - Pandas
 - Scikit-learn
 - Matplotlib
 - [Dlib](https://pypi.org/project/dlib/)
 - [Face recognition](https://github.com/ageitgey/face_recognition)

## Step 1: Face Extraction

To extract faces from a directory of images, the `extract_faces` function can be called from the `extract_faces.py` script. 

This function requires four parameters: 

 1. Path to the image directory as a string
 2. Path to the weights file for the DSFD face detector as a string. The weights file can be downloaded from the [original DSFD repository](https://github.com/TencentYoutuResearch/FaceDetection-DSFD). The default directory for storing the weights file is `face_detector/dsfd_inference/dsfd/weights/`
 3. The `extract_faces` boolean parameter determines whether to save the extracted faces into a separate directory as images. If set to true, the script will save all the extracted faces into a separate `faces_testfoldername` (where `testfoldername` is whatever the name was of the folder containing the test images) directory where the original test images are located. **This parameter must be set to true if running the full face clustering pipeline, as the extracted face images are necessary for clustering in step 3.**
 4. The save_bounding_boxes parameter determines whether to save the extracted bounding boxes into a separate CSV file. If set to true, the script will save the bounding box coordinates of the extracted faces into `all_face_bounding_boxes.csv`, which will be saved to the same directory where the script is located. **This parameter must be set to true if running the full face clustering pipeline, as the bounding box data is necessary for generating face embeddings in step 2**. 

## Step 2: Generating Embeddings

Using the extracted bounding box data from the previous step, 128-dimensional encodings can be generated for all of the detected faces. In order to do this, call the `generate_embeddings` function. 

This function requires two parameters: 

 1. Path to the bounding box data file, generated in the previous step as `all_face_founding_boxes.csv`
 2. Path to the image directory as a string

The script will save the generated face encodings into a `encodings.pickle` file, which will be saved in the same directory as the script. 

## Step 3: Clustering Embeddings

Using the previously generated 128-dimensional encodings the extracted face images, the `cluster_embeddings` function can be called from `cluster_embeddings.py`. This will use the Chinese Whispers algorithm to cluster and sort the face images. 

This function requires one parameter: 

 1. Path to the face encodings data file, generated in the previous step as `encodings.pickle`. 

The clustered faces will automatically be saved in a new `clustered_faces` directory, located in the same directory as where the face images directory is located. The clustered face images will be named in the following format: `cluster_originalimagename.format`(e.g. `0_daniel1_0.jpg`, where `0` is the cluster label, and `daniel1_0` is the original image name, and `jpg` is the image format). 

## Additional Information
All of these scripts can be easily modified to use different face detection models, different clustering algorithms, etc. Simply open and modify the respective script. Thanks and enjoy!
