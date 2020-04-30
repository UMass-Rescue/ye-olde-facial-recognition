# Background Extraction
This project allows users to extract backgrounds from portrait-like images, blacking out people in the images.

## Requirements
Install [Anaconda](https://www.anaconda.com/distribution/) and set up a Conda virtual environment as follows:
 1. Create a new virtual environment: `conda create -n background_extraction python=3.7`
 2. Activate the environment: `conda activate background_extraction`
 3. Install pip: `conda install pip`
 4. Install PyTorch: `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
 5. Install OpenCV: `conda install -c conda-forge opencv`

## Step 1: Background Extraction

To extract backgrounds from a directory of images, the `segment_background` function can be called from the `test_segmentation.py` script. The extracted backgrounds will be saved as RGB images to a new folder called `extracted_backgrounds` in the same directory as the test images.

This function requires two parameters: 

 1. Path to the image directory as a string
 2. Path to the weights file for the DSFD face detector as a string. The weights file can be downloaded from the [original Human-Segmentation-PyTorch repository](https://github.com/thuyngch/Human-Segmentation-PyTorch). The default directory for storing the weights file is simply the base directory containing all the scripts.

## Additional Information
Much of this repository was based off of the original human segmentation repository located [here](https://github.com/thuyngch/Human-Segmentation-PyTorch). Huge thanks to Thuy Ng for his work!

All of these scripts can be easily modified; simply open and modify the respective script. Thanks and enjoy!
