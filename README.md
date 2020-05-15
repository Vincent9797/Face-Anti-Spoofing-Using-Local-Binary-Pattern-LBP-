## Face Anti-Spoofing Using Local Binary Pattern (LBP)

Face spoofing is an attempt to acquire someone else’s access rights by using a photo, video for an authorized person’s face.

This repository will be split into 3 parts:

 1. Setup
 2. Training
 3. Inference

## 1. Setup
Organise the real images and images used for spoofing attacks into folders "real" and "fake" respectively. The data I have used was taken from: [https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/](https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/).

Clone the repository and run the following command to ensure all necessary packages are available: `pip install -r requirements.txt`

## 2. Training
Run `lbp.py` to start the training the process. The current method of training loads all the images at once and generates the LBP values. This method is not scalable and is only feasible up to ~30K images. Hence, the values are saved so that this time-consuming process don't have to be repeateed. For more images, you might have to load it in batches and then train the model.

## 3.Inference
I have included 2 ways to carry out inference seen in `inference.py`. 

 1. The first way is by extracting the learnt parameters from the model and then carrying out the operations without using any libraries
 2. The second way is to load the pickled model and call the `.predict` method
