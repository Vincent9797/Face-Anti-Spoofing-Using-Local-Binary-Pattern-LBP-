from skimage import feature
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

image = cv2.imread(r'fake\0000.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

NEIGHBORS = 24
RADIUS = 8
lbp = feature.local_binary_pattern(gray, NEIGHBORS, RADIUS, method="uniform")

(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, NEIGHBORS + 3), range=(0, NEIGHBORS + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

coef = np.array([-4.60885692e+01, -1.96856605e+01,  4.75445486e+01,  6.77233907e+00,
		   2.11002549e+00,  2.77746066e+01, -2.63103585e+01,  1.44045578e+01,
		   1.76653189e-01,  4.24613658e+01, -5.65977423e+01,  1.34514926e+01,
		  -7.12849353e+00, -5.55027897e-02, -1.46882564e+01,  5.38751501e+01,
		  -1.06494060e+01, -8.59358873e+00,  3.13511400e+01, -2.69635460e+01,
		  -1.44978052e+01, -4.88137345e-01, -3.46730967e+01,  2.39758837e+01,
		  -8.24209248e-01,  2.77339786e+00])
intercept = np.array([-0.57321123])

"""
matrix multiplication - hist has shape of 1x26, coef.T transposes coef, convrting it from 1x26 to 26x1
matrix multiplication produces a 1x1 array which is then added with intercept
any score below 0 is fake, above 0 is real, threshold can be changed
"""
print(hist @ coef.T + intercept)

filename = 'model_24,8_1000000.sav'
model = pickle.load(open(filename, 'rb'))
print('loaded')

image = cv2.imread(r'fake\0000.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

NEIGHBORS = 24
RADIUS = 8
lbp = feature.local_binary_pattern(gray, NEIGHBORS, RADIUS, method="uniform")

(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, NEIGHBORS + 3), range=(0, NEIGHBORS + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)
print(model.predict(hist))

