from skimage import feature
import numpy as np
import cv2
import os
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import pickle


class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
        	self.numPoints = numPoints
        	self.radius = radius

	def describe(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
		# print(hist, len(hist), sum(hist), lbp.shape[0]*lbp.shape[1])
		hist = hist.astype("float")
		hist /= (hist.sum() + eps) # values range from 0 to 1
		return hist

images = []
for folder in ['fake', 'real']:
	for image in os.listdir(folder):
		images.append(os.path.join(folder, image))

train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

random.shuffle(train_images)
random.shuffle(val_images)

# counting classes
# from collections import Counter
# train_labels = [image.split('\\')[0] for image in train_images]
# val_labels = [image.split('\\')[0] for image in val_images]
# print(Counter(train_labels))
# print(Counter(val_labels))

desc = LocalBinaryPatterns(24, 8)

# pre-processing and then saving it. no need to generate again
data = []
labels = []
for image_path in tqdm(train_images):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	labels.append(image_path.split('\\')[0])
	data.append(hist)

with open("X_train_24,8.txt", "wb") as fp:
	pickle.dump(data, fp)

with open("Y_train_24,8.txt", "wb") as fp:
	pickle.dump(labels, fp)

data = []
labels = []
for image_path in tqdm(val_images):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	labels.append(image_path.split('\\')[0])
	data.append(hist)

with open("X_test_24,9.txt", "wb") as fp:
	pickle.dump(data, fp)

with open("Y_test_24,8.txt", "wb") as fp:
	pickle.dump(labels, fp)

# # training from saved X and Y
# with open("X_train_9,3.txt", "rb") as fp:   # Unpickling
# 	X_train = pickle.load(fp)
#
# with open("Y_train_9,3.txt", "rb") as fp:   # Unpickling
# 	Y_train = pickle.load(fp)

model = LinearSVC(C=100.0, random_state=42, max_iter=1000000, verbose=1)
print("Training")
model.fit(X_train, Y_train)

print(model.coef_)
print(model.intercept_)

filename = 'model_24,8_1000000.sav'
pickle.dump(model, open(filename, 'wb'))
del model

# loading model
# model = pickle.load(open(filename, 'rb'))
# print('loaded')
