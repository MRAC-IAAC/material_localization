# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from imutils import paths

import argparse
import cv2
import os
import pickle
import progressbar
import random
import sklearn

import numpy as np

# handle sklearn versions less than 0.18
if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.grid_search import GridSearchCV

# otherwise, sklearn.grid_search is deprecated
# and we'll import GridSearchCV from sklearn.model_selection
else:
	from sklearn.model_selection import GridSearchCV


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to the training images")
args = vars(ap.parse_args())

category_ids = {'brick' : 0,
                'concrete' : 1,
                'metal' : 2,
                'wood' : 3,
                'z_none' : 4}

# initialize the local binary patterns descriptor along with
# the data and label lists
#desc = LocalBinaryPatterns(24, 8) # 0.64
#desc = LocalBinaryPatterns(24, 16) # 0.55
#desc = LocalBinaryPatterns(48, 16) # 0.55
#desc = LocalBinaryPatterns(24, 32) # 0.58
desc4 = LocalBinaryPatterns(24,4) # 0.58
desc8 = LocalBinaryPatterns(24,8) # 0.58

data4 = []
data8 = []

labels = []

# loop over the training images
image_paths = list(paths.list_images(args["input"]))
random.shuffle(image_paths)

split_point = int(len(image_paths) / 5 * 3)

training_set = image_paths[:split_point]
testing_set = image_paths[split_point:]

#training_set = image_paths[::]
#testing_set = image_paths[::]


print("Found {} training images".format(len(training_set)))
print("Found {} testing images".format(len(testing_set)))

widgets = ["Analyzing Images: ",progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(training_set),widgets=widgets).start()

for i,image_path in enumerate(training_set):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hist4 = desc4.describe(gray)
    hist4 /= hist4.sum()
    data4.append(hist4)

    hist8 = desc8.describe(gray)
    hist8 /= hist8.sum()
    data8.append(hist8)

    label = image_path.split(os.path.sep)[-2]
    labels.append(label)


    pbar.update(i)

pbar.finish()

# train a Linear SVM on the data
print("Fitting Model")

#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler.fit(data)
#data = scaler.transform(data)

params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
#model = GridSearchCV(LinearSVC(random_state=42,max_iter=30000),params,cv=3)
model4 = GridSearchCV(SVC(random_state=42,probability = True,max_iter=50000),params,cv=3)
model8 = GridSearchCV(SVC(random_state=42,probability = True,max_iter=50000),params,cv=3)

model4.fit(data4, labels)
model8.fit(data8, labels)

# loop over the testing images
test_labels = []
test_data4 = []
test_data8 = []
for i,image_path in enumerate(testing_set):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hist4 = desc4.describe(gray)
    hist4 /= hist4.sum()

    hist8 = desc8.describe(gray)
    hist8 /= hist8.sum()
    
    label = image_path.split(os.path.sep)[-2]

    test_data4.append(hist4)
    test_data8.append(hist8)
    test_labels.append(label)

#predictions4 = model.predict(test_data)
predictions_proba_4 = model4.predict_proba(test_data4)
predictions_proba_8 = model8.predict_proba(test_data8)

predictions_combined = predictions_proba_4 + predictions_proba_8
predictions = np.argmax(predictions_combined,axis = 1)

#print(predictions_proba)
test_labels = [category_ids[c] for c in test_labels]

test_labels = np.asarray(test_labels)

print(classification_report(test_labels, predictions))

f = open('model/model_lbp4.cpickle', "wb")
f.write(pickle.dumps(model4))
f.close()

f = open('model/model_lbp8.cpickle', "wb")
f.write(pickle.dumps(model8))
f.close()
