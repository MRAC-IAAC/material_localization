# USAGE
# python train_model.py --dataset caltech5 --features-db output/features.hdf5 --bovw-db output/bovw.hdf5 \
#	--model output/model.cpickle

# import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import numpy as np
import argparse
import pickle
import h5py
import cv2

# handle sklearn versions less than 0.18
if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.grid_search import GridSearchCV

# otherwise, sklearn.grid_search is deprecated
# and we'll import GridSearchCV from sklearn.model_selection
else:
	from sklearn.model_selection import GridSearchCV

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the original images")
ap.add_argument("-f", "--features-db", required=True,
	help="Path the features database")
ap.add_argument("-b", "--bovw-db", required=True,
	help="Path to where the bag-of-visual-words database")
ap.add_argument("-m", "--model", required=True,
	help="Path to the output classifier")
args = vars(ap.parse_args())

# open the features and bag-of-visual-words databases
featuresDB = h5py.File(args["features_db"],'r')
bovwDB = h5py.File(args["bovw_db"],'r')

# Divide image with 3/5 for training and 2/5 for testing
split_point = int(featuresDB["image_ids"].shape[0] / 5 * 3)

# grab the training and testing data from the dataset using the first 300
# images as training and the remaining 200 images for testing
print("[INFO] loading data...")
(trainData, trainLabels) = (bovwDB["bovw"][:split_point], featuresDB["image_ids"][:split_point])
(testData, testLabels) = (bovwDB["bovw"][split_point:], featuresDB["image_ids"][split_point:])



# prepare the labels by removing the filename from the image ID, leaving
# us with just the class name
trainLabels = [l.split(":")[0] for l in trainLabels]
testLabels = [l.split(":")[0] for l in testLabels]

# define the grid of parameters to explore, then start the grid search where
# we evaluate a Linear SVM for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}

# Trying SVC instead of LinearSVC, to see what predict_proba does
#model = GridSearchCV(LinearSVC(random_state=42), params, cv=3)
model = GridSearchCV(SVC(random_state=42,probability = True), params, cv=3)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# show a classification report
print("[INFO] evaluating...")
predictions = model.predict(testData)
#print(type(testData))
#print(testData.shape)
#print(testData)


print(classification_report(testLabels, predictions))

# loop over a sample of the testing data
for id,i in enumerate(np.random.choice(np.arange(300, 500), size=(20,), replace=False)):
    # randomly grab a testing image, load it, and classify it
    (label, filename) = featuresDB["image_ids"][i].split(":")
    image = cv2.imread("{}/{}/{}".format(args["dataset"], label, filename))
    prediction = model.predict(bovwDB["bovw"][i].reshape(1, -1))[0]
    #print(model.predict_proba(bovwDB["bovw"][i].reshape(1,-1)).shape)
    # show the prediction
    print("[PREDICTION] {}: {}".format(filename, prediction))
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    #cv2.imshow("Image", image)
    #cv2.imwrite("image{}.jpg".format(id),image)
    #cv2.waitKey(0)
    
# close the databases
featuresDB.close()
bovwDB.close()

# dump the classifier to file
print("[INFO] dumping classifier to file...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()
