# USAGE
# python test_model.py --images test_images --codebook output/vocab.cpickle --model output/model.cpickle

# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.ir import BagOfVisualWords
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import pickle
import imutils
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="Path to input images directory")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the codebook")
ap.add_argument("-m", "--model", required=True,
	help="Path to the classifier")
ap.add_argument("-e", "--extractor",default="BRISK")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create(args["extractor"])
dad = DetectAndDescribe(detector, descriptor)
idf = None

#idf = pickle.loads(open("output/idf.cpickle","rb").read())

print("Testing using {} descriptor extractor".format(args["extractor"]))

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the classifier
model = pickle.loads(open(args["model"], "rb").read())

probabilities = []

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
    # load the image and prepare it from description
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=min(364, image.shape[1]))

    # describe the image and classify it
    (kps, descs) = dad.describe(gray)
    
    hist = bovw.describe(descs)
    hist /= hist.sum()

    # Absolutely critical to allow predict_proba
    hist = hist.toarray()

    # Be sure to shape properly to avoid wonkiness
    if idf is not None:
        idf = idf.reshape(1,-1)
        hist *= idf


    probs = model.predict_proba(hist)
    print(probs)
    probabilities.append(probs)
    prediction = model.predict(hist)[0]
    
    # show the prediction
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[PREDICTION] {}: {}".format(filename, prediction))

    image = imutils.resize(image,width = int(image.shape[1] / 2))
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(image,np.array2string(probs),(10,70),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    
    cv2.imwrite("output_detection/" + filename,image)
    #dcv2.imshow("Image", image)
    #cv2.waitKey(0)

with open("output_detection/detection_probs.txt",'w') as f:
    for prob in probabilities:
        f.write(np.array2string(prob) + "\n")
