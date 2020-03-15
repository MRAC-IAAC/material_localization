from __future__ import print_function

from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.ir import BagOfVisualWords
from pyimagesearch.object_detection.helpers import sliding_window_double
from pyimagesearch.object_detection.helpers import pyramid

from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths

import argparse
import h5py
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

# Load inverse document frequency file
#idf = pickle.loads(open("output/idf.cpickle","rb").read())

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the classifier
model = pickle.loads(open(args["model"], "rb").read())

# Load hue and saturation data
db_hs = h5py.File('output/hs-db.hdf5',mode='r')
hue_set = db_hs['hue']
sat_set = db_hs['sat']

category_colors = ((0,0,255),(255,0,0),(0,255,0),(0,255,255))

# loop over the image paths
image_paths = paths.list_image(args["images"])
for img_id,imagePath in enumerate(image_paths):
    print("{}/{} : {}",img_id.zfill(3),len(image_paths).zfill(3),image_path)
    name = imagePath.split("/")[-1].split(".")[0]
    # load the image and prepare it from description
    img_main = cv2.imread(imagePath)
    
    img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
    img_gray = imutils.resize(img_gray, width=min(1024, img_main.shape[1]))

    img_hsv = cv2.cvtColor(img_main,cv2.COLOR_BGR2HSV)
    img_hsv = imutils.resize(img_hsv,width = min(1024,img_main.shape[1]))

    
    display_img = img_main.copy()
    display_img = imutils.resize(display_img, width=min(1024, img_main.shape[1]))

    squares_img = np.zeros(display_img.shape,np.uint8)

    prediction_list = []

    win_size = 50
    for (x,y,window_gray,window_hsv) in sliding_window_double(img_gray,img_hsv,stepSize=100,windowSize=(100,100)):
        # Ensure patch size
        window_gray = imutils.resize(window_gray,width = 364)
        window_hsv = imutils.resize(window_hsv,width=364)

        # Describe gray patch
        (kps,descs) = dad.describe(window_gray)
        if kps is None or descs is None:
            continue
        hist = bovw.describe(descs)
        hist /= hist.sum()
        hist = hist.toarray()

        # Predict based on gray patch
        predictions = model.predict_proba(hist)
        #print(predictions)
        prediction = predictions[0]

        # Extract hue and sat histograms
        (h,s,v) = cv2.split(window_hsv)
        hist_hue = cv2.calcHist(h,[0],None,[16],[0,180])
        hist_sat = cv2.calcHist(s,[0],None,[16],[0,256])
        hist_hue /= hist_hue.sum()
        hist_sat /= hist_sat.sum()

        # Get hue and sat diff values
        hue_diffs = np.zeros(4)
        sat_diffs = np.zeros(4)
        for i in range(4):
            hue_slice = hue_set[i * 16 : i * 16 + 16]
            sat_slice = sat_set[i * 16 : i * 16 + 16]

            hue_diffs[i] = np.abs(hue_slice - hist_hue).sum()
            sat_diffs[i] = np.abs(sat_slice - hist_sat).sum()

        #print(hue_diffs)

        predictions -= hue_diffs * 0.5
        predictions -= sat_diffs * 2.0

        prediction_list.append(predictions)

        id = np.argmax(prediction)
        cv2.rectangle(squares_img,(x,y),(x + 100,y + 100),category_colors[id],-1)

        #print(predictions)
        #if (prediction[id] > 0.5):
        #cv2.rectangle(squares_img,(x,y),(x + 100,y + 100),category_colors[id],-1)
            #cv2.circle(display_img, (x + 25,y + 25),2,category_colors[id],-1)

    cv2.addWeighted(squares_img,0.5,display_img,0.5,0,display_img)
            
    (kps,descs) = dad.describe(img_gray)
    hist = bovw.describe(descs)
    hist /= hist.sum()
    hist = hist.toarray()
    #print(model.predict(hist)[0])

    #cv2.imshow("window",display_img)
    cv2.imwrite("output_localization/" + imagePath.split("/")[-1],display_img)
    with open("output_localization/" + name + ".txt",'w') as f:
        for line in prediction_list:
            f.write(np.array2string(line) + "\n")
    
    #cv2.waitKey(0)
