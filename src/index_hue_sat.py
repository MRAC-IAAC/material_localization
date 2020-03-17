from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.indexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import random
import cv2
import h5py

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images to be indexed")
ap.add_argument("-b", "--hs-db", required=True,
	help="Path to where the hue database will be stored")
args = vars(ap.parse_args())


imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

categories = ["brick","concrete","metal","wood","z_none"]

db_hs = h5py.File(args['hs_db'],mode='w')
hue_set = db_hs.create_dataset("hue",(16 * len(categories),1),dtype='float')
sat_set = db_hs.create_dataset("sat",(16 * len(categories),1),dtype='float')

total_hue = {}
total_sat = {}
category_totals = {}

for c in categories:
    total_hue[c] = np.reshape(np.zeros(16),(16,1))
    total_sat[c] = np.reshape(np.zeros(16),(16,1))
    category_totals[c] = 0

for (i, imagePath) in enumerate(imagePaths):

    # Load and Reize Image
    p = imagePath.split("/")
    category = p[-2]
    imageID = "{}:{}".format(p[-2], p[-1])
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(364, image.shape[1]))
    category_totals[category] += 1

    # Get image hsv channels
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    (h,s,v) = cv2.split(hsv_image)

    # Turn image into 1d array
    h_flat = h.reshape(1,h.shape[0] * h.shape[1])
    s_flat = s.reshape(1,s.shape[0] * s.shape[1])
    
    # Calculate Histograms
    hist_hue = cv2.calcHist(h_flat,[0],None,[16],[0,180])
    hist_sat = cv2.calcHist(s_flat,[0],None,[16],[0,256])
    
    # Normalize Histograms
    hist_hue /= hist_hue.sum()
    hist_sat /= hist_sat.sum()

    # Add to histogram totals
    total_hue[category] += hist_hue
    total_sat[category] += hist_sat

for i,c in enumerate(categories):
    # Find average histograms
    total_hue[c] /= category_totals[c]
    total_sat[c] /= category_totals[c]

    # Put average histograms in database
    hue_set[i * 16 : i * 16 + 16] = total_hue[c]
    sat_set[i * 16 : i * 16 + 16] =  total_sat[c]
    
db_hs.close()
