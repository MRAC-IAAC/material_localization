from __future__ import print_function

from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.ir import BagOfVisualWords
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.object_detection.helpers import sliding_window_double,sliding_window
from pyimagesearch.object_detection.helpers import pyramid

from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths

import argparse
import cv2
import h5py
import imutils
import os
import pickle
import progressbar
import time
import scipy

import numpy as np

# Initial results:
# SF : 1.81
# HF : 0.18


# === GLOBALS === 

# Offsets for applying detection windows to subpatches
kernel_size = 4
dx = list(range(kernel_size)) * kernel_size
dy = []
for i in range(kernel_size):
    for j in range(kernel_size):
        dy.append(i)

win_size = 100
patch_size = win_size // kernel_size


# Load hue and saturation data (note we immediately convert to a normal python array)
# (probably these shouldn't be h5 dbs after all, cpickle would be better)
db_hs = h5py.File('model/hs-db.hdf5',mode='r')
hue_set = db_hs['hue'][::]
sat_set = db_hs['sat'][::]
sat_total_set = db_hs['sat_total'][::]

# The None category is a very dark magenta
category_colors = ((0,0,255),(255,0,0),(0,255,0),(0,255,255),(10,0,10))

category_hue_backwards = {0   : 0,
                          240 : 1,
                          120 : 2,
                          60  : 3,
                          300 : 4}

model_lbp4 = pickle.loads(open('output/model_lbp4.cpickle', "rb").read())
desc4 = LocalBinaryPatterns(24,4) # 0.58
desc8 = LocalBinaryPatterns(24,8) # 0.58
                          
# === DEFINITIONS ===

def test_args_hs(sat_factor,hue_factor):
    score = 0
    for img_id,asset_name in enumerate(asset_names):
        for patch_id in range(annotations.shape[1]):
            prediction = probs_raw[img_id,patch_id]

            hist_hue = hue_histograms[img_id,patch_id]
            hist_sat = sat_histograms[img_id,patch_id]

            hue_diffs = np.zeros(5)
            sat_diffs = np.zeros(5)
            for i in range(5):
                hist_hue_avg = hue_set[i * 16 : i * 16 + 16]
                hist_sat_avg = sat_set[i * 16 : i * 16 + 16]

                hue_diffs[i] = np.abs(hist_hue_avg - hist_hue).sum()
                sat_diffs[i] = np.abs(hist_sat_avg - hist_sat).sum()

            prediction -= sat_diffs * sat_factor
            for i in range(5):
                prediction[i] -= hue_diffs[i] * hue_factor * sat_total_set[i]

            prediction_id = np.argmax(prediction)
            
            guess_cat = annotations[img_id,patch_id]

            if prediction_id == guess_cat:
                score += 1
    return(score)

def test_args_lbp(lbp_factor):
    score = 0
    for img_id,asset_name in enumerate(asset_names):
        for patch_id in range(annotations.shape[1]):
            lbp_proba = lbp_probas[img_id,patch_id]
            prediction = probs_raw[img_id,patch_id].reshape(1,-1)
            
            prediction += lbp_proba * lbp_factor
            prediction_id = np.argmax(prediction)
            
            guess_cat = annotations[img_id,patch_id]

            if prediction_id == guess_cat:
                score += 1
    return(score)

def run_hs_tests():
    s_steps = 10
    h_steps = 10

    best_score = -1
    best_sf = -1
    best_hf = -1

    widgets = ["Testing factors : ",
               progressbar.Percentage(), " ",
               progressbar.Bar(), " ",
               progressbar.ETA(), " "]
    pbar = progressbar.ProgressBar(maxval = s_steps * h_steps,widgets=widgets).start()

    for sid,sf_try in enumerate(np.linspace(4.7,4.9,s_steps)):
        for hid,hf_try in enumerate(np.linspace(0.3,0.5,h_steps)):

            score = test_args_hs(sf_try,hf_try)
            if (score > best_score):
                best_score = score
                best_sf = sf_try
                best_hf = hf_try
            #print("Score : {},{} ({},{}) : {} ".format(sid,hid,sf_try,hf_try,score))
            scores_hs[sid,hid] = score
            pbar.update(sid * h_steps + hid)
            #pbar.update(sid * 30 + hid,sf=sf_try,hf=hf_try,score=score)

    pbar.finish()
    
    best = np.argmax(scores_hs)

    print(best_score)
    print(best_sf)
    print(best_hf)
    print(best)

def run_lbp_tests():
    steps = 10

    best_factor = -1
    best_score = -1
    
    widgets = ["Testing factors : ",
               progressbar.Percentage(), " ",
               progressbar.Bar(), " ",
               progressbar.ETA(), " "]
    pbar = progressbar.ProgressBar(maxval = steps,widgets=widgets).start()

    for fid,f_try in enumerate(np.linspace(0,5,steps)):
        score = test_args_lbp(f_try)
        if (score > best_score):
            best_score = score
            best_factor = f_try
        print("Score : {} ({}) : {}".format(fid,f_try,score))
        scores_lbp[fid] = score
        pbar.update(fid)
    
    pbar.finish

    best = np.argmax(scores_lbp)

    print(best_score)
    print(best_factor)
    print(best)


# === MAIN SCRIPT ===
scores_hs = np.zeros(shape=(30,30))
scores_lbp = np.zeros(10)

asset_names = []
image_paths = []

# Find all asset names, given annotated assets
for f in os.listdir("data/test_images_site_32_annotated"):
    name = f.split('/')[-1].split('.')[0]
    asset_names.append(name)
    image_paths.append(name + ".JPG")
image_paths.sort()
asset_names.sort()

# Load raw probabilities from text files
probs_raw = np.zeros(shape=(len(asset_names),1148,5))
for asset_id,asset_name in enumerate(asset_names):
    f = open('output/localization_raw/' + asset_name + '.txt')
    lines = f.readlines()
    for patch_id in range(len(lines)):
        prediction = lines[patch_id][1:-2].split(' ')
        prediction = [float(s) for s in prediction if s != '']
        probs_raw[asset_id,patch_id] = prediction



annotations = np.zeros(shape=(len(image_paths),1148))
hue_histograms = np.zeros(shape=(len(image_paths),1148,16))
sat_histograms = np.zeros(shape=(len(image_paths),1148,16))
lbp_probas = np.zeros(shape=(len(image_paths),1148,5))


for img_id,path in  enumerate(image_paths):
    ann_image = cv2.imread("data/test_images_site_32_annotated/" + (path.split(".")[0] + ".png"))
    (ann_h,ann_s,ann_v) = cv2.split(cv2.cvtColor(ann_image,cv2.COLOR_BGR2HSV))

    img_width = ann_image.shape[1]
    img_height = ann_image.shape[0]

    patch_width = (img_width // patch_size) + 1
    patch_height = (img_height // patch_size) + 1
    patch_length = patch_width * patch_height

    img_name = path.split('.')[0] + ".JPG"
    img = cv2.imread("data/test_images_site_32/" + img_name)
    img = imutils.resize(img,width=1024)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (img_h,img_s,img_v) = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2HSV))

    # Store gray patch descriptions
    for (patch_id, (x,y,win)) in enumerate(sliding_window(img_gray,stepSize = patch_size,windowSize = (win_size,win_size))):

        hist4  = desc4.describe(win)
        hist4 /= hist4.sum()
        hist4 = hist4.reshape(1,-1)
        proba_4 = model_lbp.predict_proba(hist4)

        lbp_probas[img_id,patch_id] = proba_4
        


    # Precompute h and s histograms
    for (patch_id,(x,y,win_h,win_s)) in enumerate(sliding_window_double(img_h,img_s,stepSize = patch_size,windowSize = (win_size,win_size))):
        
        h_flat = win_h.reshape(1,win_h.shape[0] * win_h.shape[1])
        s_flat = win_s.reshape(1,win_s.shape[0] * win_s.shape[1])

        hist_hue = cv2.calcHist(h_flat,[0],None,[16],[0,180])
        hist_sat = cv2.calcHist(s_flat,[0],None,[16],[0,180])

        if hist_hue.sum() > 0:
            hist_hue /= hist_hue.sum()
        if hist_sat.sum() > 0:
            hist_sat /= hist_sat.sum()

        hist_hue = hist_hue.reshape(16)
        hist_sat = hist_sat.reshape(16)
        hue_histograms[img_id,patch_id] = hist_hue
        sat_histograms[img_id,patch_id] = hist_sat
        
    # Find best match for each patch in annotated images
    for (patch_id,(x,y,win_h)) in enumerate(sliding_window(ann_h,stepSize=patch_size,windowSize=(win_size,win_size))):
        mode = scipy.stats.mode(win_h.flatten())[0][0]
        annotations[img_id][patch_id] = int(category_hue_backwards[2 * int(mode)])

            
#run_hs_tests()
run_lbp_tests()




    
        
    
