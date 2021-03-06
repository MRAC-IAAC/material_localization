from __future__ import print_function

from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.ir import BagOfVisualWords
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.object_detection.helpers import sliding_window_double
from pyimagesearch.object_detection.helpers import pyramid

from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths

import argparse
import cv2
import h5py
import imutils
import pickle
import progressbar
import time

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

# How many classification patches a single window covers on a side
kernel_size = 4
# Offsets for applying detection windows to subpatches
dx = list(range(kernel_size)) * kernel_size
dy = []
for i in range(kernel_size):
    for j in range(kernel_size):
        dy.append(i)

win_size = 100
patch_size = win_size // kernel_size
print(win_size)
print(patch_size)

# Initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create(args["extractor"])
dad = DetectAndDescribe(detector, descriptor)

# Load inverse document frequency file
idf = pickle.loads(open("model/idf.cpickle","rb").read())

# Load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# Load the bovw classifier
model = pickle.loads(open(args["model"], "rb").read())

# Load hue and saturation data
db_hs = h5py.File('model/hs-db.hdf5',mode='r')
hue_set = db_hs['hue'][::]
sat_set = db_hs['sat'][::]
sat_total_set = db_hs['sat_total'][::]

# Load lbp models
model_lbp4 = pickle.loads(open('model/model_lbp4.cpickle', "rb").read())
model_lbp8 = pickle.loads(open('model/model_lbp8.cpickle', "rb").read())
desc4 = LocalBinaryPatterns(24,4)
desc8 = LocalBinaryPatterns(24,8)

# The None category is a very dark magenta
#category_colors = ((0,0,255),(255,0,0),(0,255,0),(0,255,255),(10,0,10))
category_colors = ((0,0,255),(255,0,0),(0,255,0),(0,255,255),(0,0,0))

category_names = ("brick","concrete","metal","wood","z_none")

# Whether to scale the input image down to 1024 width
flag_resize_image = True

# Whether to resize all patches to 364x364. Slows things down immensely, but may be necessary for accuracy?
flag_resize_patch = False

# Whether to show each image to the user as its localized
flag_display = False

# Whether to save all the classified subpatches by category
flag_export_patches = True


# === MAIN SCRIPT === 

start_time = time.time()

# loop over the image paths
image_paths = paths.list_images(args["images"])
for img_id,imagePath in enumerate(image_paths):
    print("{} : {}".format(str(img_id).zfill(3),imagePath))
    name = imagePath.split("/")[-1].split(".")[0]
    
    # img_main is the original, full size image
    img_main = cv2.imread(imagePath)
    if flag_resize_image:
        img_main = imutils.resize(img_main,width=min(1024,img_main.shape[1]))

    # img_gray is resized to 1024 width and in grayscale
    img_gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
    #img_gray = imutils.resize(img_gray, width=min(1024, img_main.shape[1]))

    # img_hsv is resized to 1024 width and hsv space
    img_hsv = cv2.cvtColor(img_main,cv2.COLOR_BGR2HSV)
    #img_hsv = imutils.resize(img_hsv,width = min(1024,img_main.shape[1]))

    #img_display is resized to 1024 width, to be annotated with patch colors
    img_display = img_main.copy()
    #img_display = imutils.resize(img_display, width=min(1024, img_main.shape[1]))

    #img_squares is the size of img_display, and contains the category color squares to draw on top of it
    img_squares = np.zeros(img_display.shape,np.uint8)

    prediction_list_weighted = []
    prediction_list_raw = []

    img_width = img_gray.shape[1]
    img_height = img_gray.shape[0]

    patch_width = (img_width // patch_size) + 1
    patch_height = (img_height // patch_size) + 1
    patch_length = patch_width * patch_height

    widgets = ["Localizing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval = patch_length,widgets=widgets).start()

    # All score totals including h/s calculations
    patch_totals_weighted = np.zeros(shape=(patch_width,patch_height,5))

    # Score totals only using bovw scores
    patch_totals_raw = np.zeros(shape=(patch_width,patch_height,5))

    for (patch_id,(x,y,window_gray,window_hsv)) in enumerate(sliding_window_double(img_gray,img_hsv,stepSize=patch_size,windowSize=(win_size,win_size))):
        # Find x and y position in the patch grid
        patch_x = patch_id % patch_width
        patch_y = patch_id // patch_width

        # Ensure patch size
        if flag_resize_patch:
            window_gray = imutils.resize(window_gray,width = 100)
            window_hsv = imutils.resize(window_hsv,width=100)

        # Describe gray patch
        (kps,descs) = dad.describe(window_gray)
        if kps is None or descs is None:
            continue
        hist = bovw.describe(descs)
        hist /= hist.sum()
        hist = hist.toarray()

        # Get lbp descriptions
        hist4 = desc4.describe(window_gray)
        hist8 = desc8.describe(window_gray)

        hist4 /= hist4.sum()
        hist8 /= hist8.sum()

        hist4 = hist4.reshape(1,-1)
        hist8 = hist8.reshape(1,-1)

        proba_4 = model_lbp4.predict_proba(hist4).flatten()
        proba_8 = model_lbp8.predict_proba(hist8).flatten()

        # Apply idf factor to histogram
        if idf is not None:
            idf = idf.reshape(1,-1)
            hist *= idf

        # Get prediction probabilities based on gray patch
        prediction_raw = model.predict_proba(hist)[0]
        prediction_weighted = np.copy(prediction_raw)

        # Extract hue and sat histograms
        (h,s,v) = cv2.split(window_hsv)

        # Turn image into 1d array
        h_flat = h.reshape(1,h.shape[0] * h.shape[1])
        s_flat = s.reshape(1,s.shape[0] * s.shape[1])

        # Calculate histograms
        hist_hue = cv2.calcHist(h_flat,[0],None,[16],[0,180])
        hist_sat = cv2.calcHist(s_flat,[0],None,[16],[0,256])

        # Normalize Histograms
        hist_hue /= hist_hue.sum()
        hist_sat /= hist_sat.sum()

        # Get hue and sat diff values
        hue_diffs = np.zeros(5)
        sat_diffs = np.zeros(5)
        for i in range(5):
            
            hist_hue_avg = hue_set[i * 16 : i * 16 + 16]
            hist_sat_avg = sat_set[i * 16 : i * 16 + 16]

            hue_diffs[i] = np.abs(hist_hue_avg - hist_hue).sum()
            sat_diffs[i] = np.abs(hist_sat_avg - hist_sat).sum()

        # Weight the predictions using hue and sat diffs
        sat_factor = 1.9
        hue_factor = 0.14

        # Weight prediction with hue/sat
        prediction_weighted -= sat_diffs * sat_factor
        for i in range(5):
            prediction_weighted[i] -= hue_diffs[i] * hue_factor * sat_total_set[i]

        # Weight prediction with LBP
        prediction_weighted += proba_4
        prediction_weighted += proba_8


        # Apply window predictions to patches
        for i in range(kernel_size * kernel_size):
            nx = patch_x + dx[i]
            ny = patch_y + dy[i]
            if nx >= patch_width or ny >= patch_height:
                continue
            patch_totals_weighted[nx,ny] += prediction_weighted
            patch_totals_raw[nx,ny] += prediction_raw

        pbar.update(patch_id)
        
    pbar.finish()
    
    # Loop through each patch, draw the prediction color and save to text file
    for i in range(patch_length):
        patch_x = i % patch_width
        patch_y = i // patch_width
        pixel_x = patch_x * patch_size
        pixel_y = patch_y * patch_size
        id = np.argmax(patch_totals_weighted[patch_x,patch_y])
        prediction_list_weighted.append(patch_totals_weighted[patch_x,patch_y])
        prediction_list_raw.append(patch_totals_raw[patch_x,patch_y])
        cv2.rectangle(img_squares,(pixel_x,pixel_y),(pixel_x + win_size,pixel_y + win_size),category_colors[id],-1)

        if flag_export_patches and id != 4:
            output_name = "output/subpatches/{}/{}_{}.png".format(category_names[id],name,i)
            subpatch = img_main[pixel_x:pixel_x + patch_size,pixel_y :pixel_y + patch_size]

            if subpatch.shape[0] == patch_size and subpatch.shape[1] == patch_size:
                cv2.imwrite(output_name,subpatch)
            

    # Draw the category colors onto the image
    #cv2.addWeighted(img_squares,0.5,img_display,0.5,0,img_display)

    img_squares_hsv = cv2.cvtColor(img_squares,cv2.COLOR_BGR2HSV)
    (sh,ss,sv) = cv2.split(img_squares_hsv)

    (imh,ims,imv) = cv2.split(img_hsv)

    img_display = cv2.merge((sh,ss,imv))
    img_display = cv2.cvtColor(img_display,cv2.COLOR_HSV2BGR)

    # Display the image to the user
    if flag_display:
        cv2.imshow("Localization",img_display)
        cv2.waitKey(0)

    # Write the images and text file output
    cv2.imwrite("output/localization/" + name + ".png",img_display)
    cv2.imwrite("output/localization_annotations/" + name + '.png',img_squares)
    with open("output/localization_probs_weighted/" + name + ".txt",'w') as f:
        for line in prediction_list_weighted:
            f.write(np.array2string(line) + "\n")
    with open("output/localization_probs_raw/" + name + ".txt",'w') as f:
        for line in prediction_list_raw:
            f.write(np.array2string(line) + "\n")
    
end_time = time.time()


print("Localization of {} images took {} seconds".format(img_id + 1,int(end_time - start_time)))
