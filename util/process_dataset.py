import cv2
import json
import os

input_path = "../data/s2g1_dataset_raw/images/"
output_path = "../data/s2g1_dataset/images/"

categories = ['brick','concrete','metal','wood','z_none']

category_counts = {'brick':0,'concrete':0,'metal':0,'wood':0,'z_none':0}

for category in categories:
    print("")
    print("==={}===".format(category))
    folders = os.listdir(input_path + category)
    for folder in folders:
        files = os.listdir(input_path + category + "/" + folder)
        existing_files = os.listdir(output_path + category)
        for imgfile in files:

            category_counts[category] += 1
            
            outname = folder + "_" + imgfile
            
            # Dont process file if it already exists
            if outname in existing_files:
                continue
            
            img = cv2.imread(input_path + category + "/" + folder + "/" + imgfile)
            (h,w) = img.shape[:2]
            # Crop if necessary
            if w < h:
                diff = int(round((h - w) / 2,0))
                img=img[diff:h - diff,::]
            elif h < w:
                diff = int(round((w - h) / 2,0))
                img=img[::,diff:w - diff]

            (h2,w2) = img.shape[:2]
            print("{} - {} : {}x{}->{}x{}".format(folder,imgfile,w,h,w2,h2))

            img = cv2.resize(img,(362,362))

            cv2.imwrite(output_path + category + "/" + outname,img)
    print("{} images".format(category_counts[category]))

            
                

    
