import cv2
import imutils
import json
import os

import numpy as np

input_path = "../data/test_images_site_32/"
output_path = "../data/test_images_site_32_annotated/"

category_colors = {'brick' : (0,0,255),
                   'concrete' : (255,0,0),
                   'metal' : (0,255,0),
                   'wood' : (0,255,255),
                   'z_none' : (0,0,0)}

# Reads the vott json file and draws just the annotated regions

with open(input_path + "vott-json-export/test_images_site_32-export.json") as f:
    json_file = json.load(f)
    for asset_id in (json_file['assets']):
        asset = json_file['assets'][asset_id]

        name = asset['asset']['name'].split('.')[0]
        w = asset['asset']['size']['width']
        h = asset['asset']['size']['height']

        scale = w / 1024

        w = int(w / scale)
        h = int(h / scale)

        img_out = np.zeros(shape=(h,w,3))
        cv2.rectangle(img_out,(0,0),(img_out.shape[1],img_out.shape[0]),(10,0,10),thickness = -1)

        for region in asset['regions']:
            tag = region['tags'][0]

            pts = []
            for point in region['points']:
                pts.append([int(point['x'] / scale),int(point['y'] / scale)])
            pts = np.array(pts)
                
            cv2.fillPoly(img_out,[pts],category_colors[tag])

        #img_out = imutils.resize(img_out,width=min(1024,img_out.shape[1]))

        cv2.imwrite(output_path + name + ".png",img_out)
