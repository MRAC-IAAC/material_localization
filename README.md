Material Detection and Localization
===============================================

Performs material localization in images taken from demolition site drone inspection.

The system performs a series of classifications on a sliding window of the photograph, then averages these predictions with some additional heuristics to produce a final classification for each subpatch.   
Patch classification is done primarily using a BOVW algorithm operating on feature descripttions, with heuristics from a comparison of hue and saturation histograms.

Feature Detection uses the GFTT corner detection algorithm

Descriptor Extraction uses the BRISK binary descriptor algorithm.

The primary heuristic adds a simple Local Binary Pattern (LBP) analysis of the patch window, to reduce errors accumulated from non-optimal chosen feature or cluster counts.  
Additional heuristics perform a simple histogram comparison of hue and saturation levels in the patch window compared to the averages for each material. The strength of the hue heuristic is also weighted based on the average saturation strength (i.e. hue will affect the brick category more than the concrete category)


To Use
------

### Environment

This system was tested under Ubuntu 18.04, with Python 3.8, primarily using OpenCV 4.2 and Numpy 1.18  
Full python enironment is in requirements.txt

Some util scripts also require Processing 3. 



### Input Image Processing

The script at util/process_dataset will scan the directories in data/s2g1_dataset_raw, and rename, crop, and resize images as necessary to form an input set for training. It is recomended to add new images to the raw directory and let the script process them rather than processing them by hand.

The script expects the raw directory to contain one directory for each category, with subdirectories therein describing the images source. For instance a photograph the user has taken of a brick wall should be file roughly as so:  
`dataset_raw/brick/photo/IMG_001.jpg`

### Setup

To process the input images and train the classifier, run

`./setup_model`

The script will look for input images in the path data/s2g1_dataset/images, organized in directories by category name.

This may take ~30 minutes to run, mostly due to sampling the database during the feature clustering step.

This will produce several files in the 'model/' folder: 

- **hs-db.hdf5** : Averaged hue and saturation histograms for each category of the input dataset
- **features.hdf5** : Raw list of all features detected in input set
- **bovw.hdf5** : Results of clusterization of image features
- **vocab.cpickle** : Association of visual words with categories
- **model.cpickle** : Classification model trained from above results
- **idf.cpickle** : Inverse Document Frequency information, contains weighting information for the importance of different features. 

### Localization

To perform material localization, run 

`./localize`
 
Ensure the '--images' argument in the call to localize.py is properly set for the folder of images to process. 

The time to process each image depends on the kernel_size and thus patch_size variables in the localization script. Each classification window is always 100 pixels, so the patch size is determined by dividing 100 by the kernel size. (e.g. the default kernel_size is 4, which leads to a final resolution of classification patches 25px on a side)

This will write images with category colors overlayed to the 'output/localization' folder, as well as text files containing the heuristically weighted category scores for each image subpatch. 

Not these are currently not re-normalized, and will ofen be negative, due to the way heuristics are applied.

---

Developed For :  
IaaC MRAC_19-20 Studio 2  
Faculty : Aldo Sollazzo, Daniel Serrano

Image Analysis Code Based on:  
Adrian Rosebrock, 'Content Based Image Retrieval', 'Image Classification and Machine Learning', 'Local Binary Patterns with Python & OpenCV', PyImageSearch