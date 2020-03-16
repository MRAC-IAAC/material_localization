MRAC19-20 Studio 2 Group 1 : Material Detection
===============================================

Performs material localization in images taken from demolition site drone inspection. 

Still very alpha, accuracy and resolution both need major work.

Patch classification is done primarily using a BOVW algorithm, with heuristics from a comparison of hue and saturation histograms.

Feature Detection uses the GFTT corner detection algorithm

Descriptor Extraction uses the BRISK binary descriptor algorithm. 


To Use
------

### Environment



### Setup

To process the input images and train the classifier, run

`./setup_detector`

Ensure that the 'dataset\_path' variable in the setup_detector script is set correctly for your system. 

This will take ~20 minutes to run, mostly due to sampling the database during the feature clustering step.

This will produce several files in the 'output/' folder: 

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

Each image takes ~5 seconds to process. 

This will write images with category colors to the 'output_localization' folder, as well as text files containing the heuristically weighted category probabilities for each image subpatch. 

Not these are currently not re-normalized, and will almost always be negative. 

---

Based on:

Adrian Rosebrock, 'Content Based Image Retrieval' & 'Image Classification and Machine Learning', PyImageSearch