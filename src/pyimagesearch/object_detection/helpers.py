# import the necessary packages
import imutils

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def sliding_window_double(imageA,imageB,stepSize,windowSize):
    """ Dead simple way to slide across multiple windows, obv doesn't scale, assumes both images are the same shape"""
    # slide a window across the image
    for y in range(0, imageA.shape[0], stepSize):
        for x in range(0, imageA.shape[1], stepSize):
            # yield the current window
            yield (x, y, imageA[y:y + windowSize[1], x:x + windowSize[0]],imageB[y:y + windowSize[1], x:x + windowSize[0]])
