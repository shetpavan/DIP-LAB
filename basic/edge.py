import cv2
import numpy as np

FILE_NAME = 'testimage.jpeg'
try:
	# Read image from disk.
	img = cv2.imread(FILE_NAME)

	# Canny edge detection.
	res = cv2.Canny(img, 100, 200)

	# Write image back to disk.
	cv2.imshow("Result",res)
	cv2.waitKey(0)
except IOError:
	print ('Error while reading files !!!')
