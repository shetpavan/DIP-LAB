import numpy as np
import pandas as pd
import cv2
img = cv2.imread("testimage.jpeg",cv2.IMREAD_UNCHANGED)
domainFilter = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.6)
cv2.imshow('Domain Filter',domainFilter)
cv2.waitKey(0)
cv2.destroyAllWindows()
gaussBlur = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
cv2.imshow("Gaussian Smoothing",np.hstack((img,gaussBlur)))
cv2.waitKey(0)
cv2.destroyAllWindows()