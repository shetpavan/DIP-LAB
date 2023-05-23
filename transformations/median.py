import numpy as np
import pandas as pd
import cv2
img = cv2.imread("testimage.jpeg",cv2.IMREAD_UNCHANGED)
medianFilter = cv2.medianBlur(img,5)
cv2.imshow("Median Filter",np.hstack((img, medianFilter)))
cv2.waitKey(0)
cv2.destroyAllWindows()