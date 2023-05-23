import numpy as np
import pandas as pd
import cv2
img = cv2.imread("testimage.jpeg",cv2.IMREAD_UNCHANGED)
kernel = np.ones((10,10),np.float32)/25
lowPass = cv2.filter2D(img,-1, kernel)
lowPass = img - lowPass
cv2.imshow("Low Pass",np.hstack((img, lowPass)))
cv2.waitKey(0)
cv2.destroyAllWindows()