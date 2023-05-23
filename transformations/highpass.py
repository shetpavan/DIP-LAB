import numpy as np
import pandas as pd
import cv2
img = cv2.imread("testimage.jpeg",cv2.IMREAD_UNCHANGED)
gaussBlur = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
highPass = img - gaussBlur
cv2.imshow("High Pass",np.hstack((img, highPass)))
cv2.waitKey(0)
cv2.destroyAllWindows()