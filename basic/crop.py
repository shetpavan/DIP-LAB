import cv2
import numpy as np
image= cv2.imread('testimage.jpeg')
resized_img= image[15:170, 20:200]
cv2.imshow("Resize", resized_img)
cv2.waitKey(0)