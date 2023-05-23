import cv2
image = cv2.imread('testimage.jpeg')
gray_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)