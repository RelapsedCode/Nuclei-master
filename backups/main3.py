# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import pylab
import mahotas as mh
import requests

windowName = ['mask', 'res']

img = cv.imread(r"D:\side-projects\nuclei\img\cells.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)


#bgr brown
green = np.uint8([[[50,90,120]]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print (hsv_green)

lower_brown = np.array([5,50,50])
upper_brown = np.array([40, 255, 255])

color = np.uint8([[[50,90,120]]])
hsv_color = cv.cvtColor(color, cv.COLOR_BGR2HSV)
print(hsv_color)



# Threshold the HSV image to get only blue colors
mask = cv.inRange(img, lower_brown, upper_brown)

# Bitwise-AND mask and original image
res = cv.bitwise_and(img, img, mask=mask)

dnaf = mh.gaussian_filter(res, 8)
T = mh.thresholding.otsu(dnaf)
pylab.imshow(dnaf > T)
pylab.show()


# for elem in windowName:
#     cv.namedWindow(elem, cv.WINDOW_NORMAL)
#     #cv.resizeWindow(elem, 1920, 1080)
#     cv.imshow(elem, eval(elem))

k = cv.waitKey(0)

cv.destroyAllWindows()

# alpha = 0.9 # Simple contrast control
# beta = 0  # Simple brightness control
# Initialize values
# print(' Basic Linear Transforms ')
# print('-------------------------')

#
# for y in range(img.shape[0]):


# windowsName = ['Original Image']
# imgList = 0
#
# print (img.dtype)
#
# for idx, elem in enumerate(windowsName):
#     cv.namedWindow(elem, cv.WINDOW_NORMAL)
#     cv.resizeWindow(elem, 1920, 1080)
#     cv.imshow(elem, img)
#
# k = cv.waitKey(0)
#
# # #cv.namedWindow(windowName, cv.WINDOW_NORMAL)
# # #cv.resizeWindow(windowName, 1920, 1080)
# #
# # global thresholding
# ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# blur = cv.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in range(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     cv.imwrite(titles[i*3] + '.png', images[i*3])
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     cv.imwrite(titles[i*3+2] + '.png', images[i*3+2])
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#
# plt.show()
#
# k = cv.waitKey(0)

