# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab
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

T = mh.thresholding.otsu(img)
pylab.imshow(img > T)
pylab.show()

# for elem in windowName:
#     cv.namedWindow(elem, cv.WINDOW_NORMAL)
#     #cv.resizeWindow(elem, 1920, 1080)
#     cv.imshow(elem, eval(elem))

k = cv.waitKey(0)

cv.destroyAllWindows()