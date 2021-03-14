# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# DEFAULT BROWN LIMITS
# lower_brown = np.array([5,50,50])
# upper_brown = np.array([40, 255, 255])

import cv2
import sys
import numpy as np
from matplotlib import pylab
import mahotas as mh
import requests
from functools import reduce

def nothing(x):
    pass

global l_h
global l_s
global l_v
global u_h
global u_s
global u_v
global lower_range
global upper_range
global img, mask, res, hsv


def SetVals():
    l_h = 5
    l_s = 50
    l_v = 50
    u_h = 40
    u_s = 255
    u_v = 255
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

def CreateTrackbar():
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

def SaveImage(x):
    pass

def CountPixels(mask, hsv):
    maskShape = mask.shape
    allPixels = 1
    for i in maskShape:
        allPixels *= i
    print('simple for loop: ' + str(allPixels))

    allPixels2 = reduce(lambda x, y: x * y, maskShape)
    print('lambda expression: ' + str(allPixels2))

    allPixels3 = mask.size
    print('img.size function: ' + str(allPixels3))

    whitePixels = cv2.countNonZero(mask)
    print ('white pixelstry.py: ' + str(whitePixels))

    blackPixels = allPixels - whitePixels
    print('black pixes:' + str(blackPixels))

    whitePixPerc = (whitePixels  / allPixels) * 100
    blackPixPerc = (blackPixels  / allPixels) * 100
    print('white pixels (cancerogenni) define: ' + str(round(whitePixPerc, 2)) + '%' + ' of the image.')
    print('black pixels (fine) define: ' + str(round(blackPixPerc, 2)) + '%' + ' of the image.')

cv2.namedWindow("Trackbars")
img = cv2.imread(r"D:\side-projects\nuclei\img\nuclei.jpg")
CreateTrackbar()
SetVals()

while True:



    # Get the new values of the trackbar in real time as the user changes them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Set the lower and upper HSV range according to the value selected by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors. Filter the image and get the binary mask, where white represents your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)

    #You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3, img, res))

    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars',cv2.resize(stacked, None, fx=0.8, fy=0.8))

    # Count pixels
    CountPixels(mask, hsv)

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

    # If the user presses `s` then print this array.
    if key == ord('s'):
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)

        # Also save this array as penval.npy
        np.save('hsv_value', thearray)
        break
    i = 0
    i=+i

cv2.imwrite(r'D:\side-projects\nuclei\th-ing.jpg', mask)
cv2.destroyAllWindows()


# dna = mh.imread(r'D:\side-projects\nuclei\th-ing.jpg')
# dna = dna.astype('uint8')
# dnaf = mh.gaussian_filter(dna, 2)
# dnaf = dnaf.astype('uint8')
#
# T = mh.thresholding.otsu(dnaf)
# pylab.imshow(dnaf > T)
# pylab.show()
#
#
# labeled,nr_objects = mh.label(dnaf > T)
# print (nr_objects)
# pylab.imshow(labeled)
# pylab.jet()
# pylab.show()
#
# dnaf = mh.gaussian_filter(dna, 16)
# rmax = mh.regmax(dnaf)
# pylab.imshow(mh.overlay(dna, rmax))
# pylab.show()
#
#
# k = cv.waitKey(0)
#
# cv.destroyAllWindows()
