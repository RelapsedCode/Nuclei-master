# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2 as cv, cv2
import sys
import numpy as np
from matplotlib import pylab
import mahotas as mh
import requests
from functools import reduce

# AFTER THE FIRST VALUE ALL OTHER VALUES ARE ZEROED BECAUSE GETPOSITION IN THE PROCESS FUNC

def createWindows(val):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 1620, 500)
    cv2.namedWindow("Results")

    pass

def CreateTrackbars(val):
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, ProcessImage)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, ProcessImage)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, ProcessImage)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, ProcessImage)

    cv2.createTrackbar("L - H /2", "Trackbars", 0, 179, ProcessImage)
    cv2.createTrackbar("L - S /2", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("L - V /2", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("U - H /2", "Trackbars", 179, 179, ProcessImage)
    cv2.createTrackbar("U - S /2", "Trackbars", 255, 255, ProcessImage)
    cv2.createTrackbar("U - V /2", "Trackbars", 255, 255, ProcessImage)

    cv2.createTrackbar("L - H /3", "Trackbars", 0, 179, ProcessImage)
    cv2.createTrackbar("L - S /3", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("L - V /3", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("U - H /3", "Trackbars", 179, 179, ProcessImage)
    cv2.createTrackbar("U - S /3", "Trackbars", 255, 255, ProcessImage)
    cv2.createTrackbar("U - V /3", "Trackbars", 255, 255, ProcessImage)

    # cv.createTrackbar("Morph Open", "Trackbars", 5, ksize_max_value, VisualAffrair)
    # cv.createTrackbar("Morph Close", "Trackbars", 5, ksize_max_value, VisualAffrair)

    cv.createTrackbar("ksize", "Trackbars", 0, 31, ProcessImage)
    pass

def idontdoshit():
    print('pAsS dA BuTThAaA')

def SetVals(val):

    # Make them lists?
    global hsvColorValues, hsvColorValues2, hsvColorValues3
    global lower_range
    global upper_range
    global img, mask, res, hsv
    global gaussian_value
    # low hue [0]; low saturation [1]; low value [2]; high hue [3]; high saturation [4]; high value [5]
    hsvColorValues = [5, 50, 50, 40, 255, 255]
    hsvColorValues2 = [5, 50, 50, 40, 255, 255]
    hsvColorValues3 = [5, 50, 50, 40, 255, 255]

    gaussian_value = 1

    lower_range = np.array([hsvColorValues[0], hsvColorValues[1], hsvColorValues[2]])
    upper_range = np.array([hsvColorValues[3], hsvColorValues[4], hsvColorValues[5]])

    img = cv2.imread(r"D:\side-projects\nuclei\img\nuclei.jpg")

    cv2.setTrackbarPos("L - H", "Trackbars", hsvColorValues[0])
    cv2.setTrackbarPos("L - S", "Trackbars", hsvColorValues[1])
    cv2.setTrackbarPos("L - V", "Trackbars", hsvColorValues[2])
    cv2.setTrackbarPos("U - H", "Trackbars", hsvColorValues[3])
    cv2.setTrackbarPos("U - S", "Trackbars", hsvColorValues[4])
    cv2.setTrackbarPos("U - V", "Trackbars", hsvColorValues[5])

def ProcessImage(val):
    # Get the new values of the trackbar in real time as the user changes them
    hsvColorValues[0] = cv2.getTrackbarPos("L - H", "Trackbars")
    hsvColorValues[1] = cv2.getTrackbarPos("L - S", "Trackbars")
    hsvColorValues[2] = cv2.getTrackbarPos("L - V", "Trackbars")
    hsvColorValues[3] = cv2.getTrackbarPos("U - H", "Trackbars")
    hsvColorValues[4] = cv2.getTrackbarPos("U - S", "Trackbars")
    hsvColorValues[5] = cv2.getTrackbarPos("U - V", "Trackbars")

    hsvColorValues2[0] = cv2.getTrackbarPos("L - H /2", "Trackbars")
    hsvColorValues2[1] = cv2.getTrackbarPos("L - S /2", "Trackbars")
    hsvColorValues2[2] = cv2.getTrackbarPos("L - V /2", "Trackbars")
    hsvColorValues2[3] = cv2.getTrackbarPos("U - H /2", "Trackbars")
    hsvColorValues2[4] = cv2.getTrackbarPos("U - S /2", "Trackbars")
    hsvColorValues2[5] = cv2.getTrackbarPos("U - V /2", "Trackbars")

    hsvColorValues3[0] = cv2.getTrackbarPos("L - H /3", "Trackbars")
    hsvColorValues3[1] = cv2.getTrackbarPos("L - S /3", "Trackbars")
    hsvColorValues3[2] = cv2.getTrackbarPos("L - V /3", "Trackbars")
    hsvColorValues3[3] = cv2.getTrackbarPos("U - H /3", "Trackbars")
    hsvColorValues3[4] = cv2.getTrackbarPos("U - S /3", "Trackbars")
    hsvColorValues3[5] = cv2.getTrackbarPos("U - V /3", "Trackbars")

    morphOpenKernel = cv.getTrackbarPos("Morph Open", "Trackbars")
    morphCloseKernel = cv.getTrackbarPos("Morph Close", "Trackbars")


    # Gaussian things, not pretty but working.
    gaussian_value = cv2.getTrackbarPos("ksize", "Trackbars")

    cv.setTrackbarPos("ksize", "Trackbars", gaussian_value)
    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if gaussian_value != 0:
        gaussian_value = isEven(gaussian_value)
        hsv = GaussianBlur(hsv, gaussian_value)

    # Set the lower and upper HSV range according to the value selected by the trackbar
    lower_range1 = np.array([hsvColorValues[0], hsvColorValues[1], hsvColorValues[2]])
    upper_range1 = np.array([hsvColorValues[3], hsvColorValues[4], hsvColorValues[5]])

    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the HSV image to get only blue colors. Filter the image and get the binary mask, where white represents your target color
    mask = cv2.inRange(hsv, lower_range1, upper_range1)
    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(img, img, mask=mask)
    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # stack the mask, orginal frame and the filtered result // mask is grayscale
    stacked = np.hstack((mask_3, img, res))

    # Morph operations



    # Show this stacked frame at 40% of the size.
    cv2.imshow('Results', cv2.resize(stacked, None, fx=1.0, fy=1.0))
    CountPixels(mask, hsv)
    print('1 time 2 time')

def main():

    createWindows(0)
    CreateTrackbars(0)
    SetVals(0)

    # Init that shit
    ProcessImage(0)

    # Count pixels
    #CountPixels(mask, hsv)

    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        exit()

    #If the user presses ESC then exit the program
    key = cv2.waitKey(1)

    # If the user presses `s` then print this array.
    if key == ord('s'):
        thearray = [[hsvColorValues[0], hsvColorValues[1], hsvColorValues[2]], [hsvColorValues[3], hsvColorValues[4],hsvColorValues[5]]]
        print(thearray)

        # Also save this array as penval.npy
        np.save('hsv_value', thearray)
        exit()

# def MorphClose(dst):
#     kernel = cv.getTrackbarPos(trackbar_close, windowName)
#     closingImg = cv.morphologyEx(dst, cv.MORPH_CLOSE, (256, 256))
#     #ShowImages(MorphClose.__name__, closingImg)
#     return closingImg
#
# def MorphOpen(dst, kernel):
#
#
#     openingImg = cv.morphologyEx(dst, cv.MORPH_OPEN, (kernel, kernel))
#
#     #ShowImages(MorphOpen.__name__, openingImg)
#
#     #SaveImages(MorphOpen.__name__, openingImg, 256)
#
#     return openingImg

def GaussianBlur(grayImg, ksize):
    #windowName.append(GaussianBlur.__name__)
    gaussianImg = grayImg
    #isEven(trackbar_gaussian, windowName, gaussian_value)
    gaussianImg = cv.GaussianBlur(gaussianImg, (ksize, ksize), 0)
    # ShowImages(GaussianBlur.__name__, gaussianImg)
    return gaussianImg

#
def isEven(value):
    if (((value % 2) == 1) or (value == 0)):
        pass
    else:
        value += 1
    return value

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

def SaveImage(x):
    pass

main()