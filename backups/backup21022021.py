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

    cv.createTrackbar("Morph Open", "Trackbars", 5, ksize_max_value, VisualAffrair)
    cv.createTrackbar("Morph Close", "Trackbars", 5, ksize_max_value, VisualAffrair)

    cv.createTrackbar("ksize", "Trackbars", 0, 31, ProcessImage)
    pass


def idontdoshit():
    print('pAsS dA BuTThAaA')


def SetVals(val):
    # Make them lists?
    global l_h
    global l_s
    global l_v
    global u_h
    global u_s
    global u_v
    global lower_range
    global upper_range
    global img, mask, res, hsv
    global gaussian_value
    hsvColorValues = [5, 50, 50, 40, 255, 255]
    hsvColorValues2 = [5, 50, 50, 40, 255, 255]
    hsvColorValues3 = [5, 50, 50, 40, 255, 255]
    l_h = 5
    l_s = 50
    l_v = 50
    u_h = 40
    u_s = 255
    u_v = 255

    gaussian_value = 1
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    img = cv2.imread(r"D:\side-projects\nuclei\img\nuclei.jpg")
    cv2.setTrackbarPos("L - H", "Trackbars", l_h)
    cv2.setTrackbarPos("L - S", "Trackbars", l_s)
    cv2.setTrackbarPos("L - V", "Trackbars", l_v)
    cv2.setTrackbarPos("U - H", "Trackbars", u_h)
    cv2.setTrackbarPos("U - S", "Trackbars", u_s)
    cv2.setTrackbarPos("U - V", "Trackbars", u_v)


def ProcessImage(val):
    # Get the new values of the trackbar in real time as the user changes them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    l_h2 = cv2.getTrackbarPos("L - H /2", "Trackbars")
    l_s2 = cv2.getTrackbarPos("L - S /2", "Trackbars")
    l_v2 = cv2.getTrackbarPos("L - V /2", "Trackbars")
    u_h2 = cv2.getTrackbarPos("U - H /2", "Trackbars")
    u_s2 = cv2.getTrackbarPos("U - S /2", "Trackbars")
    u_v2 = cv2.getTrackbarPos("U - V /2", "Trackbars")

    l_h3 = cv2.getTrackbarPos("L - H /3", "Trackbars")
    l_s3 = cv2.getTrackbarPos("L - S /3", "Trackbars")
    l_v3 = cv2.getTrackbarPos("L - V /3", "Trackbars")
    u_h3 = cv2.getTrackbarPos("U - H /3", "Trackbars")
    u_s3 = cv2.getTrackbarPos("U - S /3", "Trackbars")
    u_v3 = cv2.getTrackbarPos("U - V /3", "Trackbars")

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
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the HSV image to get only blue colors. Filter the image and get the binary mask, where white represents your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(img, img, mask=mask)
    # too many dimentions
    mask2 = 255 - mask
    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3, img, res, mask2))

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
    # CountPixels(mask, hsv)

    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        exit()

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)

    # If the user presses `s` then print this array.
    if key == ord('s'):
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)

        # Also save this array as penval.npy
        np.save('hsv_value', thearray)
        exit()


def MorphClose(dst):
    kernel = cv.getTrackbarPos(trackbar_close, windowName)
    closingImg = cv.morphologyEx(dst, cv.MORPH_CLOSE, (256, 256))
    ShowImages(MorphClose.__name__, closingImg)
    return closingImg


def MorphOpen(dst, kernel):
    openingImg = cv.morphologyEx(dst, cv.MORPH_OPEN, (kernel, kernel))

    # ShowImages(MorphOpen.__name__, openingImg)

    # SaveImages(MorphOpen.__name__, openingImg, 256)

    return openingImg


def GaussianBlur(grayImg, ksize):
    # windowName.append(GaussianBlur.__name__)
    gaussianImg = grayImg
    # isEven(trackbar_gaussian, windowName, gaussian_value)
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
    print('white pixelstry.py: ' + str(whitePixels))

    blackPixels = allPixels - whitePixels
    print('black pixes:' + str(blackPixels))

    whitePixPerc = (whitePixels / allPixels) * 100
    blackPixPerc = (blackPixels / allPixels) * 100
    print('white pixels (cancerogenni) define: ' + str(round(whitePixPerc, 2)) + '%' + ' of the image.')
    print('black pixels (fine) define: ' + str(round(blackPixPerc, 2)) + '%' + ' of the image.')


def SaveImage(x):
    pass


main()