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
    cv2.resizeWindow("Trackbars", 1920, 1200)

    cv2.namedWindow("Results")
    cv2.resizeWindow("Results", 1920, 1200)

    cv2.namedWindow("Results2")
    cv2.resizeWindow("Results2", 1920, 1200)
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

    cv.createTrackbar("Morph Open", "Trackbars", 1, 21, ProcessImage)
    cv.createTrackbar("Morph Close", "Trackbars", 1, 21, ProcessImage)

    cv.createTrackbar("ksize", "Trackbars", 0, 21, ProcessImage)
    pass


def idontdoshit():
    print('pAsS dA BuTThAaA')


def SetVals(val):
    # Make them DICS?
    global hsvValues
    hsvValues = {
        'color_1': {'l_h': 0, 'l_s': 0, 'l_v': 0, 'u_h': 0, 'h_s': 0, 'u_v': 0},
        'color_2': {'l_h': 0, 'l_s': 0, 'l_v': 0, 'u_h': 0, 'h_s': 0, 'u_v': 0},
        'color_3': {'l_h': 0, 'l_s': 0, 'l_v': 0, 'u_h': 0, 'h_s': 0, 'u_v': 0}
    }
    hsvValues['color_1']['l_h'] = 5
    hsvValues['color_1']['l_s'] = 50
    hsvValues['color_1']['l_v'] = 50
    hsvValues['color_1']['u_h'] = 40
    hsvValues['color_1']['u_s'] = 255
    hsvValues['color_1']['u_v'] = 255

    hsvValues['color_2']['l_h'] = 5
    hsvValues['color_2']['l_s'] = 50
    hsvValues['color_2']['l_v'] = 50
    hsvValues['color_2']['u_h'] = 40
    hsvValues['color_2']['u_s'] = 255
    hsvValues['color_2']['u_v'] = 255

    global lower_range, lower_range2
    global upper_range, upper_range2
    global img, mask, res, hsv
    global gaussian_value, morphOpenKernel

    morphOpenKernel = 5
    morphCloseKernel = 5

    gaussian_value = 1
    lower_range = np.array([hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']])
    upper_range = np.array([hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']])

    lower_range2 = np.array([hsvValues['color_2']['l_h'], hsvValues['color_2']['l_s'], hsvValues['color_2']['l_v']])
    upper_range2 = np.array([hsvValues['color_2']['u_h'], hsvValues['color_2']['u_s'], hsvValues['color_2']['u_v']])

    img = cv2.imread(r"D:\side-projects\nuclei\img\nuclei.jpg")
    cv2.setTrackbarPos("L - H", "Trackbars", hsvValues['color_1']['l_h'])
    cv2.setTrackbarPos("L - S", "Trackbars", hsvValues['color_1']['l_s'])
    cv2.setTrackbarPos("L - V", "Trackbars", hsvValues['color_1']['l_v'])
    cv2.setTrackbarPos("U - H", "Trackbars", hsvValues['color_1']['u_h'])
    cv2.setTrackbarPos("U - S", "Trackbars", hsvValues['color_1']['u_s'])
    cv2.setTrackbarPos("U - V", "Trackbars", hsvValues['color_1']['u_v'])

    cv2.setTrackbarPos("Morph Open", "Trackbars", morphOpenKernel)
    cv2.setTrackbarPos("Morph Open", "Trackbars", morphCloseKernel)

def ProcessImage(val):
    # defaultImg = np((300, 300, 3), np.uint8)
    # # Fill image with red color(set each pixel to red)
    # defaultImg[:] = (0, 0, 255)
    # rows, cols, ch = defaultImg.shape

    # for y in defaultImg.cols:
    #     for x in defaultImg.rows:
    #         for z in defaultImg.ch:
    #             if defaultImg.at(x,y,z) == 255:
    #                 pass


   # #for y in mask.co


    # Get the new values of the trackbar in real time as the user changes them
    hsvValues['color_1']['l_h'] = cv2.getTrackbarPos("L - H", "Trackbars")
    hsvValues['color_1']['l_s'] = cv2.getTrackbarPos("L - S", "Trackbars")
    hsvValues['color_1']['l_v'] = cv2.getTrackbarPos("L - V", "Trackbars")
    hsvValues['color_1']['u_h'] = cv2.getTrackbarPos("U - H", "Trackbars")
    hsvValues['color_1']['u_s'] = cv2.getTrackbarPos("U - S", "Trackbars")
    hsvValues['color_1']['u_v'] = cv2.getTrackbarPos("U - V", "Trackbars")

    hsvValues['color_2']['l_h'] = cv2.getTrackbarPos("L - H /2", "Trackbars")
    hsvValues['color_2']['l_s'] = cv2.getTrackbarPos("L - S /2", "Trackbars")
    hsvValues['color_2']['l_v'] = cv2.getTrackbarPos("L - V /2", "Trackbars")
    hsvValues['color_2']['u_h'] = cv2.getTrackbarPos("U - H /2", "Trackbars")
    hsvValues['color_2']['u_s'] = cv2.getTrackbarPos("U - S /2", "Trackbars")
    hsvValues['color_2']['u_v'] = cv2.getTrackbarPos("U - V /2", "Trackbars")

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
    lower_range = np.array([hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']])
    upper_range = np.array([hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']])

    lower_range2 = np.array([hsvValues['color_2']['l_h'], hsvValues['color_2']['l_s'], hsvValues['color_2']['l_v']])
    upper_range2 = np.array([hsvValues['color_2']['u_h'], hsvValues['color_2']['u_s'], hsvValues['color_2']['u_v']])

    # Threshold the HSV image to get only blue colors. Filter the image and get the binary mask, where white represents your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(img, img, mask=mask)
    # too many dimentions
    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3, img, res))
    cv2.imshow('Results', cv2.resize(stacked, None, fx=1.0, fy=1.0))
    cv2.imwrite(r"D:\side-projects\nuclei\results\bitwiseNoOpenNoClose.jpg", mask_3)

    # Threshold the HSV image to get only blue colors. Filter the image and get the binary mask, where white represents your target color
    mask2 = cv2.inRange(hsv, lower_range2, upper_range2)
    # You can also visualize the real part of the target color (Optional)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    # too many dimentions
    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_4 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    stacked2 = np.hstack((mask_4, img, res2))
    cv2.imshow('Results2', cv2.resize(stacked2, None, fx=0.4, fy=0.4))
    cv2.imwrite(r"D:\side-projects\nuclei\results\bitwiseNoOpenNoClose2.jpg", mask_4)

    # Morph operations
    if morphOpenKernel != 0:
        morphOpenKernel = isEven(morphOpenKernel)
        openImg = mask
        openImg = MorphOpen(mask, morphOpenKernel)
        openImg = MorphClose(mask, morphCloseKernel)

    # Show this stacked frame at 40% of the size.
    #stacked2 = np.hstack((openImg))
    cv2.imshow('Results3', openImg)
    cv2.imwrite(r"D:\side-projects\nuclei\results\bitwiseOpened.jpg", openImg)

    CountPixels(mask, hsv, openImg)
    print('1 time 2 time')


def main():
    defaultImg = np.zeros((300, 300, 3), np.uint8)
    defaultImg[:] = (0, 0, 255)

    print(defaultImg.shape)
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
        thearray = [[hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']],
                    [hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']]]
        print(thearray)
        # Also save this array as penval.npy
        np.save('hsv_value', thearray)
        exit()

# def MorphClose(dst):
#     kernel = cv.getTrackbarPos(trackbar_close, windowName)
#     closingImg = cv.morphologyEx(dst, cv.MORPH_CLOSE, (256, 256))
#     ShowImages(MorphClose.__name__, closingImg)
#     return closingImg
#
def MorphOpen(dst, kernel):
    openingImg = cv.morphologyEx(dst, cv.MORPH_OPEN, (kernel, kernel))
    # ShowImages(MorphOpen.__name__, openingImg)
    # SaveImages(MorphOpen.__name__, openingImg, 256)
    return openingImg

def MorphClose(dst, kernel):
    openingImg = cv.morphologyEx(dst, cv.MORPH_CLOSE, (kernel, kernel))
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

def isEven(value):
    if (((value % 2) == 1) or (value == 0)):
        pass
    else:
        value += 1
    return value


def CountPixels(mask, hsv, openimg):
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

    # Morph open pixs
    whitePixOpen = cv2.countNonZero(openimg)
    print('white pixelstry.py: ' + str(whitePixels))

    blackPixels = allPixels - whitePixels
    print('black pixes:' + str(blackPixels))

    whitePixPerc = (whitePixels / allPixels) * 100

    whitePixOpenPerc = (whitePixOpen / allPixels) * 100

    blackPixPerc = (blackPixels / allPixels) * 100

    print('white pixels (cancerogenni) define: ' + str(round(whitePixPerc, 2)) + '%' + ' of the image.')
    print('white OPEN pixels (cancerogenni) define: ' + str(round(whitePixOpenPerc, 2)) + '%' + ' of the image.')

    print('black pixels (fine) define: ' + str(round(blackPixPerc, 2)) + '%' + ' of the image.')

def SaveImage(x):
    pass

main()