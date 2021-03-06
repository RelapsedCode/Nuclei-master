# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# lower_brown = np.array([5,50,50])
# upper_brown = np.array([40, 255, 255])

import cv2 as cv, cv2
from PIL import Image
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import pylab
import mahotas as mh
import requests
import sys
import time

hsvValues = {
            'color_1': {'l_h': 0, 'l_s': 0, 'l_v': 0, 'u_h': 0, 'u_s': 0, 'u_v': 0},
        }

def createWindows():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 1000, 500)
    cv2.moveWindow("Trackbars", 0, 0)
    cv2.namedWindow("Results")
    cv2.resizeWindow("Results", 1920, 1200)


def CreateTrackbars():
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, ProcessImage)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, ProcessImage)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, ProcessImage)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, ProcessImage)

    cv.createTrackbar("Morph Open", "Trackbars", 1, 21, ProcessImage)
    cv.createTrackbar("Morph Close", "Trackbars", 1, 21, ProcessImage)

    cv.createTrackbar("ksize", "Trackbars", 0, 21, ProcessImage)


def SetVals():
    global mask_3, mask_1, mask_2
    global lower_range, lower_range2
    global upper_range, upper_range2
    global img, mask, res, hsv
    global gaussian_value, morphOpenKernel

    morphOpenKernel = 1
    morphCloseKernel = 1

    gaussian_value = 1
    lower_range = np.array([hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']])
    upper_range = np.array([hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']])

    img = cv2.imread(r"images\source-img\demo.jpg")

    cv2.setTrackbarPos("Morph Open", "Trackbars", morphOpenKernel)
    cv2.setTrackbarPos("Morph Open", "Trackbars", morphCloseKernel)


def dicValues(flag):
    global hsvValues
    if flag == 1:
        hsvValues = {
            'color_1': {'l_h': 0, 'l_s': 0, 'l_v': 0, 'u_h': 179, 'u_s': 255, 'u_v': 93},
        }
    elif flag == 2:
        hsvValues = {
            'color_1': {'l_h': 102, 'l_s': 56, 'l_v': 100, 'u_h': 179, 'u_s': 255, 'u_v': 190},
        }
    elif flag == 0:
        hsvValues = {
            'color_1': {'l_h': 0, 'l_s': 0, 'l_v': 0, 'u_h': 0, 'u_s': 0, 'u_v': 0},
        }


def setTrackbarsInitialPosition(hsvValues):
    # SetTrackbarPos replaces the elements of the inner dict with zeros!
    hsvLetters = ["L - H", "L - S", "L - V", "U - H", "U - S", "U - V"]
    outerKey = list(hsvValues.keys())[0]
    hsvValues1 = {outerKey: {}}
    temp = hsvValues[outerKey]
    for key, value in temp.items():
        hsvValues1[outerKey][key] = value

    # for ind in range(len(hsvLetters)):
    #     cv2.setTrackbarPos(hsvLetters[ind], "Trackbars", hsvValues1[outerKey][key])

    cv2.setTrackbarPos("L - H", "Trackbars", hsvValues1[outerKey]['l_h'])
    cv2.setTrackbarPos("L - S", "Trackbars", hsvValues1[outerKey]['l_s'])
    cv2.setTrackbarPos("L - V", "Trackbars", hsvValues1[outerKey]['l_v'])
    cv2.setTrackbarPos("U - H", "Trackbars", hsvValues1[outerKey]['u_h'])
    cv2.setTrackbarPos("U - S", "Trackbars", hsvValues1[outerKey]['u_s'])
    cv2.setTrackbarPos("U - V", "Trackbars", hsvValues1[outerKey]['u_v'])


def ProcessImage(val):
    # Get the new values of the trackbar in real time as the user changes them
    hsvValues['color_1']['l_h'] = cv2.getTrackbarPos("L - H", "Trackbars")
    hsvValues['color_1']['l_s'] = cv2.getTrackbarPos("L - S", "Trackbars")
    hsvValues['color_1']['l_v'] = cv2.getTrackbarPos("L - V", "Trackbars")
    hsvValues['color_1']['u_h'] = cv2.getTrackbarPos("U - H", "Trackbars")
    hsvValues['color_1']['u_s'] = cv2.getTrackbarPos("U - S", "Trackbars")
    hsvValues['color_1']['u_v'] = cv2.getTrackbarPos("U - V", "Trackbars")
    morphOpenKernel = cv.getTrackbarPos("Morph Open", "Trackbars")
    morphCloseKernel = cv.getTrackbarPos("Morph Close", "Trackbars")
    # Gaussian things, not pretty but working.
    gaussian_value = cv2.getTrackbarPos("ksize", "Trackbars")
    cv.setTrackbarPos("ksize", "Trackbars", gaussian_value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Image blur
    if gaussian_value != 0:
        gaussian_value = isEven(gaussian_value)
        hsv = GaussianBlur(hsv, gaussian_value)

    # Set the lower and upper HSV range according to the value selected by the trackbar
    lower_range = np.array([hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']])
    upper_range = np.array([hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']])

    # Threshold the HSV image to get only blue colors. Filter the image and get the binary mask, where white represents your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Apply morphologial opening and closing.
    # !!! --- Input GRAYSCALE IMAGE --- !!!
    global mask_3
    mask_3 = mask.copy()
    # mask_3 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if morphCloseKernel != 0:
        morphCloseKernel = isEven(morphCloseKernel)
        kernelClose = np.ones((morphCloseKernel, morphCloseKernel), np.uint8)
        mask_3 = cv.morphologyEx(mask_3, cv.MORPH_CLOSE, kernelClose)

    if morphOpenKernel != 0:
        morphOpenKernel = isEven(morphOpenKernel)
        kernelOpen = np.ones((morphOpenKernel, morphOpenKernel), np.uint8)
        mask_3 = cv.morphologyEx(mask_3, cv.MORPH_OPEN, kernelOpen)

    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_3 = cv2.cvtColor(mask_3, cv2.COLOR_GRAY2BGR)

    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3, img, res))
    cv2.imshow('Results', cv2.resize(stacked, None, fx=0.3, fy=0.3))
    cv2.imwrite(r"images\results\bitwiseNoOpenNoClose.jpg", mask_3)


# Add assertions if the files exist, because if there is not demo image the first run will trow an error when calling this func.
def overlap():
    cv2.destroyWindow("Mask 1 colored");
    cv2.destroyWindow("Mask 2 colored");

    # Close / Clear the pylab figure window
    # if plt.fignum_exists(1):
    #     plt.cla()
    #     plt.clf()
    # plt.close()

    mask_1 = plt.imread(r"images\results\test-1\1\mask1.jpg")
    ret, mask_1 = cv.threshold(mask_1, 250, 255, cv.THRESH_BINARY)
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2RGB)

    mask_2 = plt.imread(r"images\results\test-1\2\mask2.jpg")
    ret, mask_2 = cv.threshold(mask_2, 250, 255, cv.THRESH_BINARY)
    mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2RGB)

    print("image.shape()", mask_1.shape)

    height, width, depth = mask_1.shape
    overlap_img = np.zeros([height, width, depth], dtype=np.uint8)

    img1_black = mask_1.copy()
    img2_black = mask_2.copy()

    # # Try 1
    # img1_black[np.where((img1_black == [0, 0, 0]).all(axis=2))] = [0, 255, 0]
    # cv2.imshow("Result4", img1_black)

    # Try 2
    img1_black_pixels_mask = np.all(mask_1 == [0, 0, 0], axis=-1)
    img1_white_pixels_mask = np.any(mask_1 != [0, 0, 0], axis=-1)
    # Setting colors instead of masks
    img1_black[img1_black_pixels_mask] = [0, 0, 0]  # Is it RGB/BGR?
    img1_black[img1_white_pixels_mask] = [255, 255, 255]
    ret4, img1_black = cv.threshold(img1_black, 250, 255, cv.THRESH_BINARY)

    # Image 2
    img2_black_pixels_mask = np.all(mask_2 == [0, 0, 0], axis=-1)
    img2_white_pixels_mask = np.any(mask_2 != [0, 0, 0], axis=-1)
    # Revert black and white so we can make a difference between img-1 and img2
    # Where we HAD (PAST) black, replace them with white, and otherwise. Where we had white, replace it with black
    img2_black[img2_black_pixels_mask] = [255, 255, 255]  # Is it RGB?
    img2_black[img2_white_pixels_mask] = [0, 0, 0]
    ret5, img2_black = cv.threshold(img2_black, 250, 255, cv.THRESH_BINARY)

    stacked2 = np.hstack((img1_black, img2_black))
    cv2.imshow('Mask 1 - Mask 2', cv2.resize(stacked2, None, fx=0.4, fy=0.4))

    overlap_img[:] = [0, 0, 255]

    overlap_img[img1_white_pixels_mask] = [255, 255, 255]
    overlap_img[img2_white_pixels_mask] = [0, 0, 0]
    cv2.imshow('overlapped', cv2.resize(overlap_img, None, fx=0.4, fy=0.4))
    cv2.imwrite(r"images\results\overlap.jpg", overlap_img)

    CountPixels(overlap_img)

    # That works - iot and clear func NOT commented - dynamic closing, otherwise you have to close the image manually
    # pylab.imshow(img1_black)
    # pylab.imshow(img2_black)
    # plt.pause(0.001)
    #
    # pylab.jet()
    # #plt.ion()
    # pylab.show()

    # overlap_img[:] = (0, 0, 255)
    # overlap_img[0:height, 0:width // 4, 0:depth] = 0  # DO THIS INSTEAD


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


def CountPixels(img):
    redPix = np.count_nonzero(np.all(img == [0, 0, 255], axis=2))
    whitePix = np.count_nonzero(np.all(img == [255, 255, 255], axis=2))
    blackPix = np.count_nonzero(np.all(img == [0, 0, 0], axis=2))

    height, width, depth = img.shape
    allPixels = height*width
    print('All pixels: ' + str(allPixels))
    print('Red pixels are: ' + str(round(((redPix / allPixels) * 100), 2)) + '%' + ' of the image.')
    print('White pixels are: ' + str(round(((whitePix / allPixels) * 100), 2)) + '%' + ' of the image.')
    print('Black pixels are: ' + str(round(((blackPix / allPixels) * 100), 2)) + '%' + ' of the image.')
    print('Others pixels are: ' + str((allPixels) - (redPix+whitePix + blackPix)) + '. If this value is different than 0 than you have a problem.')


def ExportToCSV():
    pass


def SaveImage(x):
    pass


# defaultImg = np.zeros((300, 300, 3), np.uint8)
# defaultImg[:] = (0, 0, 255)
# cv2.imshow('Results4', defaultImg)
# print("mask shape: ", defaultImg.shape)


# Maybe add that settings to the txt file
def RGBToHSV():
    r, g, b = input("Your lower RGB values: ").split()
    r2, g2, b2 = input("Your upper RGB values: ").split()

    rgb_brown_lower = np.uint8([[[r, g, b]]])
    rgb_brown_upper = np.uint8([[[r2, g2, b2]]])
    hsv_brown_lower = cv2.cvtColor(rgb_brown_lower, cv2.COLOR_RGB2HSV)
    hsv_brown_upper = cv2.cvtColor(rgb_brown_upper, cv2.COLOR_RGB2HSV)
    print(hsv_brown_lower, hsv_brown_upper)


def main():
    print("Press \'K\' to convert RGB values to HSV.\n" + "Press \'Q\'/\'W\' to save image 1 / image 2.\n" + "M and N to load default HSV values.")
    global hsvValues

    # print('Which hich image would you like to process (write it with the file extension)?')
    # global imageToUse
    # imageToUse = input

    createWindows()
    CreateTrackbars()
    SetVals()
    dicValues(0)
    setTrackbarsInitialPosition(hsvValues)
    # Init that shit
    ProcessImage(0)

    # Count pixels
    # CountPixels(mask, hsv)

    # if cv2.waitKey(0) & 0xFF == ord("q"):
    #     cv2.destroyAllWindows()
    #     exit()

    while (1):
        key = cv2.waitKey(0)
        print(key)
        # If the user presses `s` then print this array.
        # if key == ord('q') & 0xFF == ord("q"):

        if (key == ord('q') & 0xFF == ord("q")):
            cv2.imwrite(r"images\results\test-1\1\mask1.jpg", mask_3)
            thearray = [[hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']],
                        [hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']]]
            np.save('hsv_value', thearray)
            print(thearray)
            overlap()

        elif key == ord('w') & 0xFF == ord("w"):
            cv2.imwrite(r"images\results\test-1\2\mask2.jpg", mask_3)

            thearray2 = [[hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']],
                         [hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']]]
            print(thearray2)
            overlap()
            # Also save this array as penval.npy
            np.save('hsv_value_2', thearray2)

        elif key == ord('k') & 0xFF == ord("k"):
            RGBToHSV()

        elif key == ord('m') & 0xFF == ord("m"):
            dicValues(2)
            setTrackbarsInitialPosition(hsvValues)

        elif key == ord('n') & 0xFF == ord("n"):
            dicValues(1)
            setTrackbarsInitialPosition(hsvValues)

        else:
            cv2.destroyAllWindows()
            exit()


# key = cv2.waitKey(0)
main()