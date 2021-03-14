# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# lower_brown = np.array([5,50,50])
# upper_brown = np.array([40, 255, 255])

import cv2 as cv, cv2
import sys
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
import mahotas as mh
import requests
from functools import reduce
import time

def createWindows(val):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 1000, 500)
    cv2.moveWindow("Trackbars", 0, 0)
    cv2.namedWindow("Results")
    cv2.resizeWindow("Results", 1920, 1200)
    pass

def CreateTrackbars(val):
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, ProcessImage)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, ProcessImage)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, ProcessImage)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, ProcessImage)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, ProcessImage)

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
    }

    hsvValues['color_1']['l_h'] = 5
    hsvValues['color_1']['l_s'] = 50
    hsvValues['color_1']['l_v'] = 50
    hsvValues['color_1']['u_h'] = 40
    hsvValues['color_1']['u_s'] = 255
    hsvValues['color_1']['u_v'] = 255

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

    #img = cv2.imread(r"D:\side-projects\nuclei\img\" + imagetoUse)
    img = cv2.imread(r"D:\side-projects\nuclei\img\irl-img\ki001.jpg")

    cv2.setTrackbarPos("Morph Open", "Trackbars", morphOpenKernel)
    cv2.setTrackbarPos("Morph Open", "Trackbars", morphCloseKernel)

def setTrackbarsInitialPosition(val):
    cv2.setTrackbarPos("L - H", "Trackbars", hsvValues['color_1']['l_h'])
    cv2.setTrackbarPos("L - S", "Trackbars", hsvValues['color_1']['l_s'])
    cv2.setTrackbarPos("L - V", "Trackbars", hsvValues['color_1']['l_v'])
    cv2.setTrackbarPos("U - H", "Trackbars", hsvValues['color_1']['u_h'])
    cv2.setTrackbarPos("U - S", "Trackbars", hsvValues['color_1']['u_s'])
    cv2.setTrackbarPos("U - V", "Trackbars", hsvValues['color_1']['u_v'])

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
    mask_3 = mask.copy()
    #mask_3 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
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
    cv2.imwrite(r"D:\side-projects\nuclei\results\temp\bitwiseNoOpenNoClose.jpg", mask_3)

    #CountPixels(mask, hsv, mask_4)

def overlap():
    cv2.destroyWindow("Mask 1 colored");
    cv2.destroyWindow("Mask 2 colored");

    # Close / Clear the pylab figure window
    # if plt.fignum_exists(1):
    #     plt.cla()
    #     plt.clf()
        # plt.close()

    mask_1 = plt.imread(r"D:\side-projects\nuclei\results\test-1\1\mask1.jpg")
    ret, mask_1 = cv.threshold(mask_1, 250, 255, cv.THRESH_BINARY)
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2RGB)

    mask_2 = plt.imread(r"D:\side-projects\nuclei\results\test-1\2\mask2.jpg")
    ret, mask_2 = cv.threshold(mask_2, 250, 255, cv.THRESH_BINARY)
    mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2RGB)

    print(mask_1.shape)

    height, width, depth = mask_1.shape
    overlap_img = np.zeros([height, width, depth], dtype=np.uint8)

    img1_black = mask_1.copy()
    img2_black = mask_2.copy()

    # # Try 1
    # img1_black[np.where((img1_black == [0, 0, 0]).all(axis=2))] = [0, 255, 0]
    # cv2.imshow("Result4", img1_black)

    # Try 2
    img1_black_pixels_mask = np.all(mask_1 == [0, 0, 0], axis=-1)
    img1_non_black_pixels_mask = np.any(mask_1 != [0, 0, 0], axis=-1)

    # Creating mask for white and black
    img2_black_pixels_mask = np.all(mask_2 == [0, 0, 0], axis=-1)
    img2_non_black_pixels_mask = np.any(mask_2 != [0, 0, 0], axis=-1)

   # Setting colors instead of masks
    img1_black[img1_black_pixels_mask] = [0, 0, 0] #Is it RGB/BGR?
    img1_black[img1_non_black_pixels_mask] = [255, 0, 0]

    img2_black[img2_black_pixels_mask] = [0, 0, 0] #Is it RGB?
    img2_black[img2_non_black_pixels_mask] = [0, 255, 0]

    stacked2 = np.hstack((img1_black, img2_black))
    cv2.imshow('Mask 1 - Mask 2', cv2.resize(stacked2, None, fx=0.4, fy=0.4))

    # for x in height:
    #     for y in width:
    #         for z in depth:
    #             if (img1_black[x][y][z] == )

    # img3_black = img1_black
    # img3_black[img1_black] = [255, 125, 0]
    # img3_black[img2_black] = [255, 125, 0]
    # cv2.imshow('Mask 3',img3_black)

    # That works - iot and clear func NOT commented - dynamic closing, otherwise you have to close the image manually
    # pylab.imshow(img1_black)
    # pylab.imshow(img2_black)
    # plt.pause(0.001)
    #
    # pylab.jet()
    # #plt.ion()
    # pylab.show()

    # overlap_img[:] = (0, 0, 255)
    #
    # overlap_img[0:height, 0:width // 4, 0:depth] = 0  # DO THIS INSTEAD
    #
    # cv2.imshow('Results4', img1_black)
    # cv2.imwrite(r"D:\side-projects\nuclei\results\\test-1\overlal.jpg", overlap_img)
    #
    # img[0:height, 0:width//4, 0:depth] = 0 # DO THIS INSTEAD

# def MorphClose(dst):
#     kernel = cv.getTrackbarPos(trackbar_close, windowName)
#     closingImg = cv.morphologyEx(dst, cv.MORPH_CLOSE, (256, 256))
#     ShowImages(MorphClose.__name__, closingImg)
#     return closingImg
#

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
    #openimg = cv2.cvtColor(openimg, cv2.COLOR_HSV2RGB)
    #openimg = cv2.cvtColor(openimg, cv2.COLOR_RGB2GRAY)
    maskShape = mask.shape
    allPixels = 1
    for i in maskShape:
        allPixels *= i
    # print('simple for loop: ' + str(allPixels))
    allPixels2 = reduce(lambda x, y: x * y, maskShape)
    # print('lambda expression: ' + str(allPixels2))
    allPixels3 = mask.size
    # print('img.size function: ' + str(allPixels3))

    whitePixels = cv2.countNonZero(mask)
    # print('white pixelstry.py: ' + str(whitePixels))

    # Morph open pixs
    whitePixOpen = cv2.countNonZero(openimg)
    # print('white pixelstry.py: ' + str(whitePixels))

    blackPixels = allPixels - whitePixels
    # print('black pixes:' + str(blackPixels))

    whitePixPerc = (whitePixels / allPixels) * 100

    whitePixOpenPerc = (whitePixOpen / allPixels) * 100

    blackPixPerc = (blackPixels / allPixels) * 100

    print('white pixels (cancerogenni) define: ' + str(round(whitePixPerc, 2)) + '%' + ' of the image.')
    print('white OPEN pixels (cancerogenni) define: ' + str(round(whitePixOpenPerc, 2)) + '%' + ' of the image.')

    print('black pixels (fine) define: ' + str(round(blackPixPerc, 2)) + '%' + ' of the image.')

def CountColouredPixels():
    pass

def ReportToCSV():
    pass

def SaveImage(x):
    pass

# defaultImg = np.zeros((300, 300, 3), np.uint8)
# defaultImg[:] = (0, 0, 255)
# cv2.imshow('Results4', defaultImg)
# print("mask shape: ", defaultImg.shape)


def morphworking():
    kernel = np.ones((8, 8), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    morph = cv2.imread(r"D:\side-projects\nuclei\results\test-1\1\mask1 - Copy.jpg")
    cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)
    morph2 = morph.copy()
    morph2 = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
    morph3 = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel2)
    stacked3 = np.hstack((morph, morph3))
    cv2.imshow('Results', cv2.resize(stacked3, None, fx=0.5, fy=0.5))
    cv2.imshow('Results2', morph)
    pylab.imshow(morph)
    pylab.imshow(morph2)
    pylab.show()

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
    print("Press \'K\' to convert RGB values to HSV.\n" + "Press \'Q\'/\'W\' to save image 1/image2")

    # print('Which hich image would you like to process (write it with the file extension)?')
    # global imageToUse
    # imageToUse = input

    createWindows(0)
    CreateTrackbars(0)
    SetVals(0)
    setTrackbarsInitialPosition(0)
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
            cv2.imwrite(r"D:\side-projects\nuclei\results\test-1\1\mask1.jpg", mask_3)
            thearray = [[hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']],
                        [hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']]]
            np.save('hsv_value', thearray)
            print(thearray)
            overlap()

        elif key == ord('w') & 0xFF == ord("w"):
            cv2.imwrite(r"D:\side-projects\nuclei\results\test-1\2\mask2.jpg", mask_3)

            thearray2 = [[hsvValues['color_1']['l_h'], hsvValues['color_1']['l_s'], hsvValues['color_1']['l_v']],
                        [hsvValues['color_1']['u_h'], hsvValues['color_1']['u_s'], hsvValues['color_1']['u_v']]]
            print(thearray2)
            overlap()
            # Also save this array as penval.npy
            np.save('hsv_value_2', thearray2)

        elif key == ord('k') & 0xFF == ord("k"):
            RGBToHSV()

        else:
            cv2.destroyAllWindows()
            exit()

#morphworking()
#key = cv2.waitKey(0)
main()