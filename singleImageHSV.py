import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

#####################################################
#
#  Author: Joseph Cheng
#  HSV: Hue Saturation Value
#  Hue is the color
#  Saturation is the greyness
#  Value is the brightness of the pixel
#
#
#####################################################


# STEP 1: Capture and store a background frame
# Creating a VideoCapture object
# This will be used for image acquisition later in the code.

cap = cv2.VideoCapture("JosephInvisible.mp4")

# We give some time for the camera to warm-up!
time.sleep(3)
background = 0
 
for i in range(30):
    ret, background = cap.read()
 
# Laterally invert the image / flip the image.
background = np.flip(background, axis=1)
# imgPlot = plt.imshow(background)
# plt.show()

#Step 2: Color detection
# Capturing the live frame
# ret, img = cap.read()
img = cv2.imread("ImageForColor.PNG")
# Laterally invert the image / flip the image
img = np.flip(img, axis=1)
# imgPlot = plt.imshow(img)
# plt.show()

# converting from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgPlot = plt.imshow(hsv)
plt.show()

# First Range boundary
lower_first_boundary = np.array([90, 100, 10])
upper_first_boundary = np.array([108, 220, 140])
mask = cv2.inRange(hsv, lower_first_boundary, upper_first_boundary)
# Second Range boundary
# lower_second_boundary = np.array([90, 100, 10])
# upper_second_boundary = np.array([108, 220, 140])
# mask2 = cv2.inRange(hsv, lower_second_boundary, upper_second_boundary)

# Generating the final mask to detect color of my clothing, acts like an or gate
# mask = mask1 + mask2

cv2.imshow("mask", mask)
cv2.waitKey(0)

#####################################################
# Morphological Transformation - basic operations on image shape
# Erosion - erodes away boundaries of foreground object (will only consider 1 if all surrounding are 1)
# Dilation - if any pixel contains 1 everything is 1
# Opening - erosion followed by dilation - useful in removing noise
# Closing - dilation followed by erosion - closing small holes inside the foreground
#####################################################

kernal = np.ones((3, 3), np.uint8)
img_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal, iterations=1)
img_dilate = cv2.morphologyEx(img_open, cv2.MORPH_DILATE, kernal, iterations=1)

# creating an inverted mask to segment out the cloth from the frame
clothing_mask = cv2.bitwise_not(img_dilate)

imgPlot = plt.imshow(clothing_mask)
plt.show()


# Segmenting the cloth out of the frame using bitwise and with the inverted mask
res1 = cv2.bitwise_and(img, img, mask=clothing_mask)

cv2.imshow("res1", res1)
cv2.waitKey(0)

# creating image showing static background frame pixels only for the masked region
# error might happen here because the bits are different between background and captured image
#res2 = cv2.bitwise_and(background, background, mask=clothing_mask)

# Generating the final output
# final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
# cv2.imshow("magic", final_output)
# cv2.waitKey(0)