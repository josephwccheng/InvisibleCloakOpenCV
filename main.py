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

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# We give some time for the camera to warm-up!
time.sleep(3)
background = 0

for i in range(30):
    ret, background = cap.read()


# Laterally invert the image / flip the image.
background = np.flip(background, axis=1)



#Boundary parameters
lower_boundary = np.array([90, 100, 10])
upper_boundary = np.array([108, 220, 120])
#Parameter for morphological Transformation
kernal = np.ones((3, 3), np.uint8)

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        img = np.flip(frame, axis=1)
        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_boundary, upper_boundary)

        #####################################################
        # Morphological Transformation - basic operations on image shape
        # Erosion - erodes away boundaries of foreground object (will only consider 1 if all surrounding are 1)
        # Dilation - if any pixel contains 1 everything is 1
        # Opening - erosion followed by dilation - useful in removing noise
        # Closing - dilation followed by erosion - closing small holes inside the foreground
        #####################################################

        # mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal, iterations=1)
        # mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernal, iterations=2)
        # mask_dilate = cv2.morphologyEx(mask_close, cv2.MORPH_DILATE, kernal, iterations=1)

        # creating an inverted mask to segment out the cloth from the frame
        clothing_mask = cv2.bitwise_not(mask)
        # Segmenting the cloth out of the frame using bitwise and with the inverted mask
        # I.e. clothing is now in 0,0,0 rgb value and other pixels are normal
        res1 = cv2.bitwise_and(img, img, mask=clothing_mask)

        # creating image showing static background frame pixels only for the masked region
        # I.e. everything is all 0,0,0 rgb value except the clothing which is in background rgb
        res2 = cv2.bitwise_and(background, background, mask=mask)

        # OR Gate by combining the two. now the background are normal and the clothing is now background
        final_output = cv2.bitwise_or(res1, res2)

        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        final_output = cv2.resize(final_output, None, fx=0.5, fy=0.5)
        numpy_horizontal = np.vstack((img,final_output))

        cv2.imshow('Joseph Invisible Video', numpy_horizontal)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


