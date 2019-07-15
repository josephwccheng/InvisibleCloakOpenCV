import numpy as np
import cv2
import matplotlib.pyplot as plt

# import argparse
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help="path to the image")
# args = vars(ap.parse_args())
#
# # load the image
# image = cv2.imread(args["image"])
# cv2.imshow('image for testing', image)
# cv2.waitKey(0)

# load the image
image = cv2.imread("ImageForColor.PNG")
# cv2.imshow('image for testing', image)
# cv2.waitKey(0)

imgPlot = plt.imshow(image)
plt.show()

# define the list of boundaries
boundaries = [
    ([30, 13, 9], [115, 84, 25])
]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # show the images
    #cv2.imshow("images", np.hstack([image, output]))
cv2.imshow("image", output)
#cv2.imshow("images", np.hstack([image, output]))
cv2.waitKey(0)