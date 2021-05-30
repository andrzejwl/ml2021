import numpy as np
import argparse
import glob
import cv2
from PIL import Image

#path = r'/Segmentacja\zd_3.png'

def imageResizeTarget(image):
    maxD = 156
    height, width, channel = image.shape
    aspectRatio = width / height
    if aspectRatio < 1:
        newSize = (int(maxD * aspectRatio), maxD)
        # print("New size:{}".format(newSize))
    else:
        newSize = (maxD, int(maxD / aspectRatio))
        # print("New size:{}".format(newSize))
    image = cv2.resize(image, newSize)
    return image


if __name__ == "__main__":
    # Iterate through contours and filter for ROI
    image_number = 0
    for filepath in glob.iglob('strony_klasera/*.jpg'):
        image = cv2.imread(filepath)
        if image is None:
            print("pies")
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = original[y:y + h, x:x + w]
            height, width, channel = ROI.shape  # segmented image
            if height > 100 and width > 100:  # skip segmented trash
                resized_img = imageResizeTarget(ROI)    #resize
                height, width, channel = resized_img.shape
                crop = 3
                croped_resized_img = resized_img[crop:height - crop, crop:width - crop]     #crop image
                cv2.imwrite("target\\target_IMG_{}.jpg".format(image_number), croped_resized_img)
                image_number += 1
