import cv2
import numpy as np
from matplotlib import pyplot as plt

class FeatureFinder:

    def __init__(self, img_1_name, img_2_name):

        # load the images
        img_1 = cv2.imread(img_1_name, 0)
        img_2 = cv2.imread(img_2_name, 0)

        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(img_1, None)
        kp2, des2 = orb.detectAndCompute(img_2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        img_3 = cv2.drawMatches(img_1, kp1, img_2, kp2, matches[:10], None, flags=2)
        plt.imshow(img_3)
        plt.show()


if __name__ == "__main__":
    finder = FeatureFinder("test_images/opencv-feature-matching-template.jpg", "test_images/opencv-feature-matching-image.jpg")
