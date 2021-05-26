__author__      = "Anirudh NJ"
__email__ = "anijaya9@gmail.com"
__license__ = "GNU"

import cv2
import sift
import os
from glob import glob
import pickle
import numpy as np

"""
# Geenee Image Recognition Challenge
### Algorithm
The general steps of the algorithm are :    
1. Finding the SIFT keypoints(kp) and the descriptors(d) of the Target image
1. Generate a SIFT feature database (Dictionary in our case) of all the query image.
1. Compare the features of the target image to each of the query features in the database.
(either FLANN or BruteForce matcher can be used)
1. Use statestical methods to find the best matches for a target image.      
### Requirements
1. OpenCV with contrib and non_free modules compiled 
### Result
Depending on the target image we are getting between 90% to 100% accuracy 
according to the calculation metrices given.    
#### Note : There are two different images in the Query folder named 'image10.jpg'.
"""
def get_sift_features(_in_path,_debug_view = False):
    '''
    Generating the SIFT features
    :param _in_path: path to image
    :param _debug_view: -
    :return: keypoints , descriptors
    '''
    img = cv2.imread(_in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp,desc = sift.detectAndCompute(gray, None)

    if _debug_view:
        img = cv2.drawKeypoints(gray, kp, img)
        cv2.imshow('sift_keypoints', img)
        cv2.waitKey(0)

    return kp,desc


def compare_features_flann(_kp1,_dsc1,_kp2,_dsc2,_thres=0):

    # FLANN parameters
    FLANN_INDEX_KDTREE = 3
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(_dsc1, _dsc2, k=2)
    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0,] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            #matches_mask[i] = [1, 0]
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(_kp1) <= len(_kp2):
        number_keypoints = len(_kp1)
    else:
        number_keypoints = len(_kp2)

    return good_points , len(good_points) / number_keypoints * 100


def compare_features_bf(_kp1,_dsc1,_kp2,_dsc2,_thres = 0):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(_dsc1, _dsc2, k=2)
    # Apply ratio test
    good_points = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            #matches_mask[i] = [1, 0]
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(_kp1) <= len(_kp2):
        number_keypoints = len(_kp1)
    else:
        number_keypoints = len(_kp2)

    # print("Keypoints 1ST Image: " + str(len(_kp1)))
    # print("Keypoints 2ND Image: " + str(len(_kp2)))
    # print("GOOD Matches:", len(good_points))
    # print("How good it's the match: ", len(good_points) / number_keypoints * 100)

    return good_points , len(good_points) / number_keypoints * 100

def create_query_database(_path):
    """
    Creating a feature database : a dictionary with filename and SIFT features
    :param _path: path to query path
    :return: image_db
    """
    img_db = {}

    for file in glob(_path):
        kp, desc = sift.get_sift_features(file)
        img_db[os.path.basename(file)] = {"keypoint": kp,
                                              "descriptors": desc}

    # Saving the query db in a file
    #with open('queries.txt', 'wb') as file:
    #    file.write(pickle.dumps(img_db))

    return img_db

def get_best_matches(_result_dict):
    """
    Using statistical methods to remove the best results
    :param _result_dict: query results as a dictionary
    :return: results of thresholding
    """
    mean = np.mean([val for key,val in _result_dict.items()])

    positive = {}
    negative = {}

    for key,val in _result_dict.items() :
        res = (val - mean)
        if  res > mean:
            positive[key] = val
        else:
            negative[key] = val

    return positive

if __name__ == "__main__":

    # Give paths to the Query and Targets folder
    target_path = "query/*.jpg"
    query_path = "images/*.jpg"

    query_db = create_query_database(query_path)

    for files in glob(target_path, recursive=True):
        results = {}
        kb1, des1 = sift.get_sift_features(files)
        print(os.path.basename(files), "\n")
        for keys, values in query_db.items():
            kb2 = values["keypoint"]
            des2 = values["descriptors"]
            good, percentage = sift.compare_features_flann(kb1, des1, kb2, des2)

            results[keys] = percentage

        print(get_best_matches(results))
        print("-----------------")
