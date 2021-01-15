import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage


def extract_features(filename):
    '''Here, we extrac the seven selected features from the input image'''
    
    img = cv2.imread(filename, 3)

    aspect_ratio = img.shape[1] / img.shape[0]                     # aspect ratio                     


    B = np.mean(img[0])
    G = np.mean(img[1])
    R = np.mean(img[2])
    average_perceived_brightness = (math.sqrt(0.241*(R**2) + 0.691*(G**2) + 0.068*(B**2)))  # average percevied brightness


    edges = cv2.Canny(img, 100, 200)
    threshold = 100
    labeled, nr_objects = ndimage.label(edges > threshold) 
    unique, lengths = np.unique(labeled, return_counts=True)
    y, x, g = plt.hist(lengths[1:], bins = 7)
    y_min = np.where(y == y.min())[0][0]
    x_min = np.mean((x[y_min:y_min + 2]))
    y_max = np.where(y == y.max())[0][0]
    x_max = np.mean((x[y_max:y_max + 2]))
    edge_length1 = ((x_max * y.max()) + (x_min * y.min())) / (y.max() + y.min())           # edge length

    #edges = cv2.Canny(img, 100, 200)
    #plt.imshow(edges, cmap = "gray")
    #threshold = 100
    #labeled, nr_objects = ndimage.label(edges > threshold) 
    #unique, lengths = np.unique(labeled, return_counts=True)
    #y, x, g = plt.hist(lengths[1:], bins = 7)
    #y_max = np.where(y == y.max())[0][0]
    #x_max = np.mean((x[y_max:y_max + 2]))
    #edge_length1 = ((x_max * y.max())) / (y.max())



    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    hue, sat, val = cv2.split(hsv)
    hue = hue.flatten()
    y, x, bars = plt.hist(hue, bins = 7)
    y_min = np.where(y == y.min())[0][0]
    x_min = np.mean((x[y_min:y_min + 2]))
    y_max = np.where(y == y.max())[0][0]
    x_max = np.mean((x[y_max:y_max + 2]))
    hue1 = ((x_max * y.max()) + (x_min * y.min())) / (y.max() + y.min())             # hue

    #hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
    #hue, sat, val = cv2.split(hsv)
    #hue = hue.flatten()
    #y, x, bars = plt.hist(hue, bins = 7)
    #y_max = np.where(y == y.max())[0][0]
    #x_max = np.mean((x[y_max:y_max + 2]))
    #hue1 = ((x_max * y.max())) / (y.max())


    area_by_perim = img.shape[0] * img.shape[1] / ((img.shape[0] + img.shape[1]) * 2)             # area by perimeter


    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    contrast = imgGrey.std()                                          # contrast


    orb = cv2.ORB_create(nfeatures = 10000) 
    keypoints, descriptors = orb.detectAndCompute(img, None)
    kp_surf = len(keypoints)                                            # number of keypoints


    return np.array([kp_surf, average_perceived_brightness, contrast, area_by_perim, aspect_ratio, edge_length1, hue1])     # returning the values

