#Based on the 'Brute Force matching base on ORB Algorithm' from OpenCV.org
#   pattern-image-matcher.matchconfidence(img1,img2) compares two images and
#   reports if they are the same image (or contains the same image) by returning 1.
#Author : Andre Bertomeu
#Date Updated : 08/01/2020

import numpy as np
import cv2
from matplotlib import pyplot as plt


debug = 0 #0 off, 1 on

img1 = cv2.imread('testimg\856-0656-B_VisProAdventure2.png',0)          # queryImage
img2 = cv2.imread('856-0656-B_VisProAdventure1.png',0) # trainImage

if debug:
    print("--Debug--")
    print(img1.shape)
    print(img2.shape)
    print(img1.shape[0]/img2.shape[0])


def findLargerImg(image1,image2):
    s1 = image1.shape
    s2 = image2.shape
    if s1>s2:
        return 1
    else:
        return 2
    
def scaleDownImg(largerimg,scale):
    height = largerimg.shape[0]*scale
    width = largerimg.shape[1]*scale
    smallerimg = cv2.resize(largerimg,(width,height))
    return smallerimg

def filterImg(image):
    # make it grayscale
    #Gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Make canny Function
    canny=cv2.Canny(image,100,140)
    return canny

def matchconfidence(img1,img2):
    '''given 2 images, returns 1 if confident that they are the same image, 0 if not'''
    # Initiate ORB detector
    orb = cv2.ORB_create()

    #canny filter images
    img1_filtered = filterImg(img1)
    img2_filtered = filterImg(img2)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1_filtered,None)
    kp2, des2 = orb.detectAndCompute(img2_filtered,None)

    # create BFMatcher object
    # old bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher()

    # Match descriptors.
    #old matches = bf.match(des1,des2)
    matches = bf.knnMatch(des1,des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # Sort them in the order of their distance.
    # old matches = sorted(matches, key = lambda x:x.distance)

    if debug:
    #debug prints
        print("--Debug--")
        print(len(matches))
        for i in range(10):
            print("\n".format(type(matches[i])))

    #Draw first 10 matches.
    #old img3 = cv2.drawMatches(img1_filtered,kp1,img2_filtered,kp2,good,None, flags=2)

    # cv2.drawMatchesKnn expects list of lists as matches.
    if debug:
        img3 = cv2.drawMatchesKnn(img1_filtered,kp1,img2_filtered,kp2,good,None,flags=2)
        plt.imshow(img3),plt.show()

    if debug:
        print("--Debug--")
        print(len(good))
        
    if len(good)>30:
        return 1
    else:
        return 0
    

thelargerimg = findLargerImg(img1,img2)
if debug:
    print("--Debug--")
    print("the larger image is img{}".format(thelargerimg))

if thelargerimg>1:
    #second image is larger
    scaledownby = round((img1.shape[0]/img2.shape[0])*100)/100
    imgheight = int(img2.shape[0]*(scaledownby))
    imgwidth = int(img2.shape[1]*(scaledownby))
    img2 = cv2.resize(img2,(imgwidth,imgheight))
    
else:
    #first image is larger
    scaledownby = round((img2.shape[0]/img1.shape[0])*100)/100
    imgheight = int(img1.shape[0]*(scaledownby))
    imgwidth = int(img1.shape[1]*(scaledownby))
    img1 = cv2.resize(img1,(imgwidth,imgheight))

if debug:
    print("scaling down img{} by {}%".format(thelargerimg,scaledownby*100))

if __name__=='__main__':
    if matchconfidence(img1,img2):
        print("Image are matching")
    else:
        print("Images not matching")
