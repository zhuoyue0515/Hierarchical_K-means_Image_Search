# ！／usr/bin/env python
# -*- coding:utf-8 -*-
# author:Zhuoyue time:2018/5/5

import pickle
import cv2

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.2, edgeThreshold=10)
Feature = []
for i in list(range(1, 51)):
    if i == 26 or i == 38:
        for a in list(range(1, 3)):
            img = cv2.imread('/Users/wangzhuoyue/Desktop/Project2/Data2/server/obj' + str(i) + '_' + str(a) + '.JPG',
                             cv2.IMREAD_GRAYSCALE)
            keypoints = sift.detectAndCompute(img, None)
            a = keypoints[1]
    elif i == 37:
        for b in list(range(1, 5)):
            if b == 4:
                img = cv2.imread(
                    '/Users/wangzhuoyue/Desktop/Project2/Data2/server/obj' + str(i) + '_' + str(b+1) + '.JPG',
                    cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(
                    '/Users/wangzhuoyue/Desktop/Project2/Data2/server/obj' + str(i) + '_' + str(b + 1) + '.JPG',
                    cv2.IMREAD_GRAYSCALE)
            keypoints = sift.detectAndCompute(img, None)
    else:
        for j in list(range(1, 4)):
            img = cv2.imread('/Users/wangzhuoyue/Desktop/Project2/Data2/server/obj'+str(i)+'_'+str(j)+'.JPG',
                             cv2.IMREAD_GRAYSCALE)
            keypoints = sift.detectAndCompute(img, None)
            Feature.append()
with open('Feature', 'wb') as file:
    pickle.dump(Feature, file)
