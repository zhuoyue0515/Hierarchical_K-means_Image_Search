# ！／usr/bin/env python
# -*- coding:utf-8 -*-
# author:Zhuoyue time:2018/5/5

import pickle
import cv2

sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.17, 8)
for i in list(range(1, 51)):
    Feature1 = []
    img = cv2.imread('/Users/wangzhuoyue/Desktop/Project2/Data2/client/obj'+str(i)+'_'+'t1'+'.JPG', cv2.IMREAD_COLOR)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detectAndCompute(img_grey, None)
    Feature1.append(keypoints[1])
    with open('Query' + str(i) + '.pickle', 'wb') as file:
        pickle.dump(Feature1, file)

