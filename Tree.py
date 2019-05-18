# ！／usr/bin/env python
# -*- coding:utf-8 -*-
# author:Zhuoyue time:2018/5/5


def hi_kmeans(data, b, depth, icenter, ileaf, center):
    import cv2
    if depth > 0:
        if len(data) < b:
            icenter.append(center)
            for y in list(range(b)):
                hi_kmeans(data, b, depth - 1, icenter, ileaf, center)
        else:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(data[:, 0:128], b, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            icenter.append(center)
            for x in list(range(b)):
                hi_kmeans(data[label.ravel() == x], b, depth-1, icenter, ileaf, center)
    else:
        ileaf.append(data)
    return icenter, ileaf

