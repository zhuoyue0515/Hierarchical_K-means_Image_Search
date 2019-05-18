# ！／usr/bin/env python
# -*- coding:utf-8 -*-
# author:Zhuoyue time:2018/5/7

from Tree import hi_kmeans
import pickle
import numpy as np
import math
import pandas as pd


def query(b, depth, center, data, index, out, positionx):
    if depth > 1:
        minimum = np.sqrt(np.sum(np.square(data - (center[index][0, :]))))
        position = 0
        for x in list(range(b)):
            dist = np.sqrt(np.sum(np.square(data - (center[index][x, :]))))
            if dist < minimum:
                minimum = dist
                position = x
        index = int(index + 1 + position * ((math.pow(b, depth-1) - 1)/(b - 1)))
        positionx.append(position)
        query(b, depth-1, center, data, index, out, positionx)
    else:
        minimum = np.sqrt(np.sum(np.square(data - (center[index][0, :]))))
        position = 0
        for x in list(range(b)):
            dist = np.sqrt(np.sum(np.square(data - (center[index][x, :]))))
            if dist < minimum:
                minimum = dist
                position = x
        positionx.append(position)
        loc = positionx[0] + 1
        for i in positionx[1:len(positionx)]:
            loc = loc * b - (b - i - 1)
        out.append(loc - 1)
    return out


b1 = 5
depth1 = 7
file = open('Feature.pickle', 'rb')
idf = np.zeros(int(math.pow(b1, depth1)))
tf_data = np.zeros((int(math.pow(b1, depth1)), 51))
Feature = pickle.load(file)
center1, leaf1 = hi_kmeans(Feature, b1, depth1, [], [], [])
for v in list(range(int(math.pow(b1, depth1)))):
    idf[v] = math.log2(50/(len(set(list(leaf1[v][:, 128])))))
    for w in list(leaf1[v][:, 128]):
        w = int(w)
        tf_data[v, w] = tf_data[v, w] + 1
for u in list(range(51)):
    tf_data[:, u] = tf_data[:, u]/sum(tf_data[:, u])
count1 = 0
count5 = 0
for o in list(range(1, 51)):
    file1 = open('Query' + str(o) + '.pickle', 'rb')
    Query = pickle.load(file1)
    Query[0] = pd.DataFrame(Query[0])
    Query[0] = Query[0].sample(frac=0.5)
    Query[0] = np.array(Query[0])
    tf = np.zeros((int(math.pow(b1, depth1)), 51))
    rank = np.zeros((int(math.pow(b1, depth1)), 51))
    paiming = np.zeros(51)
    for p in list(range(np.size(Query[0], 0))):
        p = int(p)
        a = query(b1, depth1, center1, Query[0][p, :], 0, [], [])
        tf[a, :] = tf[a, :] + tf_data[a, :]
    for q in list(range(51)):
        rank[:, q] = tf[:, q] * idf
    for z in list(range(51)):
        paiming[z] = sum(rank[:, z])
    top = np.argsort(-paiming)
    if top[0] == o:
        count1 = count1 + 1
    for r in list(range(5)):
        if top[r] == o:
            count5 = count5 + 1
rate1 = count1/50
rate5 = count5/50
print(rate1)
print(rate5)
