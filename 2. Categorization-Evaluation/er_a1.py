import pandas as pd
import numpy as np
from ast import literal_eval
import sys
import pdfcrowd
import gmplot
import timeit
from math import radians, cos, sin, asin, sqrt
from dtw import dtw 

def haversine(pointx, pointy):
    lon1 = pointy[1]
    lat1 = pointy[2]
    lon2 = pointx[1]
    lat2 = pointx[2]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

trainSet = pd.read_csv( 'train_set.csv', converters={"Trajectory": literal_eval}, index_col='tripId')
testSet  = pd.read_csv('test_set_a1.csv', converters={"Trajectory": literal_eval}, sep = '\t')

#trainSet = trainSet[0:50]

testSetArr = testSet.as_matrix()
trainSetArr = trainSet.as_matrix()

start = timeit.default_timer()
JP_ID = []
DTW = [[]*5 for i in range(len(testSetArr))]
ind = [[]*5 for i in range(len(testSetArr))]
#For each trip in the test set
for t in range(len(testSetArr)):
    names = []
    distances = []
    sortDist = []
    #Calculate the distance using the dtw algorithm
    for i in range(len(trainSetArr)): 
        dist, cost, acc, path = dtw(trainSetArr[i][1], testSetArr[t][0], dist = haversine)
        distances.append(dist)
    #Sort the distances you found and keep the shortest five distances and their indexes
    sortDist = sorted(distances, key=float)
    for q in range(0, 5):
        ind[t].append(distances.index(sortDist[q]))
        DTW[t].append(sortDist[q]) 
    
    #Draw the test map
    name = "./testA1_map"
    name = name + str(t) + ".html"
    testA1_lon = []
    testA1_lat = []
    for j in range(len(testSetArr[t][0])):
        testA1_lon.append(    testSetArr[t][0][j][1] )
        testA1_lat.append(    testSetArr[t][0][j][2] )
    gmap = gmplot.GoogleMapPlotter(max(testA1_lat),min(testA1_lon), 10)
    gmap.plot(testA1_lat, testA1_lon, "green", edge_width=10)
    gmap.draw(name)
    
    #Draw the maps closest to the current test map
    for n in range(5):
        name = "./mapA1"
        name = name + '_' + str(t) + '_' + str(n) + ".html"
        lon = []
        lat = []
        for m in range(len(trainSetArr[ ind[t][n] ][1])):
            lon.append(    trainSetArr[ ind[t][n] ][1][m][1] )
            lat.append(    trainSetArr[ ind[t][n] ][1][m][2] )
        gmap = gmplot.GoogleMapPlotter(max(lat),min(lon), 10)
        gmap.plot(lat,lon , "green", edge_width=10)
        gmap.draw(name)
stop = timeit.default_timer()
