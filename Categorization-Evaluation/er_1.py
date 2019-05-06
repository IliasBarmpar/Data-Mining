import pandas as pd
from ast import literal_eval
import sys
import pdfcrowd
import gmplot
import math
import timeit


def haversine_distance(origin, destination):
	radius = 6371 # FAA approved globe radius in km
	dlat = math.radians(destination[1]-origin[1])
	dlon = math.radians(destination[2]-origin[2])
	a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(origin[1]))* math.cos(math.radians(destination[1])) * math.sin(dlon/2) * math.sin(dlon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = radius * c
	return d

def Distance(trainline, testline):
	Matrix = [[0 for x in range(len(testline)+1)] for y in range(len(trainline)+1)] 

	for i in range(len(trainline)):
		for j in range(len(testline)):
			dist = haversine_distance(testline[j],trainline[i])
			if dist <= 0.200:
				Matrix[i+1][j+1] = Matrix[i][j] + 1
			else:
				Matrix[i+1][j+1] = max(Matrix[i+1][j],Matrix[i][j+1])
	return Matrix
	

trainSet = pd.read_csv(
'train_set.csv', # replace with the correct path
converters={"Trajectory": literal_eval},
index_col='tripId'
)

testSet = pd.read_csv('test_set_a2.csv', converters={"Trajectory": literal_eval}, sep = '\t')
tempSet = testSet.as_matrix()
trainSetArr = trainSet.as_matrix()

for t in range(len(tempSet)):
	start = timeit.default_timer()

	sublist = []
	tem_list = []
	lensub = []
	namelist = []
	for i in range(len(trainSetArr)): 
		dist = Distance(trainSetArr[i][1], tempSet[t][0])
		#diabazoume anapoda ton pinaka
		train = len(dist)-1
		test = len(dist[0])-1
		while(train >= 1 and test >= 1):
			if dist[train][test] == dist[train][test-1]:
				if len(tem_list) > 0:
					namelist.append(trainSetArr[i][0])
					sublist.append(tem_list)
					lensub.append(len(tem_list))
					tem_list = []
				test = test - 1
			elif dist[train][test] == dist[train-1][test]:
				if len(tem_list) > 0:
					namelist.append(trainSetArr[i][0])
					sublist.append(tem_list)
					lensub.append(len(tem_list))
					tem_list = []
				train = train - 1
			else:
				tem_list.append(trainSetArr[i][1][train-1])
				train = train - 1
				test = test - 1


	stop = timeit.default_timer()
	#print "Time"
	#print stop - start

	name = "./test_map"

	test_lon = []
	test_lat = []
	for te in range(len(tempSet[t][0])):
		test_lat.append(tempSet[t][0][te][1])
		test_lon.append(tempSet[t][0][te][2])
	name = name + str(t) + ".html"

	gmap = gmplot.GoogleMapPlotter(max(test_lon),min(test_lat), 10)
	gmap.plot(test_lon, test_lat, "green", edge_width=10)
	gmap.draw(name)
	
	for ma in range(5):
		final_list = []
		name = "./map"
		name = name + str(t) + str(ma)  + ".html"
		index = lensub.index(max(lensub))
		#print max(lensub)
		#print namelist[index]
		final_list = sublist[index][::-1]

		lon = []
		lat = []
		for k in range(len(final_list)):
			lon.append(final_list[k][1])
			lat.append(final_list[k][2])
		
		gmap.plot(test_lon, test_lat, "green", edge_width=10)
		gmap.plot(test_lon, test_lat, "green", edge_width=10)
		gmap.plot(lat, lon, "red", edge_width=10)
		gmap.draw(name)
		sublist.pop(index)
		lensub.pop(index)


