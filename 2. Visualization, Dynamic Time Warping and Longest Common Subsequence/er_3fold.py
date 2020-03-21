from random import seed
from random import randrange
import pandas as pd
from ast import literal_eval
import sys
import pdfcrowd
import gmplot
import math
import timeit 
from dtw import dtw 
import operator
import numpy as np



def haversine_distance(origin, destination):
	radius = 6371 # FAA approved globe radius in km
	dlat = math.radians(destination[1]-origin[1])
	dlon = math.radians(destination[2]-origin[2])
	a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(origin[1]))* math.cos(math.radians(destination[1])) * math.sin(dlon/2) * math.sin(dlon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = radius * c
	return d

def FindNeig(trainingSet, testInstance, k=5):
	distances = []
	for x in range(len(trainingSet)):
		dist, cost, acc, path = dtw(trainingSet[x][1], testInstance[1], dist = haversine_distance)
		distances.append((trainingSet[x][0], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x])
	return neighbors

def Predict(neighbors):
	Votes = {}#majority voting 
	w = 1
	for x in range(len(neighbors)):
		response = neighbors[x][0]
		if response in Votes:#if it exists
			#Votes[response] += 1
			Votes[response] += (1/w)*neighbors[x][1]
		else:
			#Votes[response] =1
			Votes[response] = (1/w)*neighbors[x][1]
		w += 1
	#find the majority vote
	sortedVotes = sorted(Votes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


# Split a dataset into k folds
def cross_validation_split(dataset, folds=10):
	dataset_split = []
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = []
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# test cross validation split
seed(1)
trainSet = pd.read_csv(
'train_set.csv', # replace with the correct path
converters={"Trajectory": literal_eval},
index_col='tripId'
)

trainArr = trainSet.as_matrix()

folds = cross_validation_split(trainArr, 10)

corr = 0
for i in range(len(folds)):
	train = []
	test = []
	for j in range(len(folds)):
		if i != j:
			train.extend(folds[j])
	test = folds[i]
	for k in range(len(test)):
		neig = FindNeig(train, test[k])
		pred = Predict(neig)
		if pred == test[k][0]:
			corr = corr + 1
print corr / len(trainArr) * 100
