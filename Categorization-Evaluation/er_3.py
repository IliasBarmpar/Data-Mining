import pandas as pd
from ast import literal_eval
import sys
import pdfcrowd
import gmplot
import math
from dtw import dtw 
import operator

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
		#print trainingSet[x][1]
		dist, cost, acc, path = dtw(trainingSet[x][1], testInstance, dist = haversine_distance)
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


trainSet = pd.read_csv(
'train_set.csv', # replace with the correct path
converters={"Trajectory": literal_eval},
index_col='tripId'
)

testSet = pd.read_csv('test_set_a2.csv', converters={"Trajectory": literal_eval}, sep = '\t')
tempSet = testSet.as_matrix()
trainSetArr = trainSet.as_matrix()


data = {'Test_Trip_ID': [], 'Predicted_JourneyPatternID': []}
 
for i in range(len(tempSet)):
	#print tempSet[i][0]
	neig = FindNeig(trainSetArr,tempSet[i][0])
	prediction = Predict(neig)
	data['Test_Trip_ID'].append(i)
	data['Predicted_JourneyPatternID'].append(prediction)

df = pd.DataFrame(data, columns=['Test_Trip_ID', 'Predicted_JourneyPatternID'])

df.to_csv('testSet_JourneyPatternIDs.csv')



