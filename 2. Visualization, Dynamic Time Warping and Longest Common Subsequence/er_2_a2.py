import pandas as pd
from ast import literal_eval
import sys
import pdfcrowd
import gmplot

trainSet = pd.read_csv(
'train_set.csv', # replace with the correct path
converters={"Trajectory": literal_eval},
index_col='tripId'
)

file_arr = trainSet.as_matrix()
planed = []
for i in range(len(file_arr)):
	if(file_arr[i][0] not in planed):
		planed.append(file_arr[i][0])
	else :
		continue
	lon = []
	lat = []
	temp = len(file_arr[i][1])

	for j in range(temp):
		lon.append(file_arr[i][1][j][1])
		lat.append(file_arr[i][1][j][2])
	gmap = gmplot.GoogleMapPlotter(max(lat),min(lon), 10)
	gmap.plot(lat, lon, "green", edge_width=10)
	gmap.draw('./gmap0.html')
	break

for i in range(len(file_arr)):
	if(file_arr[i][0] not in planed):
		planed.append(file_arr[i][0])
	else :
		continue
	lon = []
	lat = []
	temp = len(file_arr[i][1])

	for j in range(temp):
		lon.append(file_arr[i][1][j][1])
		lat.append(file_arr[i][1][j][2])
	gmap = gmplot.GoogleMapPlotter(max(lat),min(lon), 10)
	gmap.plot(lat, lon, "green", edge_width=10)
	gmap.draw('./gmap1.html')
	break

for i in range(len(file_arr)):
	if(file_arr[i][0] not in planed):
		planed.append(file_arr[i][0])
	else :
		continue
	lon = []
	lat = []
	temp = len(file_arr[i][1])

	for j in range(temp):
		lon.append(file_arr[i][1][j][1])
		lat.append(file_arr[i][1][j][2])
	gmap = gmplot.GoogleMapPlotter(max(lat),min(lon), 10)
	gmap.plot(lat, lon, "green", edge_width=10)
	gmap.draw('./gmap2.html')
	break

for i in range(len(file_arr)):
	if(file_arr[i][0] not in planed):
		planed.append(file_arr[i][0])
	else :
		continue
	lon = []
	lat = []
	temp = len(file_arr[i][1])

	for j in range(temp):
		lon.append(file_arr[i][1][j][1])
		lat.append(file_arr[i][1][j][2])
	gmap = gmplot.GoogleMapPlotter(max(lat),min(lon), 10)
	gmap.plot(lat, lon, "green", edge_width=10)
	gmap.draw('./gmap3.html')
	break

for i in range(len(file_arr)):
	if(file_arr[i][0] not in planed):
		planed.append(file_arr[i][0])
	else :
		continue
	lon = []
	lat = []
	temp = len(file_arr[i][1])

	for j in range(temp):
		lon.append(file_arr[i][1][j][1])
		lat.append(file_arr[i][1][j][2])
	gmap = gmplot.GoogleMapPlotter(max(lat),min(lon), 10)
	gmap.plot(lat, lon, "green", edge_width=10)
	gmap.draw('./gmap4.html')
	break
