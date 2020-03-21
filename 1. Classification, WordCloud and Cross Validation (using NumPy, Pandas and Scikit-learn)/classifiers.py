from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from nltk.stem import PorterStemmer, WordNetLemmatizer
import math
import operator


def outputfile(newlist, newda):
    newda["ID"].append(newlist[0])
    newda["Predicted_Category"].append(newlist[1])

class MyDict():
    """docstring for MyDict"""
    def __init__(self,x,y):
        self.content = x
        self.category = y
        
class KNN():
    """docstring for KNN"""
    def __init__(self,train_data):
        self.predictions = []
        self.id = []
        self.category = []
        self.result_fit = []
        self.train_data = train_data
        
    def fit(self,train_data, X, y, trainingSet=[] , testSet=[]):
        self.id = train_data["Id"]

        ###################### START TITLE###################
        #alltogether = ['A' for pos in range(len(train_data["Content"]))]
        #for k in range(len(train_data["Content"])):
            #print k
            #tem_list = []
        #    alltogether[k] = train_data["Content"][k]
        #    for i in range(3):
        #        alltogether[k] = alltogether[k] + train_data["Title"][k]
        ###################### END TITLE###################

        #count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
        #X = count_vectorizer.fit_transform(alltogether)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
        X_train.todense()
        X_array = X_train.toarray()

        X_test.todense()
        test_array = X_test.toarray()

        # ---------------------------------------- #

        for x in range(len(X_array)):#for every line
            trainingSet.append(X_array[x])
            NewLine = MyDict(X_array[x],y_train[x])
            self.result_fit.append(NewLine)

        self.category = y_test
        for x in range(len(test_array)):
            testSet.append(test_array[x])

    #calculate the distance
    def EuclideanDistance(self,instance1, instance2, length):
        distance = 0
        for x in range(length):#sum
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)
 
    def FindNeig(self, trainingSet, testInstance, k):
        
        distances = []#all distances

        length = len(testInstance)-1
        for x in range(len(self.result_fit)):#for every line of the trainset
            #bres tin apostasi aytoy toy test apo kaaaathe simeio tou train
            dist = self.EuclideanDistance(testInstance, self.result_fit[x].content, length)#calculate the distance
            distances.append((self.result_fit[x], dist))#put at the list
        #sort the list  
        distances.sort(key=operator.itemgetter(1))

        #find the k closest neighboors
        neighbors = []
        for x in range(k):
            #bres toys k pio kontinous
            neighbors.append(distances[x])
        return neighbors#aytoi einai oi geitones soy
 
    def Predict(self, neighbors):
        Votes = {}#majority voting 
        w = 1
        #for every neighboor
        for x in range(len(neighbors)):
            response = neighbors[x][0].category
            if response in Votes:#if it exists
                Votes[response] += (1/w)*neighbors[x][1]
            else:
                Votes[response] = (1/w)*neighbors[x][1]
            w += 1
        #find the majority vote
        sortedVotes = sorted(Votes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]
    
    def score(self,testSet, predictions):
        corr_pred = 0
        for x in range(len(testSet)):
            if self.category[x] == self.predictions[x]:
                corr_pred += 1
        percent = (corr_pred/float(len(testSet))) * 100.0
        return percent
        

###### epeksergasia dedomenon ###################
#train_data = pd.read_csv('train_set.csv', sep="\t")
#train_data = train_data[0:100]
#le = preprocessing.LabelEncoder()
#le.fit(train_data["Category"])
#y = le.transform(train_data["Category"])
#stopwords = set(ENGLISH_STOP_WORDS)

#vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf = True)
#X = vectorizer.fit_transform(train_data['Content'])

############ klisi knn ###########
#knn = KNN(train_data)
#trainingSet = []
#testSet = []
#k = 5
#knn.fit(knn.train_data, trainingSet, testSet)

############### Predict ####################
#for x in range(len(testSet)):
#    neighbors = knn.FindNeig(trainingSet, testSet[x], k)
#    result = knn.Predict(neighbors)
#    print testSet[x]
#    knn.predictions.append(result)


############### file #######################


def outputfile1(Bayes_list, Forest_list, SVM_list, KNN_list, My_list):
    newda = {"Statistic Measure" : [], "Naive Bayes" : [], "Random Forest" : [], "SVM" : [], "KNN" :[], "My Method": [] }
    Statistic_list = ['Accuracy','Precision', 'Recall','F-Measure']

    for i in Statistic_list:
        newda["Statistic Measure"].append(i)
    for i in Bayes_list:
        newda["Naive Bayes"].append(i)
    for i in Forest_list:
        newda["Random Forest"].append(i)
    for i in SVM_list:
        newda["SVM"].append(i)
    for i in KNN_list:
        newda["KNN"].append(i)
    for i in My_list:
        newda["My Method"].append(i)

    df = pd.DataFrame(newda, columns=["Statistic Measure", "Naive Bayes", "Random Forest", "SVM", "KNN", "My Method"])
    df.to_csv('EvaluationMetric_10fold.csv', sep = '\t')
    file = pd.read_csv('EvaluationMetric_10fold.csv', sep = '\t')

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def stemm(data):
    new_data = {"RowNum": [], "Id": [], "Title" : [], "Content": [], "Category" : []}
    final = []
    stemmer = PorterStemmer()
    line = 0

    for i in data["Content"]:
        newdata = ' '
        for word in i.split():
            if(is_ascii(word)):
                temp = stemmer.stem(word)
                newdata += temp
                newdata += ' '  
        final.append([newdata])
        new_data["RowNum"].append(line)
        new_data["Id"].append(data["Id"][line])
        new_data["Title"].append(data["Title"][line])
        new_data["Content"].append(final)
        new_data["Category"].append(data["Category"][line])
        line +=1

    #new_data = final
    return new_data

def addmore(data, stop):
    for i in data: #for every content
        newwords = []
        for wor in i.split():#for every word
            for k in stop:#for every stopword 
                if wor in k:
                    if abs(len(wor) - len(k)) < 5:
                        newwords.append(wor)
                if k in wor:
                    if abs(len(wor) - len(k)) < 5:
                        newwords.append(wor)
        for wooo in newwords:
            stop.add(wooo)

train_data = pd.read_csv('input/train_set.csv', sep="\t")
train_data = train_data[0:100]

###################### START TITLE ###################
alltogether = ['A' for pos in range(len(train_data["Content"]))]
for k in range(len(train_data["Content"])):
    alltogether[k] = train_data["Content"][k]
    for i in range(3):
        alltogether[k] = alltogether[k] + train_data["Title"]

new_train = stemm(train_data)

df = pd.DataFrame(new_train, columns=["RowNum","Id","Title","Content","Category"])
df.to_csv('NewData.csv', sep = '\t')
new_train_data = pd.read_csv('NewData.csv', sep = '\t')
new_train_data = train_data

le = preprocessing.LabelEncoder()
y = le.fit_transform(new_train_data["Category"])

stopwords = set(ENGLISH_STOP_WORDS)
addmore(new_train_data, stopwords)

vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf = True)
X = vectorizer.fit_transform(new_train_data["Content"])

array = [2,3,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

scoring = {  'Precision': make_scorer(precision_score, average = 'macro'),
             'Recall': make_scorer(recall_score,    average = 'macro'),
             'F-Measure': 'f1_macro',
             'Accuracy': 'accuracy' }

clf = svm.SVC(kernel='linear', C=1, random_state=0)
svm_scores = ['Accuracy']

svm_final = {}
sum_score_acc = []
sum_score_f1 = []
sum_score_rec = []
sum_score_pre = []

for x in array:
    lsa = TruncatedSVD(n_components = x)
    X_new = lsa.fit_transform(X)
    scores = cross_validate(clf, X_new, y, scoring = scoring, cv = 10)
    sum_score_acc.append(scores['test_Accuracy'].mean())
    sum_score_pre.append(scores['test_Precision'].mean())
    sum_score_f1.append(scores['test_F-Measure'].mean())
    sum_score_rec.append(scores['test_Recall'].mean())

svm_final["Accuracy"] = np.mean(sum_score_acc)*100
svm_final["Recall"] = np.mean(sum_score_rec)*100
svm_final["F1-Measure"] = np.mean(sum_score_f1)*100
svm_final["Precision"] = np.mean(sum_score_pre)*100

SVM_list = []
SVM_list.append(svm_final["Accuracy"])
SVM_list.append(svm_final["Recall"])
SVM_list.append(svm_final["F1-Measure"])
SVM_list.append(svm_final["Precision"])

clf = RandomForestClassifier()
rf_scores = []

rf_final = {}
rf_score_acc = []
rf_score_f1 = []
rf_score_rec = []
rf_score_pre = []

for x in array:
    lsa = TruncatedSVD(n_components = x)
    X_new = lsa.fit_transform(X)
    scores = cross_validate(clf, X_new, y, scoring = scoring, cv = 10)
    rf_score_acc.append(scores['test_Accuracy'].mean())
    rf_score_pre.append(scores['test_Precision'].mean())
    rf_score_f1.append(scores['test_F-Measure'].mean())
    rf_score_rec.append(scores['test_Recall'].mean())

rf_final["Accuracy"] = np.mean(sum_score_acc)*100
rf_final["Recall"] = np.mean(sum_score_rec)*100
rf_final["F1-Measure"] = np.mean(sum_score_f1)*100
rf_final["Precision"] = np.mean(sum_score_pre)*100

Forest_list = []
Forest_list.append(rf_final["Accuracy"])
Forest_list.append(rf_final["Recall"])
Forest_list.append(rf_final["F1-Measure"])
Forest_list.append(rf_final["Precision"])

array = [2,3,5,10,15,20,25,30] 

clf = MultinomialNB()
mnb_scores = []
nb_final = {}
nb_score_acc = []
nb_score_f1 = []
nb_score_rec = []
nb_score_pre = []

for x in array:
    lsa = NMF(n_components = x)
    X_new = lsa.fit_transform(X)
    scores = cross_validate(clf, X_new, y, scoring = scoring, cv = 10)
    nb_score_acc.append(scores['test_Accuracy'].mean())
    nb_score_pre.append(scores['test_Precision'].mean())
    nb_score_f1.append(scores['test_F-Measure'].mean())
    nb_score_rec.append(scores['test_Recall'].mean())

nb_final["Accuracy"] = np.mean(sum_score_acc)*100
nb_final["Recall"] = np.mean(sum_score_rec)*100
nb_final["F1-Measure"] = np.mean(sum_score_f1)*100
nb_final["Precision"] = np.mean(sum_score_pre)*100

Bayes_list = []
Bayes_list.append(nb_final["Accuracy"])
Bayes_list.append(nb_final["Recall"])
Bayes_list.append(nb_final["F1-Measure"])
Bayes_list.append(nb_final["Precision"])


############ klisi knn ###########
knn = KNN(new_train_data)
trainingSet = []
testSet = []
k = 5
knn.fit(knn.train_data,X, y, trainingSet, testSet)

############### Predict ####################
for x in range(len(testSet)):
    neighbors = knn.FindNeig(trainingSet, testSet[x], k)
    result = knn.Predict(neighbors)
    knn.predictions.append(result)
accuracy = knn.score(testSet, knn.predictions)
print accuracy

#data = pd.read_csv('train_set.csv',sep='\t')
newda = {'ID' : [], 'Predicted_Category': []}

for x in range(len(testSet)):
    newlist = []
    newlist.append(knn.id[x])
    newlist.append(knn.predictions[x])
    outputfile(newlist, newda)

df = pd.DataFrame(newda, columns=['ID', 'Predicted_Category'])
df.to_csv('testSet_Categories.csv', sep = '\t')
file = pd.read_csv('testSet_Categories.csv', sep = '\t')

KNN_list = [accuracy,' ', ' ', ' ']
My_list = [accuracy,' ',' ',' ']

outputfile1(Bayes_list, Forest_list, SVM_list, KNN_list, My_list)
