# import numpy as np
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import os, sys
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report
import json


from sklearn.neural_network import MLPClassifier

# def loadGloveModel(gloveFile):
#     # print ("Loading Glove Model")
    
     
#     with open(gloveFile, encoding="utf8" ) as f:
#        content = f.readlines()
#     model = {}
#     for line in content:
#         splitLine = line.split()
#         word = splitLine[0]
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         model[word] = embedding
#     # print ("Done.",len(model)," words loaded!")
#     return model

# file = "/home/vishal/Files/Study/Stream-Project/SEM V/word embedding/glove.840B.300d.txt"
# model= loadGloveModel(file)

vectors_path = '/home/vishal/Files/Study/Stream-Project/SEM V/word embedding/glove.840B.300d.txt'
glove = dict()
with open(vectors_path,'r') as f:
    for l in f:
        if l.strip == '':
            continue
        l = l.strip().split()
        w,v = l[0], np.asarray([float(i) for i in l[1:]])
        glove[w] = v

def sent2vec(sent):
    sentvec = np.zeros(glove['is'].shape)
    data_list= word_tokenize(sent)
    sentlen = len(data_list)
    for w in data_list:
        w = w.strip().lower()
#         print(w,)
        try:
            sentvec += glove[w]
        except KeyError as e:
            sentlen -=1
#         print()
    return sentvec/sentlen

# def process_data(Data):
#     X = [sent2vec(raw['text']) for raw in Data]
#     y = [raw['label'] for raw in Data]
#     return (X,y)        


X=[]
y=[]
i=0
with open("trial_train.txt","rb") as f1:
	for row in f1:
		col= row.strip().split()
		filename=col[0]
		with open("semeval19-hyperpartisan-news-detection-trial-data/"+filename+".xml","rb") as myfile:
			data=myfile.read()

		X.append(sent2vec(data))
		if col[2].strip()=="true":
			y.append(1)
		else:
			y.append(0)	
		i+=1
		if(i%1000==0):
			# break
			print("Train: ",i)	
		# tree = ET.parse("semeval19-hyperpartisan-news-detection-trial-data/"+filename+".xml")
		# root = tree.getroot()
		# data=""
		# for w in root.iter('p'):
		# 	data+=str(w.text)

		# data_list= word_tokenize(data)
		# data_vec=[]
		# for word in data_list:
		# 	if word in model.keys():
		# 		data_vec.append(model[word])

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30, 10, 5), random_state=1)
clf.fit(X, y)

X=[]
y=[]
i=0
with open("trial_dev.txt","rb") as f1:
	for row in f1:
		col= row.strip().split()
		filename=col[0]
		with open("semeval19-hyperpartisan-news-detection-trial-data/"+filename+".xml","rb") as myfile:
			data=myfile.read()

		X.append(sent2vec(data))
		if col[2].strip()=="true":
			y.append(1)
		else:
			y.append(0)	

		i+=1
		if(i%1000==0):
			# break
			print("Dev: ",i)		

predicted_y = clf.predict(X)
print('Test Accuracy is :',accuracy_score(y,predicted_y))
print(classification_report(y, predicted_y))