#Decision Tree implementation of SciKit learn using ID3
#sahils (Sahil Sachdeva)
#vmanocha (Varun Manocha)

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import subprocess
from numpy import array
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder


attributes = ["Occupied", "Price", "Music", "Location", "VIP", "Favorite Beer", "Enjoy"]
df=pd.read_csv('training_data_2.csv',skipinitialspace=True) #read training data

#Start encoding the categorical data
lb_make = LabelEncoder()
df_labeled=df.apply(lb_make.fit_transform)
y1=df_labeled[df_labeled.columns[-1:]]
X1_encode=df_labeled[df_labeled.columns[:-1]]
y1=y1[:-1]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(X1_encode)
predict=onehot_encoded[-1:,:]
onehot_encoded=onehot_encoded[:-1]

#Use ID3 Decision Tree from sklearn
treeDecision = tree.DecisionTreeClassifier(criterion='entropy',random_state=50)
treeDecision = treeDecision.fit(onehot_encoded, y1)
output=treeDecision.predict(predict)
tree.export_graphviz(treeDecision,out_file='tree2.dot')
command = ["dot", "-Tpng", "tree2.dot", "-o", "dt2.png"]
subprocess.check_call(command)
inverted = lb_make.inverse_transform(output)
print("Predicted outcome for given case is",)
print(inverted[0])