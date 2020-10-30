#!/usr/bin/env python

import pandas as pd 
import tensorflow as tf
from sklearn import preprocessing
import numpy as np

all_data = pd.read_csv("prep_cereal_data", sep='\t', header=0)


#turn assessment type and variety into dummy vars
at_dums = pd.get_dummies(all_data["Assessment Type"])
all_data = all_data.drop(columns="Assessment Type")
all_data = pd.concat([all_data, at_dums], axis=1)

var_dums = pd.get_dummies(all_data["Variety"])
all_data = all_data.drop(columns="Variety")
all_data = pd.concat([all_data, var_dums], axis=1)

all_data = all_data.drop(columns="Site ID")
all_data = all_data.dropna()
all_data = all_data[all_data["Assessment Score"] != '*']

#split features and target
Y = all_data["Assessment Score"]
X = all_data.drop(columns="Assessment Score")

#scale features
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)

Y = np.array(Y)
Y[Y == ''] = 0.0
Y = Y.astype(np.float)

#make dense network model 
import neural_net

NeuralNet = neural_net.NeuralNet

#crop_score_model = NeuralNet(X, Y, 6, 256, "r", 20)


#check accuracy
from sklearn.metrics import mean_squared_error
'''
y_pred = crop_score_model.test_result
print(y_pred)

accuracy = mean_squared_error(crop_score_model.y_test ,y_pred)
print("Accuracy", accuracy)
'''
test_layers_up = 15
test_nodes_up = 500
out_file = open("nyas_score_model.txt", "w")
all_scores = []

#loop through possible node layer cobos
for i in range(5,test_layers_up):

    for j in range(100,test_nodes_up):
        crop_score_model = NeuralNet(X, Y, i, j, "r", 20)
        y_pred = crop_score_model.test_result
        loss = mean_squared_error(crop_score_model.y_test ,y_pred)
        result = "layers: " + str(i) + "nodes: " + str(j) + "loss: "+ str(loss)
        out_file.write(result)
        print(result)
        all_scores.append(result)

print(all_scores)
out_file.close()