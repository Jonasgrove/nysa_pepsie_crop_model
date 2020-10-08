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

crop_score_model = NeuralNet(X, Y, 5, 256, "r", 10000)
crop_score_model.model.save

crop_score_model.model.save_weights("crop_model_weights_4_256_10000.tf")


#check accuracy
from sklearn.metrics import mean_squared_error

y_pred = crop_score_model.test_result
print(y_pred)

accuracy = mean_squared_error(crop_score_model.y_test ,y_pred)
print("Accuracy", accuracy)

