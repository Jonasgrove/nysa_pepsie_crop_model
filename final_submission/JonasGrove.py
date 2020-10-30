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

samp_id = all_data["ID"]
all_data = all_data.drop(columns="Site ID")
all_data = all_data.dropna()
all_data = all_data[all_data["Assessment Score"] != '*']

#split features and target
Y = all_data["Assessment Score"]
X = all_data.drop(columns="Assessment Score")

nodes_per_layer = 259


#scale features
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)

Y = np.array(Y)
Y[Y == ''] = 0.0
Y = Y.astype(np.float)

input_nodes_num = X.shape[1]
print("####",X.shape[1])
input_layer = tf.keras.layers.Input(shape=(input_nodes_num,))

hidden_layer = tf.keras.layers.Dense(nodes_per_layer)(input_layer)
hidden_layer = tf.keras.layers.LeakyReLU()(hidden_layer)

for i in range(4):
    hidden_layer = tf.keras.layers.Dense(nodes_per_layer)(hidden_layer)
    hidden_layer = tf.keras.layers.LeakyReLU()(hidden_layer)


output_layer = tf.keras.layers.Dense(1)(hidden_layer)
nn_model=tf.keras.models.Model(input_layer,output_layer)
nn_model.compile(loss='mse',optimizer='adam')

print("here")
nn_model.load_weights('/Users/jonasgrove/Projects/pepsi_cereal/final_submission/crop_model_weights.tf')

y_pred = nn_model.predict(X)

all_data.to_csv('all_data_XY.csv', index=False)

out_file = open("all_test_model.txt", "w")
for indv_id, line in zip(samp_id,y_pred):
  # write line to output file
  new_line = str(indv_id) + "\t" + str(line[0])
  out_file.write(new_line)
  out_file.write("\n")

out_file.close()