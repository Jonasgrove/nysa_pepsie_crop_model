#!/usr/bin/env python

import pandas as pd 
import tensorflow as tf
from sklearn import preprocessing
import numpy as np

class NeuralNet:

    def __init__(self, X, Y, number_of_layers, nodes_per_layer, pred, epoch_num):
        from sklearn.model_selection import train_test_split
        self.input_nodes_num = X.shape[1]
        self.input_layer = tf.keras.layers.Input(shape=(self.input_nodes_num,))
        self.number_of_layers = number_of_layers
        self.nodes_per_layer = nodes_per_layer
        self.epochs = epoch_num
        
        self.pred = pred
        
        self.x_train, self.x_develop, self.y_train, self.y_develop = train_test_split(X, Y, test_size=0.33)
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size=0.01)
        
        self.model = self.train()
        
        self.test_result = self.predict_nn(self.x_test)
        
    def train(self):

        hidden_layer = tf.keras.layers.Dense(self.nodes_per_layer)(self.input_layer)
        hidden_layer = tf.keras.layers.LeakyReLU()(hidden_layer)

        for i in range(self.number_of_layers - 1):
            hidden_layer = tf.keras.layers.Dense(self.nodes_per_layer)(hidden_layer)
            hidden_layer = tf.keras.layers.LeakyReLU()(hidden_layer)


        if self.pred == "r":
            output_layer = tf.keras.layers.Dense(1)(hidden_layer)
            nn_model=tf.keras.models.Model(self.input_layer,output_layer)
            nn_model.compile(loss='mse',optimizer='adam')
            nn_model.fit(self.x_train, self.y_train ,epochs=self.epochs,validation_split=0.5) #Have Keras make a test/validation split for us


        elif self.pred == "m":
            output_layer = tf.keras.layers.Dense(3,activation='softmax')(hidden_layer)
            nn_model = tf.keras.models.Model(input_layer,output_layer)
            nn_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            nn_model.fit(self.x_train, self.y_train ,validation_data=(self.x_develop,self.y_develop_one_hot),epochs=self.epochs)


        elif self.pred == "b":

            output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(hidden_layer)
            nn_model = tf.keras.models.Model(self.input_layer, output_layer)
            nn_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
            nn_model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_develop, self.y_develop)) #Have Keras make a test/validation split for us

        return nn_model

    def predict_nn(self,x):
        
        prediction = self.model.predict(x)
        
        return prediction
        