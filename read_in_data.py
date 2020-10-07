#!/usr/bin/env python


import pandas as pd 
import tensorflow as tf
from sklearn import preprocessing
import numpy as np

all_data_xls = pd.read_excel("nyas-challenge-2020-data.xlsx", sheet_name=[0,1,2])

sheet_1 = all_data_xls[0]
sheet_2 = all_data_xls[1]
sheet_3 = all_data_xls[2]

#turn assessment type into dummy vars
at_dums = pd.get_dummies(sheet_1["Assessment Type"])
#drop assess column 
sheet_1 = sheet_1.drop(columns="Assessment Type")

#drop the target
target = sheet_1["Assessment Score"]
sheet_1 = sheet_1.drop(columns="Assessment Score")

sheet_1 = pd.merge(sheet_1, at_dums, left_index=True, right_index=True)

features = pd.merge(sheet_1, sheet_2, on=["Site ID"])


sheet_1["Site ID"] = sheet_1["Site ID"].replace(to_replace="Year", "")
print(sheet_1)

'''
#one hot encoding
assess_type = sheet_1["Assessment Type"]

print(assess_type)

assess_type = np.array(list(set(assess_type.astype(str).values))).reshape(-1,1)
print(assess_type.shape)
print(assess_type)

ohe = preprocessing.OneHotEncoder()

ohe.fit(assess_type)
assess_type = ohe.transform(assess_type).todense()


print(assess_type.shape)
print(assess_type)


'''


