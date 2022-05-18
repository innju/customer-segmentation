# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:43:44 2022

This script is about the deployment of deep learning model trained.

@author: User
"""

# packages
from tensorflow.keras.models import load_model
import os
import pandas as pd
import csv
import numpy as np
import pickle
from sklearn.impute import KNNImputer


#%% static code
PATH = os.path.join(os.getcwd() ,'data','new_customers.csv') #load new input data
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model_customer_segmentation.h5') 
# get the path of saved model
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'mms.pkl')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'ohe.pkl')# save one hot encoder path
LABEL_ENCODER_SAVE_PATH = os.path.join(os.getcwd(),'le.pkl')# save scaler path

#%% load model
loaded_model = load_model(MODEL_SAVE_PATH)
loaded_model.summary() # view the model again

#%% load pickle files
scaler = pickle.load(open(SCALER_SAVE_PATH,'rb'))
ohe = pickle.load(open(OHE_SAVE_PATH,'rb'))
le = pickle.load(open(LABEL_ENCODER_SAVE_PATH,'rb'))

#%% deployment with new dataset as input
new_input_x = pd.read_csv(PATH)
df = new_input_x
df.drop(['ID'],axis = 1,inplace=True)
#excluding ID column

#%% undone part: preprocessing
# preprocessing new input data
# convert string to number
# since I do the label encoder as below in a complicated way,
# with the NaN remained, I don't know how should I transform only for the test data

c_data= ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score',
          'Var_1','Segmentation'] #all categorical data
for c_data in df.columns:
    series = df[c_data] # a series of data with name of column
    df[c_data] = pd.Series(
       le.transform(series[series.notnull()]),
       index=series[series.notnull()].index)
#getting error:ValueError: y contains previously unseen labels: 'Female'


# If the step above is success,
# then can proceed deal with NaN value in the new dataset using KNN imputer
imputer = KNNImputer()
# by default, n_neighbors=5, metric='nan_euclidean'
df['Ever_Married']= imputer.fit_transform(np.expand_dims(df['Ever_Married'],-1))
df['Graduated'] = imputer.fit_transform(np.expand_dims(df['Graduated'],-1))
df['Profession']= imputer.fit_transform(np.expand_dims(df['Profession'],-1))
df['Family_Size'] = imputer.fit_transform(np.expand_dims(df['Family_Size'],-1))
df['Work_Experience']= imputer.fit_transform(np.expand_dims(df['Work_Experience'],-1))
df['Var_1']= imputer.fit_transform(np.expand_dims(df['Var_1'],-1))

df = scaler.transform(df) #transform the data using min max scaler imported

#%% undone part: prediction

# predicted
predicted = loaded_model.predict(df)

# need to do inverse transform to get back the category name for segmentation
# column first
# and the concatenate the original raw data and predicted dataset together
# before save it to new csv file
predicted = pd.DataFrame(predicted)
new_input_x2 = new_input_x.iloc[:,0:10] #get column from ID to Var_1
overall = pd.concat([new_input_x2, predicted], axis = 1)

# save the info as new csv file
filename='new_customers_predicted.csv'
with open(filename,'w') as csvfile:
    csvwriter = csv.writer(csvfile) 
    
    




