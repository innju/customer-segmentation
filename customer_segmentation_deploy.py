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
import pickle

#%% static code
PATH = os.path.join(os.getcwd() ,'data','new_customers.csv') #load new input data
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model_customer_segmentation.h5') 
# get the path of saved model
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'mms.pkl')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'ohe.pkl')# saved one hot encoder path
# label encoder saved path:
SAVE_PATH_GENDER = os.path.join(os.getcwd(),'gender.pkl')
SAVE_PATH_EMARRIED = os.path.join(os.getcwd(),'emarried.pkl')
SAVE_PATH_GRADUATED = os.path.join(os.getcwd(),'graduated.pkl')
SAVE_PATH_PROFESSION = os.path.join(os.getcwd(),'profession.pkl')
SAVE_PATH_SSCORE = os.path.join(os.getcwd(),'sscore.pkl')
SAVE_PATH_VAR1 = os.path.join(os.getcwd(),'var1.pkl')
SAVE_PATH_SEGMENTATION = os.path.join(os.getcwd(),'segmentation.pkl')
SAVE_PATH_KNN_IMPUTER = os.path.join(os.getcwd(),'knn_imputer.pkl')
NEW_FILE_SAVE_PATH = os.path.join(os.getcwd(),'data','new_customers_predicted.csv')

#%% functions
def categorical_label_deploy(col_encoder,col_name):
    le= col_encoder
    temp = new_input_x[col_name]
    temp[temp.notnull()]=le.transform(temp[temp.notnull()])

#%% load model
loaded_model = load_model(MODEL_SAVE_PATH)
loaded_model.summary() # view the model again

#%% load pickle files
scaler = pickle.load(open(SCALER_SAVE_PATH,'rb'))
ohe = pickle.load(open(OHE_SAVE_PATH,'rb'))
# load label encoder path:
le_gender = pickle.load(open(SAVE_PATH_GENDER,'rb'))
le_emarried = pickle.load(open(SAVE_PATH_EMARRIED,'rb'))
le_graduated = pickle.load(open(SAVE_PATH_GRADUATED,'rb'))
le_profession = pickle.load(open(SAVE_PATH_PROFESSION,'rb'))
le_sscore = pickle.load(open(SAVE_PATH_SSCORE,'rb'))
le_var1 = pickle.load(open(SAVE_PATH_VAR1,'rb'))
le_segmentation = pickle.load(open(SAVE_PATH_SEGMENTATION,'rb'))
knn_imputer = pickle.load(open(SAVE_PATH_KNN_IMPUTER,'rb'))

#%% 1) Load data 
# load in new dataset
ori_x = pd.read_csv(PATH)
#excluding ID column
new_input_x = (ori_x.loc[:, ori_x.columns != 'ID'])

# 2) Data inspection
new_input_x.isnull().sum()
# missing data for Ever_Married(50), Graduated(24),Profession(38),
# Work_Experience(269), Family_Size(113), Var_1(32), Segmentation (2627)
new_input_x.info()
# categorical data identified: 'Gender', 'Ever_Married', 'Graduated', 
# 'Profession', 'Spending_Score', 'Var_1'

# 3) data preprocessing
# convert string to number for categorical data in new input dataset
categorical_label_deploy(col_encoder=le_gender, col_name='Gender')
categorical_label_deploy(col_encoder=le_emarried, col_name='Ever_Married')
categorical_label_deploy(col_encoder=le_graduated, col_name='Graduated')
categorical_label_deploy(col_encoder=le_profession, col_name='Profession')
categorical_label_deploy(col_encoder=le_sscore, col_name='Spending_Score')
categorical_label_deploy(col_encoder=le_var1, col_name='Var_1')

# convert categorical columns above to numeric data type before imputation
cat_data= ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score',
          'Var_1'] #all categorical data
new_input_x[cat_data] = new_input_x[cat_data].apply(pd.to_numeric, errors='coerce')

# recheck data type to ensure successful data conversion
new_input_x.info()
# no more object data type

# then can proceed to deal with NaN value in the new dataset using KNN imputer
# by default, n_neighbors=5, metric='nan_euclidean'
new_input_x2 = knn_imputer.transform(new_input_x)
new_input_x2 = pd.DataFrame(new_input_x2)
# transform the data using min max scaler imported
# get the features by exclude the last column (target: segmentation)
X_data = scaler.transform(new_input_x2.iloc[:,:-1])


#%% Prediction
predicted = loaded_model.predict(X_data)
# need to do inverse transform to get back the number labelled for segmentation
predicted2 = ohe.inverse_transform(predicted)
print(predicted2.dtype) #float
# change to int in order to perform inverse and transform using label encoder
predicted2 = le_segmentation.inverse_transform(predicted2.astype(int))
# replace segmentation column from original data with prediction
ori_x['Segmentation'] = predicted2
ori_x.to_csv(NEW_FILE_SAVE_PATH,index=False)

