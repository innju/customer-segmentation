# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:52:24 2022

This script is used to train deep learning model for customer segmentation
purpose based on their characteristics.

@author: User
"""

# Packages
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import Input,Sequential
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import OneHotEncoder
import pickle


#%% Static code

PATH = os.path.join(os.getcwd(),'data','train.csv') #load data
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model_customer_segmentation.h5') #save model
LOG_SAVE_PATH = os.path.join(os.getcwd(),'log_customer_segmentation') 
log_customer_segmentation = os.path.join(
    LOG_SAVE_PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')# save scaler path
OHE_SAVE_PATH = os.path.join(os.getcwd(),'ohe.pkl')# save encoder path
LABEL_ENCODER_SAVE_PATH = os.path.join(os.getcwd(),'le.pkl')# save encoder path

#%% Function defined
# generate classification report
def report_generation(x_test,y_test):
    pred_x = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(pred_x,axis=1)
    cm = confusion_matrix(y_true,y_pred)
    cr = classification_report(y_true,y_pred)
    print(cr)
    print(cm)
    
#%% EDA
# 1) Load data
df= pd.read_csv(PATH)

# 2) Data inspection/visualization
df.columns
df.info() 
# dtype object: 
# df['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 
#    'Var_1','Segmentation']
# dtype int,float:
# ['ID','Age','Work_Experience','Family_Size']
df.isnull().sum() 
# missing value for 
# Ever_Married(140),Graduated(78),Profession(124),Work_Experience(829),
# Family_Size(335),Var_1(76)
df.duplicated().sum()
# no duplicates
df.describe().T
# show only for numeric data, need to convert categorical data to numerical data


# 3) Data cleaning
# remove ID column
# ID is just the unique identification for customer, will not be needed for data training
df.drop(['ID'],axis = 1,inplace=True)

# need to change categorical data to numerical data
# convert string to number
le = LabelEncoder()
encoders = dict()
c_data= ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score',
         'Var_1','Segmentation'] #all categorical data
for c_data in df.columns:
    series = df[c_data] # a series of data with name of column
    label_encoder = LabelEncoder()
    df[c_data] = pd.Series(
        label_encoder.fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index)
    encoders[c_data] = label_encoder
print(encoders)
# save label encoding
lefile='le.pkl'
pickle.dump(label_encoder,open(lefile,'wb'))

# remain NaN in the dataset 
# references:https://localcoder.org/labelencoder-that-keeps-missing-values-as-nan
print(df)

# deal with missing data
# check the null value again
df.isnull().sum() 
# Ever_Married(140),Graduated(78),Profession(124),Work_Experience(829),
# Family_Size(335),Var_1(76)
# replace missing data using KNN imputer
imputer = KNNImputer()
# by default, n_neighbors=5, metric='nan_euclidean'
df['Ever_Married']= imputer.fit_transform(np.expand_dims(df['Ever_Married'],-1))
df['Graduated'] = imputer.fit_transform(np.expand_dims(df['Graduated'],-1))
df['Profession']= imputer.fit_transform(np.expand_dims(df['Profession'],-1))
df['Family_Size'] = imputer.fit_transform(np.expand_dims(df['Family_Size'],-1))
df['Work_Experience']= imputer.fit_transform(np.expand_dims(df['Work_Experience'],-1))
df['Var_1']= imputer.fit_transform(np.expand_dims(df['Var_1'],-1))


# recheck data
df.describe().T

# 4) Feature selection
None

# 5) Data preprocessing
X= df.loc[:, df.columns != 'Segmentation'] # features
y = df['Segmentation'] #target
print(X.shape) #(8068,9)
print(y.shape) #(8068,)


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=12)

# data scaling using MinMaxScaler
# fit and transfrom for testing dataset
# transform only for testing dataset
mms= MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)
print(x_train.shape) #(5647,9)
print(x_test.shape) #(2421,9)
# save scaler to pickle file
mms_scaler='mms.pkl'
pickle.dump(mms,open(mms_scaler,'wb'))

enc = OneHotEncoder(sparse=False)
y_train = enc.fit_transform(np.expand_dims(y_train,axis=-1)) # y train
y_test = enc.transform(np.expand_dims(y_test,axis=-1)) # y test
# save one hot encoder to pickle file
ohe='ohe.pkl'
pickle.dump(enc,open(ohe,'wb'))

#%% Train model
# sequential
model = Sequential(name=('customer_segmentation'))
model.add(Input(shape=(9))) #input layer 1
model.add(Dense(100,activation='relu')) #hidden layer 1
model.add(BatchNormalization()) 
model.add(Dropout(0.2))
model.add(Dense(82,activation= 'relu')) #hidden layer 2
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(62,activation= 'relu')) #hidden layer 3
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32,activation= 'relu')) # hidden layer 4
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(4,activation='softmax')) #output layer
model.summary()
plot_model(model)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='acc')


#%% Performance evaluation
# callbacks (tensorboard) for visualization
tensorboard_callback = TensorBoard(log_dir= log_customer_segmentation, histogram_freq=1)

# early stopping callback
# to deal with overfitting
# below here ask it to refer to val_loss, stop training after overfit by 10 times (refer to patience)
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=6)

# train the model
hist = model.fit(x_train,y_train, epochs=200, validation_data=(x_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])

# to show how many epochs it run
len(hist.history['val_loss'])
# it stopped at epoch 26

#generate classification report
report_generation(x_test, y_test)
# low accuracy obtained, 54%
# reasons not able to achieves high accuracy because
# lack of feature selection 
# functional api might perform better since there is multiple input and
# multiple output in this analysis


#%% Save model
# saving model
model.save(MODEL_SAVE_PATH)



