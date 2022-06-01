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
import seaborn as sns
import matplotlib.pyplot as plt

#%% Static code
PATH = os.path.join(os.getcwd(),'data','train.csv') #load data
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model_customer_segmentation.h5') #save model
LOG_SAVE_PATH = os.path.join(os.getcwd(),'log_customer_segmentation') 
log_customer_segmentation = os.path.join(
    LOG_SAVE_PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')# save scaler path
OHE_SAVE_PATH = os.path.join(os.getcwd(),'ohe.pkl')# save encoder path
# save label encoder path:
SAVE_PATH_GENDER = os.path.join(os.getcwd(),'gender.pkl')
SAVE_PATH_EMARRIED = os.path.join(os.getcwd(),'emarried.pkl')
SAVE_PATH_GRADUATED = os.path.join(os.getcwd(),'graduated.pkl')
SAVE_PATH_PROFESSION = os.path.join(os.getcwd(),'profession.pkl')
SAVE_PATH_SSCORE = os.path.join(os.getcwd(),'sscore.pkl')
SAVE_PATH_VAR1 = os.path.join(os.getcwd(),'var1.pkl')
SAVE_PATH_SEGMENTATION = os.path.join(os.getcwd(),'segmentation.pkl')
SAVE_PATH_KNN_IMPUTER = os.path.join(os.getcwd(),'knn_imputer.pkl')

#%% Function defined
# save label encoder
def categorical_label(col_name,SAVE_PATH):
    le= LabelEncoder()
    temp = df[col_name]
    temp[temp.notnull()]=le.fit_transform(temp[temp.notnull()])
    # choose to fit and transform for data that is not null
    pickle.dump(le,open(SAVE_PATH,'wb'))

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
# convert string to number using label encoder

#%% Label encoder updated
# be careful, if there is NaN in the data
# it will treat the NaN as certain label
# because NaN is recognized as float in label encoder
# hence, have to tell label encoder to remain the NaN in dataset
# save the label encoder for each categorical column one by one
# so that it can be used in deployment file later (for preprocessing new input data)
# since the column name is set in the function, no need to write df['Gender']= ...anymore
categorical_label(col_name='Gender',SAVE_PATH=SAVE_PATH_GENDER)
categorical_label(col_name='Ever_Married',SAVE_PATH=SAVE_PATH_EMARRIED)
categorical_label(col_name='Graduated',SAVE_PATH=SAVE_PATH_GRADUATED)
categorical_label(col_name='Profession',SAVE_PATH=SAVE_PATH_PROFESSION)
categorical_label(col_name='Spending_Score',SAVE_PATH=SAVE_PATH_SSCORE)
categorical_label(col_name='Var_1',SAVE_PATH=SAVE_PATH_VAR1)
categorical_label(col_name='Segmentation',SAVE_PATH=SAVE_PATH_SEGMENTATION)

# convert categorical columns above to numeric data type
c_data= ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score',
          'Var_1','Segmentation'] #all categorical data
df[c_data] = df[c_data].apply(pd.to_numeric, errors='coerce')

# recheck data type to ensure successful data conversion
df.info()

# check the data distribution 
sns.pairplot(df,hue='Segmentation',palette='colorblind')
plt.show()
# no obvious patterns observed for 4 categories of segmentation
# could be a hint to poor performance of deep learning model
#%%

# deal with missing data
# check the null value again
df.isnull().sum() 
# Ever_Married(140),Graduated(78),Profession(124),Work_Experience(829),
# Family_Size(335),Var_1(76)
# replace missing data using KNN imputer
imputer = KNNImputer()
# by default, n_neighbors=5, metric='nan_euclidean'
# do not fit and transform certain column only
# because KNN will takes the properties of the entire dataset to impute
df2 = imputer.fit_transform(df)
# save knnn imputer to pickle file
knn='knn_imputer.pkl'
pickle.dump(imputer,open(knn,'wb'))
# rename columns name of df2
df2 = pd.DataFrame(df2,columns=[df.columns])

# recheck data
df2.isnull().sum() 
# no more missing data


# 4) Feature selection
None

# 5) Data preprocessing
X= df2.loc[:, df.columns != 'Segmentation'] # features
y = df2['Segmentation'] #target
print(X.shape) #(8068,9)
print(y.shape) #(8068,1)

# data scaling using MinMaxScaler
mms= MinMaxScaler()
X = mms.fit_transform(X)
# save scaler to pickle file
mms_scaler='mms.pkl'
pickle.dump(mms,open(mms_scaler,'wb'))


enc = OneHotEncoder(sparse=False)
y = enc.fit_transform(y)
# save one hot encoder to pickle file
ohe='ohe.pkl'
pickle.dump(enc,open(ohe,'wb'))


# train_test_split have to be the very last step after any data preprocessing
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


#%% Train model
# sequential
model = Sequential(name=('customer_segmentation'))
model.add(Input(shape=(9))) #input layer 1
model.add(Dense(64,activation='relu')) #hidden layer 1
model.add(BatchNormalization()) 
model.add(Dropout(0.2))
model.add(Dense(64,activation= 'relu')) #hidden layer 2
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64,activation= 'relu')) #hidden layer 3
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64,activation= 'relu')) # hidden layer 4
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32,activation= 'relu')) # hidden layer 5
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32,activation= 'relu')) # hidden layer 6
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
# below here ask it to refer to val_loss, stop training after overfit by 4 times (refer to patience)
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=4)

# train the model
hist = model.fit(x_train,y_train, epochs=200, validation_data=(x_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])

# to show how many epochs it run
len(hist.history['val_loss'])
# it stopped at epoch 36

#generate classification report
report_generation(x_test, y_test)
# low accuracy obtained, 54%

#%% reasons not able to achieves high accuracy:
# there is no pattern in the dataset
# can try to use unsupervised learning model to see if it can show any clusters or not
# if there is no pattern ==> deep learning fails to work
from sklearn.cluster import KMeans
# use elbow method to find the optimal number of cluster to cluster this data
cs = []
for i in range(1,6):
    kmeans = KMeans(n_clusters = i,random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1,6), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()
# based on the graph, 2 is the ideal number of cluster to cluster this dataset
# however, high kmeans inertia obtained: 6929.94
# if k=4, high kmeans inertia obtained: 4869.43
# as a conclusion, this dataset is not able to contribute obvious patterns in
# categorize 4 categories of segmentation
# thus, it resulted poor performance in the deep learning model


#%% Save model
# saving model
model.save(MODEL_SAVE_PATH)


