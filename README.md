![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)

# customer-segmentation-using-deep-learning
Approach of doing customer segmentation using deep learning

<br>This python scripts have been tested and run using Spyder(Python 3.8).
<br>The raw data can be found in the link below:
<br>[Customer segmentation](https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation)
<br>Thank you to the data contrbutor: Mr.Abishek Sudarshan

### FOLDERS AND FILES
**data** folder: raw data, data used for training and data predicted.
<br>**figures** classification report, model and tensorboard performance evaluation <br>interface.
<br>**log_customer_segmentation** folder: to view open and view performance of model using tensorboard.
<br>**customer_segmentation_deploy.py**: deployment of deep learning model built.
<br>**customer_segmentation_train.py**: deep learning model training.
<br>**model_customer_segmentation.h5**: deep learning model saved.
<br>**le.pkl**: Label encoder fitted with train data.
<br>**mms.okl**: Min max scaler fitted with train data.
<br>**ohe.pkl**: One hot encoder fitted with train data.

### DISCUSSION
Data undergo exploratory data analysia and data cleaning.
<br>Column ID is not needed for data training, hence it can be removed.
<br>6 features out of 9 features are detected to have missing value.
<br>KNN imputer is chosen to replace the NaN value.
<br>The use of label encoder have to be careful so that the NaN value remained in the data.
<br>In this analysis, I referred to the [LabelEncoder that keeps missing values as 'NaN'](https://localcoder.org/labelencoder-that-keeps-missing-values-as-nan) to do label encoding with the NaN value remained. 
<br>After that, min max scaling had been done to keep the range of features within -1 and 1. 
<br>One hot encoder enables the target variable to be used in the deep learning model.
<br>A sequential deep learning model of 4 hidden layers is built to train the data.
<br>Relu activation function and batch normalization could help in avoid exploding and vanishing gradients.
<br>Dropout layer can prevent overfitting.
<br>Classification problem is evaluated using the categorical_crossentropy as loss and accuracy as the metrics.


### RESULT




### IMPROVEMENT
1. I do found difficulties in apply to deployment later as the label encoder fitted with training data is unable to be applied to the deployment file, due to the way of coding. One of the improvement can be done here is use dictionary to manually map the categories to number instead of apply label encoder and ask it to remain the NaN value. This is one of the difficulties I faced when dealing with this dataset. 

2. The accuracy of the deep learning model is not high. One of the reason is the lack of use of feature selection. Another reason could be because I included outliers in the dataset without further data preprocessing for the outliers. Thirdly is the properties of deep learning model that tends to perform well with huge amount of dataset.

3. A modules consists of classes and functions should be separated from train file. I do need more times to practice on this.



Thanks for reading.
