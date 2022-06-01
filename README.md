![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)

# customer-segmentation-using-deep-learning
Approach of doing customer segmentation using deep learning.

This python scripts have been tested and run using Spyder(Python 3.8).
<br>The raw data can be found in the link below:
<br>[Customer segmentation](https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation)
<br>Thank you to the data contrbutor: Mr.Abishek Sudarshan

### FOLDERS AND FILES
**data** folder: data for training, new input dataset for deployment, new csv file saved after prediction.
<br>**figures** folder: pairplots, elbow graph, model, classification report and tensorboard interface for performance evaluation.
<br>**log_customer_segmentation** folder: stored information for tensorboard.
<br>**customer_segmentation_deploy.py**: deployment of deep learning model built.
<br>**customer_segmentation_train.py**: deep learning model training.
<br>**model_customer_segmentation.h5**: deep learning model saved.
<br>**variablename.pkl**: Numbers of label encoder fitted with train data. Variable name here refer to categorical data in dataset such as emarried, gender, graduated, profession, segmentation, sscore and var1.
<br>**knn_imputer.pkl**: Imputer for missing data.Trained with train data.
<br>**mms.pkl**: Min max scaler fitted with train data.
<br>**ohe.pkl**: One hot encoder fitted with train data.

### MODEL
Data undergo exploratory data analysia and data cleaning.
<br>Column ID is not needed for data training, hence it can be removed.
<br>6 features out of 9 features are detected to have missing value.
<br>KNN imputer is chosen to replace the NaN value.
<br>Min max scaling had been done to keep the range of features within -1 and 1. 
<br>One hot encoder enables the target variable to be used in the deep learning model.
<br>A sequential deep learning model of 6 hidden layers is built to train the data.

![Image](https://github.com/innju/customer-segmentation-using-deep-learning/blob/main/figures/model.png)

Relu activation function and batch normalization could help in avoid exploding and vanishing gradients while dropout layer can prevent overfitting.The model is evaluated using the categorical_crossentropy as loss and accuracy as the metrics.

### RESULTS
The classification report shown model accuracy of 0.54, which is equilvalent to 54.0%.

![Image](https://github.com/innju/customer-segmentation-using-deep-learning/blob/main/figures/classification_report_cs.png)

More layers added and increased number of neurons for layers could help to prevent underfitting problem.

![Image](https://github.com/innju/customer-segmentation-using-deep-learning/blob/main/figures/tensorboard_cs.png)

I used tensorboard in this analysis to view the performance and gained the graph as above.You could follow the step below to view the tensorboard from your own device as well.

Open anaconda prompt> activate your environment> type tensorboard --logdir (include the path to my logfile here)
<br>You may view the tensorboard once you click enter after the steps above.

Training metrics for each epoch is recorded in the history callback, including the loss and accuracy. The figure shows the plot of accuracy and loss on training and validation datasets over training epoch. Both line shown similar patterns. However, the epoch accuracy of the train (pink line) is slightly lower and the epoch loss of the train is slightly higher compared to validate (green line). Possible reason is because of the dropout layer that only applicable during the training process. Therefore, it is reasonable to obtain smaller validation error. 

### DISCUSSIONS
1. The accuracy of the deep learning model is not high. It is because of the fact that no obvious patterns observed for this dataset. The pairplots below shown no distinct groups observed for the four categories of segmentation.

![Image](https://github.com/innju/customer-segmentation-using-deep-learning/blob/main/figures/snspairplot.png)

The elbow graph obtained through k means clustering algorithm shows that the optimal k for this dataset is 2. It's the point where the inertia begins to decrease and appeared as the "elbow" of the graph. Based on the graph, k=4 could contribute to lower value of inertia. However, the inertia calculated is still high and not ideal, with value of 4869.43. High k means inertia indicates the model is not well clustered.  

![Image](https://github.com/innju/customer-segmentation-using-deep-learning/blob/main/figures/elbow_cs.png)

Thus, it can be concluded that the nature of the data that do not consist of obvious patterns caused the deep learning model to perform poorly.


2. Sometimes, it is fine to drop the columns with too many missing values as the missing data replaced through imputation method could be varied with the real world data. Imputer used can only estimates the missing data for us but not necessary to be exactly the same as in real world.


3. Dependent variable should be included during the KNN imputation. Excluding the dependent variable could caused biased estimates because the imputation model will assume there is no correlation between the dependent variable and independent variables.
[References: Graham,2009](https://www.personal.psu.edu/jxb14/M554/articles/Graham2009.pdf)



<br>Thanks for reading.
