#!/usr/bin/env python
# coding: utf-8

# In[21]:



#IMPORT LIBRARIES
import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt 

from sklearn import model_selection 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import plotly.express as px
import numpy as np


# In[2]:


tips=pd.read_csv('tips datasets.csv')


# In[3]:


#SAMPLE OF THE DATA
tips.head()


# In[4]:


#DIMENSION OF THE DATA
tips.shape


# In[5]:


#STATISTICAL SUMMARY OF THE DATA 
tips.describe()


# In[6]:


# Checking missing values for the dataset
tips.isnull().any()


# In[7]:


# Check the datatype
tips.info()


# In[8]:


# Total amount of tips by day and time
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(35, 10))
sns.boxplot(x="day", y="tip", data=tips, ax=ax1, palette="Blues_d")
ax1.set_title("Boxplot for day variable",fontsize = 30)
ax1.set_xlabel("Day" , fontsize = 20)
ax1.set_ylabel("Tip" , fontsize = 20)
sns.boxplot(x="time", y="tip", data=tips, ax=ax2, palette="Blues_d")
ax2.set_title("Boxplot for time variable",fontsize = 30)
ax2.set_xlabel("Time" , fontsize = 20)
ax2.set_ylabel("Tip" , fontsize = 20)


# In[9]:


# Total amount of tips by total bill and by size of group
fig, (ax5, ax6) = plt.subplots(ncols=2, figsize=(35, 10))
sns.boxplot(x="total_bill", y="tip", data=tips, ax=ax5, palette="Blues_d")
ax5.set_title("Boxplot for total bill variable" , fontsize=30)
ax5.set_xlabel("total bill" , fontsize = 20)
ax5.set_ylabel("tip" , fontsize = 20)
sns.boxplot(x="size", y="tip", data=tips, ax=ax6, palette="Blues_d")
ax6.set_title("Boxplot for size variable",fontsize= 30)
ax6.set_xlabel("size" , fontsize = 20)
ax6.set_ylabel("tip" , fontsize = 20)


# In[10]:


figure = px.scatter(data_frame = tips, x="total_bill",
                    y="tip", size="size", color= "sex", trendline="ols")
figure.show()


# In[11]:


#TRAINING THE MODEL 
#Transforming the categorical values into numerical values:
tips["sex"] = tips["sex"].map({"Female": 0, "Male": 1})
tips["smoker"] = tips["smoker"].map({"No": 0, "Yes": 1})
tips["day"] = tips["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
tips["time"] = tips["time"].map({"Lunch": 0, "Dinner": 1})
tips.head()


# In[12]:


#SPLITTING DATA
x = np.array(tips[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
y = np.array(tips["tip"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[13]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)


# In[14]:


# features = [[total_bill, "sex", "smoker", "day", "time", "size"]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])
model.predict(features)


# In[15]:


#Class Distribution
print("\n\n\nSplit the Data by SIZE and Provide a Count:\n", tips.groupby('size').size())


# In[16]:


#Univariate Plots – Histogram
print('\n\nHistograms\n')
tips.hist()
plt.show()


# In[17]:


#SPLIT-OUT OR CREATE A VALIDATION DATASET 
print('\n\n\nTrain the Model: Split-out validation dataset ')
array= tips.values
X = array[:,0:4]
#print(X)
Y=array[:,4]
#print(Y)
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state=seed)


# In[18]:


#TEST OPTIONS AND EVALUATION MATRIX 
#CHECK THE ALGORITHMS 
seed=7
scoring='accuracy'


# In[22]:


print('\n\n\nSpot Check Algorithms\n')
models=[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append (('SVM', SVC(gamma='auto')))
models.append(('SVM', SVC()))


# In[23]:


print('\n\n\nEvaluate Each Model in Turn\n')
results = []
names=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results= model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    msg="%s:%f(%f)"%(name, cv_results.mean(),cv_results.std())
    print('Accuracy Score:', msg)


# In[ ]:


## ACCURACY REPORT SAYS US THAT THE BEST ALGORITHM FOR THE ACCURACY IS "NB" WITH 0.45.... 


# In[25]:


#MAKE PREDICTIONS ON VALIDATION DATASETS

#MAKE PREDICTIONS USING NB
print("\nMake Predictions Using NB"),
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print('\nNB Accuracy Score:', accuracy_score (Y_validation, predictions))
print('\nNB Confusion Matrix: \n', confusion_matrix(Y_validation, predictions))
print('\nNB Classification Report: \n', classification_report (Y_validation, predictions))


# In[ ]:


### Results Show the Accuracy, the Confusion Matrix, and the Classification Report 
### ACCURACY SCORE IS %38 


# In[26]:


#SPLIT-OUT %50 - CREATE A VALIDATION DATASET 
print('\n\n\nTrain the Model: Split-out validation dataset ')
array= tips.values
X = array[:,0:4]
#print(X)
Y=array[:,4]
#print(Y)
validation_size = 0.5
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state=seed)


# In[27]:


#TEST OPTIONS AND EVALUATION MATRIX 
#CHECK THE ALGORITHMS 
seed=7
scoring='accuracy'


# In[28]:


print('\n\n\nSpot Check Algorithms\n')
models=[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append (('SVM', SVC(gamma='auto')))
models.append(('SVM', SVC()))


# In[29]:


print('\n\n\nEvaluate Each Model in Turn\n')
results = []
names=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results= model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    msg="%s:%f(%f)"%(name, cv_results.mean(),cv_results.std())
    print('Accuracy Score:', msg)


# In[ ]:


## NOW THE ACCURACY REPORT SAYS US THAT THE BEST ALGORITHM FOR THE ACCURACY IS "LDA" WITH 0.40....  


# In[30]:


#MAKE PREDICTIONS USING KNN
print("\nMake Predictions Using LDA"),
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print('\nLDA Accuracy Score:', accuracy_score (Y_validation, predictions))
print('\nLDA Confusion Matrix: \n', confusion_matrix(Y_validation, predictions))
print('\nLDA Classification Report: \n', classification_report (Y_validation, predictions))


# In[ ]:


### THE ACCURACY SCORE IS %43 
### Classification Report Shows a Breakdown of Each Class by Precision, Recall, “f1-score” and Support...

