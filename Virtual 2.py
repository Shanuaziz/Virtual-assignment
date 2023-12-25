#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


test=pd.read_csv('test.csv')


# In[3]:


test.head()


# In[4]:


train=pd.read_csv('train.csv')


# In[5]:


train.head()


# In[6]:


train.drop(columns=['Loan_ID'], inplace=True)


# In[7]:


train


# In[8]:


train.nunique()


# In[9]:


test.nunique()


# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# In[234]:


train.dropna()


# In[235]:


test.dropna()


# In[236]:


sns.countplot(data=train,x='Loan_Status',hue='Property_Area')


# In[237]:


sns.countplot(data=train,x='Loan_Status',hue='Credit_History')


# In[238]:


sns.countplot(data=train,x='Loan_Status',hue='Loan_Amount_Term')


# In[239]:


sns.barplot(data=train, x='Loan_Status', y='LoanAmount')


# In[240]:


sns.barplot(data=train, x='Loan_Status', y='CoapplicantIncome')


# In[241]:


sns.barplot(data=train, x='Loan_Status', y='ApplicantIncome')


# In[242]:


sns.barplot(data=train, x='Loan_Status', y='Credit_History')


# In[243]:


train


# In[244]:


check_missing=train.isnull().sum()*100/train.shape[0]
check_missing[check_missing>0].sort_values(ascending=False)


# In[245]:


train['Credit_History'].fillna(0, inplace=True)
train['Self_Employed'].fillna('No', inplace=True)
train['LoanAmount'].fillna(0, inplace=True)
train['Dependents'].fillna(0, inplace=True)
train['Loan_Amount_Term'].fillna(0, inplace=True)
train['Gender'].fillna('Other', inplace=True)
train['Married'].fillna('Other', inplace=True)


# In[246]:


check_missing=train.isnull().sum()*100/train.shape[0]
check_missing[check_missing>0].sort_values(ascending=False)


# In[247]:


train.dropna()
train.shape


# In[248]:


train


# In[249]:


train['Gender'].unique()


# In[250]:


train['Married'].unique()


# In[251]:


train['Dependents'].unique()


# In[252]:


train['Education'].unique()


# In[253]:


train['Self_Employed'].unique()


# In[263]:


train['Loan_Status'].unique()


# In[264]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[265]:


train['Gender']=label_encoder.fit_transform(train['Gender'])
train['Gender'].unique()


# In[266]:


train['Married']=label_encoder.fit_transform(train['Married'])
train['Married'].unique()


# In[267]:


train['Dependents']=label_encoder.fit_transform(train['Dependents'])
train['Dependents'].unique()


# In[268]:


train['Education']=label_encoder.fit_transform(train['Education'])
train['Education'].unique()


# In[269]:


train['Self_Employed']=label_encoder.fit_transform(train['Self_Employed'])
train['Self_Employed'].unique()


# In[270]:


train['Loan_Status']=label_encoder.fit_transform(train['Loan_Status'])
train['Loan_Status'].unique()


# In[289]:


train.drop('Property_Area',axis=1)


# In[290]:


sns.boxplot(x=train['ApplicantIncome'])


# In[291]:


sns.boxplot(x=train['CoapplicantIncome'])


# In[292]:


sns.boxplot(x=train['LoanAmount'])


# In[293]:


sns.heatmap(train.corr())


# In[302]:


train


# In[303]:


train.dtypes


# In[311]:


X=train.drop('Property_Area',axis=1)
y=train['Property_Area']


# In[312]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=44, shuffle=True)


# In[313]:


from sklearn import metrics


# In[314]:


from sklearn.metrics import accuracy_score


# In[315]:


from sklearn.ensemble import RandomForestClassifier


# In[316]:


rfc = RandomForestClassifier(random_state=0)


# In[317]:


rfc.fit(X_train, y_train)


# In[318]:


y_pred = rfc.predict(X_test)


# In[319]:


from sklearn.metrics import accuracy_score


# In[320]:


print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[325]:


#Decision Tree


# In[326]:


train


# In[327]:


from sklearn.tree import DecisionTreeClassifier


# In[328]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[345]:


from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[346]:


result=confusion_matrix(y_test,y_pred)
result


# In[329]:


from sklearn.naive_bayes import GaussianNB


# In[330]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


# In[331]:


from sklearn import metrics


# In[332]:


print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred))


# In[335]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[336]:


svc_classifier=SVC()
svc_classifier.fit(X_train,y_train)


# In[337]:


knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)


# In[338]:


svc_predictions= svc_classifier.predict(X_test)
knn_predictions=knn_classifier.predict(X_test)


# In[339]:


from sklearn.metrics import accuracy_score


# In[340]:


svc_accuracy=accuracy_score(y_test, svc_predictions)
knn_accuracy=accuracy_score(y_test, knn_predictions)


# In[341]:


print("Support Vector Classifier Accuracy:", svc_accuracy)
print("K Nearest Neighbors Classifier Accuracy:", knn_accuracy)


# In[342]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[343]:


print(confusion_matrix(svc_predictions,y_test))
print(classification_report(knn_predictions,y_test))


# In[ ]:




