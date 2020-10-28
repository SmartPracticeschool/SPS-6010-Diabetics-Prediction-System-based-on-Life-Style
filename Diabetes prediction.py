#!/usr/bin/env python
# coding: utf-8

# # Diabetics Prediction System Based On Life Style
# 
# In this, we need to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# The datasets consist of several medical predictor variables and one target variable, Diabetes. Predictor variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

# ## Step 1 : Import Libraries and Dataset

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('diabetes.csv') #import dataset

# 
# 1. Dataset consists of 8 predictor variables and one target variable.
# 2. Target variable is the Outcome which is 1 when diabetes is positive and 0 when negative.
# 3. There are no categorical data, feature values are of int or float type.
# 4. There are no missing values. However,it is observed that some of the entries are 0 which is invalid and has to be taken care of.
# 5. Dataset consists of 768 rows and 9 columns.

# ## Step 3: Data Visualization

# In[10]:


# https://matplotlib.org/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py


# In[13]:


# In[18]:


new_data = df # Replace 0s in Glucose,BP,Skinthickness,insulin and BMI with nan's 
new_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = new_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)


# In[19]:


new_data.isnull().sum()


# In[20]:


# Replacing NaN with mean values
new_data["Glucose"].fillna(new_data["Glucose"].mean(), inplace = True)
new_data["BloodPressure"].fillna(new_data["BloodPressure"].mean(), inplace = True)
new_data["SkinThickness"].fillna(new_data["SkinThickness"].mean(), inplace = True)
new_data["Insulin"].fillna(new_data["Insulin"].mean(), inplace = True)
new_data["BMI"].fillna(new_data["BMI"].mean(), inplace = True)


# In[21]:


new_data.isnull().sum()


# In[22]:


new_data.describe()


# In[23]:


# Feature scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
data_scaled = sc.fit_transform(new_data)


# In[24]:


data_scaled = pd.DataFrame(data_scaled)
data_scaled.head()


# In[31]:


X = data_scaled.drop(8,axis=1)
y = data_scaled[8]


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = new_data['Outcome'])


# In[33]:


# Checking dimensions
# ## Step 5: Data Modelling
# 
# Try for models : Logistic Regression
#                  K Nearest Neighbors
#                  Naive Bayes Algorithm
#                  Decision tree Algorithm
#                  Random Forest Classifier
#                  Support Vector Machine

# In[36]:



# In[40]:


# Plotting a graph for n_neighbors 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier



# In[68]:


# K nearest neighbors Algorithm

knn = KNeighborsClassifier(n_neighbors = 16, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)


# In[69]:


# Support Vector Classifier Algorithm


# In[70]:


# Naive Bayes Algorithm



import pickle
pickle.dump(knn,open("model.pkl",'wb'))
