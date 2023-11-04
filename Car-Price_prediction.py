#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[2]:


data=pd.read_csv('C:/Users/THYAGARAJU/Downloads/cardata.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.info()


# In[9]:


data.CarName.unique()


# In[10]:


sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()


# In[11]:


data.corr()


# In[12]:


plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# In[16]:


predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

from sklearn.metrics import mean_absolute_error
model.score(x_test, predictions)


# In[ ]:




