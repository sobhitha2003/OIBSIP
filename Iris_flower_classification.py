#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[26]:


iris=pd.read_csv('C:/Users/THYAGARAJU/Downloads/Iris.csv')
print(iris)


# In[27]:


print(iris.shape)


# In[28]:


print(iris.describe())


# In[29]:


#Checking for null values
print(iris.isna().sum())
print(iris.describe())


# In[30]:


iris.head()


# In[31]:


iris.tail(100)


# In[32]:


n = len(iris[iris['Species'] == 'versicolor'])
print("No of Versicolor in Dataset:",n)


# In[33]:


n1 = len(iris[iris['Species'] == 'virginica'])
print("No of Virginica in Dataset:",n1)


# In[34]:


n2 = len(iris[iris['Species'] == 'setosa'])
print("No of Setosa in Dataset:",n2)


# In[35]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[36]:


iris.hist()
plt.show()


# In[37]:


sns.pairplot(iris)


# In[38]:


X = iris['SepalLengthCm'].values.reshape(-1,1)
print(X)


# In[39]:


Y = iris['SepalWidthCm'].values.reshape(-1,1)
print(Y)


# In[40]:


sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris)


# In[42]:


corr_mat = iris.corr()
print(corr_mat)


# In[43]:


train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[46]:


train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y = train['Species']

test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test['Species']


# In[47]:


train_X.head()


# In[48]:


test_y.head()


# In[50]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))


# In[51]:


confusion_mat = confusion_matrix(test_y,pred_y)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,pred_y))


# In[52]:


#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# In[53]:


confusion_mat = confusion_matrix(test_y,y_pred2)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,y_pred2))


# In[54]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))


# In[55]:


confusion_mat = confusion_matrix(test_y,y_pred3)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,y_pred3))


# In[61]:


#Using MultinomialNB
from sklearn.naive_bayes import MultinomialNB
model5 = MultinomialNB()
model5.fit(train_X,train_y)
y_pred5 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred5))


# In[62]:


confusion_mat = confusion_matrix(test_y,y_pred3)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,y_pred5))


# In[65]:


#Using BernoulliNB
from sklearn.naive_bayes import BernoulliNB
model6 = BernoulliNB()
model6.fit(train_X,train_y)
y_pred6 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred6))


# In[66]:


confusion_mat = confusion_matrix(test_y,y_pred6)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,y_pred6))


# In[56]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))


# In[57]:


confusion_mat = confusion_matrix(test_y,y_pred4)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,y_pred4))


# In[ ]:




