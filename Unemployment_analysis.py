#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


unemployment = pd.read_csv('C:/Users/THYAGARAJU/Downloads/Up.csv')


# In[4]:


unemployment.head()


# In[5]:


unemployment.info()


# In[6]:


unemployment.shape


# In[7]:


unemployment.describe()


# In[8]:


unemployment.tail()


# In[9]:


unemployment.columns =['States','Date','Frequency','Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate','Region']
     


# In[11]:


unemployment['Frequency']= unemployment['Frequency'].astype('category')


# In[12]:


unemployment['Date'] = pd.to_datetime(unemployment['Date'],dayfirst=True)


# In[13]:


unemployment.head()


# In[15]:


unemployment["States"].unique()


# In[16]:


undf_stats1 =unemployment[['Estimated Unemployment Rate','Estimated Employed', 'Estimated Labour Participation Rate']]


# In[17]:


undf_stats1.head()


# In[18]:


undf_stats1.describe()


# In[19]:


round(undf_stats1.describe().T,2)


# In[24]:


region_stats1 = unemployment.groupby(['Region'])[['Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate']]


# In[28]:


plt.figure(figsize=(25, 8))

plt.subplot(1,2,1)
sns.barplot(x='Region', y='Estimated Unemployment Rate', data=unemployment)
plt.xlabel("Region",fontsize=16)
plt.ylabel('Estimated Unemployment Rate',fontsize=16)
plt.title('Estimated Unemployment Rate Before Lockdown',fontsize=16)
plt.show()


# In[30]:


States_stats1 = unemployment.groupby(['States'])[['Estimated Unemployment Rate','Estimated Employed','Estimated Labour Participation Rate']]


# In[31]:


States_stats1.head()


# In[34]:


color = sns.color_palette()
cnt_srs = unemployment.States.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('States', fontsize=12)
plt.title('Count the states', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:




