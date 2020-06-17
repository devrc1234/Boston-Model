#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
dataset=load_boston()
print(dataset.data)


# In[4]:


print(dataset.feature_names)


# In[5]:


print(dataset.DESCR)


# In[6]:


print(dataset.target)


# In[9]:


df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df.head()


# In[10]:


df['MEDV']=dataset.target
df.head()


# In[11]:


df.info()


# In[12]:


print(df.isnull().sum())


# In[13]:


corr=df.corr()
print(corr)


# In[14]:


print(df.corr().abs().nlargest(3, 'MEDV').index)
print(df.corr().abs().nlargest(3, 'MEDV').values[:,13])


# In[16]:


plt.scatter(df['LSTAT'],df['MEDV'],marker='^')
plt.xlavel('LSTAT')
plt.ylavel('MEDV')


# In[ ]:




