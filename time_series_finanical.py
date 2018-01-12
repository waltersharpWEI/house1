
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import pyflux as pf

matplotlib.rcParams['figure.figsize'] = (16.0, 8.0)


# In[2]:


total =  pd.read_csv('avg_price.txt',encoding='gbk')
total.head()


# In[3]:


samp = total[total['AREA'] == u'朝阳'].groupby(['DATA_DATE_RM']).mean().reset_index()
samp.head()


# In[4]:


# split into trainset and testset
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


length = samp.shape[0]
index = samp.DATA_DATE_RM.values
samples = samp['AVG_PRICE'].astype(np.float32).values
plt.plot(index,samples,'-o')
plt.show()
min_max_scaler = preprocessing.MinMaxScaler()
samples = min_max_scaler.fit_transform(samples)
plt.plot(index,samples.reshape(length,1),'-o')
plt.show()


# In[5]:


# ARIMA algorithms
model = pf.ARIMA(ar=1,ma=0,integ=1,data=samp,target='AVG_PRICE')
model.adjust_prior([0] , pf.Normal(0,30000))
print(model.latent_variables)


# In[6]:


x = model.fit('MLE')
x.summary()


# In[7]:


model.plot_fit(figsize=(15,5))


# In[9]:


model.plot_predict(h=1,past_values=100,figsize=(15,5))


# In[7]:


returns = pd.DataFrame(samp.AVG_PRICE.values)
# returns.to_csv('avg_chaoyang')
returns.index = returns.index.values + 1
model = pf.LMEGARCH(returns,p=1,q=1)
x = model.fit()
x.summary()


# In[67]:


returns.head()


# In[8]:


model.plot_fit(figsize=(15,5))

