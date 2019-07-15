#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#import data to pandas dataframe
customers = pd.read_csv('Ecommerce Customers.csv')


# In[11]:


customers.head()


# In[12]:


customers.info()


# In[13]:


customers.describe()


# In[14]:


sns.distplot(customers['Yearly Amount Spent'])


# In[15]:


sns.distplot(customers['Length of Membership'])


# In[109]:


sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent' )


# In[110]:


sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent' )


# In[ ]:


sns.jointplot()


# In[16]:


sns.pairplot(customers)


# In[30]:


customers.isna().sum()


# In[31]:


customers.corr()


# In[35]:


sns.heatmap(customers.corr(), annot = True)


# In[37]:


customers.columns


# In[69]:


customers['Avatar'].unique()


# In[94]:


customers_i1 = customers.copy(deep = True)


# In[95]:


customers_i1.drop({'Email','Address'},axis = 1, inplace=True)


# In[96]:


customers_i1.drop({'Avatar'},axis = 1, inplace=True)


# In[72]:


#customers_i1 = pd.get_dummies(customers_i1)


# In[97]:


X = customers_i1.copy(deep = True)
X.drop('Yearly Amount Spent', axis = 1 , inplace=True)


# In[98]:


y = customers_i1['Yearly Amount Spent']


# In[45]:


from sklearn.model_selection import train_test_split


# In[99]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4 , random_state = 101)


# In[47]:


from sklearn.linear_model import LinearRegression


# In[51]:


lm = LinearRegression()


# In[100]:


lm.fit(X_train, y_train)


# In[101]:


print(lm.intercept_)


# In[102]:


print(lm.coef_)


# In[103]:


coeff = pd.DataFrame(lm.coef_ , X_train.columns, columns = ['Coeff'])
coeff


# In[104]:


predictions = lm.predict(X_test)


# In[105]:


sns.distplot(y_test-predictions)


# In[87]:


from sklearn import metrics


# In[106]:


print(metrics.mean_absolute_error(y_test, predictions))


# In[107]:


print(metrics.mean_squared_error(y_test, predictions))


# In[108]:


print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

