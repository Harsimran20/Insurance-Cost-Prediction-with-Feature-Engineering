#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns


# In[4]:


insurance_data = pd.read_csv(r"C:\Users\ACER\Downloads\archive (9)\insurance.csv")


# In[5]:


insurance_data


# In[6]:


sns.scatterplot(x=insurance_data["bmi"],y=insurance_data["charges"],hue = insurance_data["smoker"])


# In[7]:


X = insurance_data.drop(columns = ["charges","region"])
y = insurance_data["charges"]
X["sex"] = X["sex"].map({"female":1, "male":0})
X["smoker"] = X["smoker"].map({"yes":1, "no":0})


# In[8]:


X.head()


# In[9]:


y.head()


# In[10]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)


# In[18]:


X_test.head()


# In[21]:


# Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[23]:


# Predict Values
y_pred = model.predict(X_test)


# In[25]:


y_pred


# In[27]:


y_test


# In[29]:


# Evaluate 
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print("r-squared:", r2)
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
print("Adjusted r^2:", adjusted_r2)


# In[31]:


X_test.shape


# In[33]:


insurance_data.head()


# In[39]:


# One Hot Encoding
X = insurance_data.drop(columns=["charges"])
y = insurance_data["charges"]
X = pd.get_dummies(X, columns = ["region"],drop_first = False)
X["sex"] = X["sex"].map({"female":1,"male":0})
X["smoker"] = X["smoker"].map({"yes":1,"no":0})


# In[41]:


X.head()


# In[45]:


X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)


# In[47]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)
print("r-squared:", r2)


# In[ ]:




