#!/usr/bin/env python
# coding: utf-8

# In[40]:


#now we will code our own linear regression , this will have some functions 
# predict function , score function ,cost function , fit function
import numpy as np


# In[41]:


data = np.loadtxt("C:\Abhinav\Course\CN\Downloaded NB\Linear Regression\data.csv", delimiter = ",") #by default delimiter is space

#data.shape() gives (100,2)

x = data[:,0].reshape(-1,1) #this is done to make it a 2d array as sklearn doesnt accept 1d array for features
y = data[:,1]


# In[42]:


def fitfunc(x,y): #this is xtrain and ytrain np arrays
    #print((x*y).mean())
    print(x.mean(), y.mean(), (x*x).mean(), (x*y).mean())
    m = ((x*y).mean() - x.mean()*y.mean())/((x*x).mean()-x.mean()*x.mean())
    #print(m)
    c = y.mean() - m*(x.mean())
    return m , c


# In[43]:


def pred (x,m,c): # we have calculated the values of m,c from fit function , then we use for predicting new array for test data
    y = m*x+c
    return y


# In[44]:


def cost (x,y,m,c): #this is used to calculate sum of squared error
    ypred = pred(x,m,c)
    ypred = (ypred - y)**2
    costsum = ypred.sum()
    return costsum


# In[45]:


def scorefunc(ypred,ytrue):
    u = ((ypred - ytrue)**2).sum()
    v = ((ytrue - ytrue.mean())**2).sum()
    return 1 - (u/v)


# In[57]:


from sklearn import model_selection
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y) #75% split by default


# In[58]:


slope,intercept = fitfunc(x_train,y_train)
print(slope,intercept)


# In[48]:


y_pred = pred(x_test,slope,intercept)
#print(y_pred)


# In[49]:


costval =  cost(x_test,y_test,slope,intercept)
print(costval)


# In[50]:


ypred = pred(x_test,slope,intercept)
scoreval =  scorefunc(ypred,y_test)
print(scoreval)


# In[52]:


import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
x_line = np.arange(30,70,0.1)
print(x_line)
print(slope,intercept)
y_line = (slope*x_line) + intercept
print(y_line)
plt.plot(x_line,y_line)
plt.show()

