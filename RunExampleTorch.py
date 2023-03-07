#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF


# In[2]:


import torch, tqdm
from torch.utils  import data
from torch import nn


# In[3]:

# Large dataset
ratings = pd.read_csv("data/ml-latest/ratings.csv")
ratings = ratings[['userId', 'movieId', 'rating']]


# In[4]:

# small dataset
# ratings = pd.read_csv("data/ml-100k/u.data", sep='\t', header=None)
# ratings.columns = ['userId', 'movieId', 'rating', 'ts']
# ratings = ratings[['userId', 'movieId', 'rating']]


# In[5]:


uid2user = dict(zip( ratings.userId.drop_duplicates(), range(ratings.userId.nunique())))
iid2item = dict(zip( ratings.movieId.drop_duplicates(), range(ratings.movieId.nunique())))


# In[6]:


ratings.userId = ratings.userId.map(uid2user)
ratings.movieId = ratings.movieId.map(iid2item)


# In[7]:


train_ratings, test_ratings = train_test_split(ratings, test_size=0.2)


# In[8]:


train_X = train_ratings[['userId', 'movieId']].values
train_y = train_ratings['rating'].values.astype('float64')
test_X = test_ratings[['userId', 'movieId']].values
test_y = test_ratings['rating'].values.astype('float64')


# In[9]:


train_X = torch.from_numpy(train_X).to(device='cuda')
train_y = torch.from_numpy(train_y).to(device='cuda')
test_X = torch.from_numpy(test_X).to(device='cuda')
test_y = torch.from_numpy(test_y).to(device='cuda')
train_y_mean = train_y.mean()


# In[10]:


# data iterater
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 1000
data_iter = load_array((train_X, train_y), batch_size)


# # ReRun from here

# In[40]:


# model
num_user = len(uid2user)
num_item = len(iid2item)
num_feat = 10

w_Item = torch.normal(0, 0.1, (num_item, num_feat), device='cuda', requires_grad=True)
w_User = torch.normal(0, 0.1, (num_user, num_feat), device='cuda', requires_grad=True)


# In[41]:


# loss
def squared_loss(y_hat, y):
    return (y_hat - y) ** 2


# In[42]:


# optimizor
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# In[44]:


lr = 0.8
stats = []
for epoch in range(100):
    for X, y in data_iter:
        yhat = (w_User[X[:, 0]] * w_Item[X[:, 1]]).sum(axis=1)
        l = squared_loss(yhat, y - train_y_mean)
        l.sum().backward()
        with torch.no_grad():
            for mat in [w_Item, w_User]:
                mat -= lr * mat.grad / batch_size
                mat.grad.zero_()
    with torch.no_grad():
        test_yhat = (w_User[test_X[:, 0]] * w_Item[test_X[:, 1]]).sum(axis=1)
        test_l = torch.sqrt(squared_loss(test_yhat + train_y_mean, test_y).mean())
        
        train_yhat = (w_User[train_X[:, 0]] * w_Item[train_X[:, 1]]).sum(axis=1)
        train_l = torch.sqrt(squared_loss(train_yhat + train_y_mean, train_y).mean())
        
        print(f"epoch: {epoch}, train loss: {train_l}, test loss: {test_l}")
        stats.append([epoch, train_l, test_l])


# In[ ]:

statsdf = pd.DataFrame(stats, columns = ['epoch', 'trainloss', 'testloss'])
statsdf = statsdf.set_index('epoch')
plt.plot(statsdf)
plt.savefig("torch_result.png")



