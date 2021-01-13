# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../../src')

import math

import torch
import matplotlib.pyplot as plt

from ssgp import SSGP
from ssbq import SSBQ
from base_samplers import RBFSampler,MaternSampler

dim = 3
X = torch.randn(10000,dim)
Y = torch.sin(X.sum(dim=-1,keepdim=True)) + 0.01*torch.randn(X.shape[0],1)
X_test = torch.randn(30,dim)
Y_test = torch.sin(X_test.sum(dim=-1,keepdim=True)) + 0.01*torch.randn(X_test.shape[0],1)

ssgp = SSBQ(MaternSampler(2.5))
ssgp.set_training_data(X,Y)
ssgp.set_params(1.0*torch.ones(1),
                1.0*torch.ones(1),
                0.001*torch.ones(1),
                25)
optimizer = torch.optim.Adam(ssgp.parameters(),lr=1e-4)

def loss_fn(Y,Ypred):
    return ((Y-Ypred)**2).mean()
    
#%%
Y_pred = ssgp.prediction(X_test)[0]
print(torch.cat(
    [Y_test,Y_pred,torch.abs(Y_pred-Y_test)/torch.abs(Y_pred)]
    ,dim=-1).detach())

for i in range(1000):
    optimizer.zero_grad()
    loss = -ssgp.marginal_log_likelihood()
    loss.backward()
    optimizer.step()
    if i%1000 == 0:
        print(i,loss.item())
        Y_pred = ssgp.prediction(X_test)[0]
        print(loss_fn(Y_test,Y_pred))
        print('--')
print(torch.cat(
        [Y_test,Y_pred,torch.abs(Y_pred-Y_test)/torch.abs(Y_pred)]
        ,dim=-1).detach())
#X_new = torch.linspace(-500,500,101).reshape(-1,1)
#with torch.no_grad():
#    Y_new_mean,Y_new_var = ssgp.prediction(X_new)
#lower = Y_new_mean - 2*torch.sqrt(Y_new_var)
#upper = Y_new_mean + 2*torch.sqrt(Y_new_var)
#plt.plot(X.numpy(),Y.numpy(),'ro',alpha=0.01)
#plt.plot(X_new.numpy(),Y_new_mean.numpy(),'b-')
#plt.fill_between(X_new.flatten().numpy(),
#                 lower.flatten().numpy(),
#                 upper.flatten().numpy(),alpha=0.8)
