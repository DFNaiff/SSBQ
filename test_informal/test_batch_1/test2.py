# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../../src')

import math

import torch
import matplotlib.pyplot as plt

from ssgp import SSGP
from ssbq import SSBQ
from base_samplers import RBFSampler,MaternSampler

X = torch.randn(10000,10)
Y = torch.sin(X.sum(dim=-1,keepdim=True)) + 0.01*torch.randn(X.shape[0],1)
X_test = torch.randn(10,10)
Y_test = torch.sin(X_test.sum(dim=-1,keepdim=True)) + 0.01*torch.randn(X_test.shape[0],1)

ssgp = SSBQ(RBFSampler())
ssgp.set_training_data(X,Y)
ssgp.set_params(1.0*torch.ones(1),
                1.0*torch.ones(1),
                0.001*torch.ones(1),
                25)
optimizer = torch.optim.Adam(ssgp.parameters(),lr=1e-3)

def loss_fn(Y,Ypred):
    return ((Y-Ypred)**2).mean()
    
for i in range(10000):
    optimizer.zero_grad()
    loss = -ssgp.marginal_log_likelihood()
    loss.backward()
    optimizer.step()
    if i%100 == 0:
        print(i,loss.item())
        print(loss_fn(Y_test,ssgp.prediction(X_test)[0]))
        print('--')

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
